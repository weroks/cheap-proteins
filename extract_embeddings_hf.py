import torch
import os
import importlib
import time
import numpy as np

# REMOVE the problematic import:
# from datasets import get_worker_info
from datasets import (
    load_from_disk,
    Features,
    Value,
    Sequence,
)  # Keep other necessary imports


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings for a dataset using a CHEAP model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CHEAP_shorten_2_dim_128",
        help="Name of the model to load from cheap.pretrained.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./uniref50_subset_leq256",
        help="Directory containing the input dataset (Arrow format).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./uniref50_embeddings_128",
        help="Directory to save the processed dataset with embeddings.",
    )
    parser.add_argument(
        "--padded-length",
        type=int,
        default=128,
        help="Target length for the sequence dimension after padding.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Expected embedding dimension of the model output.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for model inference."
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,  # Defaulting to 4 as in the previous script
        help="Number of worker processes to use for .map(). Should ideally match the number of available/requested GPUs.",
    )

    args = parser.parse_args()
    return args


args = parse_args()
# --- Configuration ---
MODEL_NAME = args.model_name
INPUT_DATASET_DIR = args.input_dir
OUTPUT_DATASET_DIR = args.output_dir
PADDED_LENGTH = args.padded_length
EMBED_DIM = args.embed_dim
BATCH_SIZE = args.batch_size
# NUM_PROC should match the number of GPUs requested in SBATCH
NUM_GPUS_REQUESTED = args.num_proc
NUM_PROC = NUM_GPUS_REQUESTED
print(f"Configured for {NUM_PROC} processes, assuming {NUM_GPUS_REQUESTED} GPUs.")
print(f"Parent Process ID: {os.getpid()}")  # Log parent PID

os.environ["CHEAP_CACHE"] = "cache"

# --- Process-specific model storage ---
# Use PID as key since get_worker_info is unavailable
_model_pipeline_per_process = {}
_device_per_process = {}
_process_to_gpu_map = {}  # Map PID to GPU ID


def get_model_and_device(model_name: str):
    """
    Gets or initializes the model and device for the current worker process.
    Uses os.getpid() and a simple modulo assignment for GPU.
    """
    pid = os.getpid()

    if pid not in _model_pipeline_per_process:
        # Assign GPU based on PID modulo number of GPUs
        # This isn't guaranteed to be perfectly balanced like worker_id,
        # but works without get_worker_info
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus == 0:
            device = torch.device("cpu")
            print(f"[Process {pid}] No GPUs found, using CPU.")
        else:
            # Assign GPU round-robin based on *some* unique aspect of the process
            # Since PIDs can be reused, this isn't perfectly stable across runs
            # but should distribute across GPUs for a single run.
            # A more robust way would involve initializing a counter or similar
            # if datasets truly doesn't expose worker ID here.
            # Let's try a simple modulo approach first.
            # We need a way to assign 0, 1, 2, 3... uniquely.
            # Let's map PIDs encountered to GPU IDs sequentially.
            if pid not in _process_to_gpu_map:
                # Assign the next available GPU ID
                next_gpu_id = len(_process_to_gpu_map) % num_available_gpus
                _process_to_gpu_map[pid] = next_gpu_id
                print(f"[Process {pid}] First encounter. Assigning GPU {next_gpu_id}.")
            else:
                # This case shouldn't happen if initialization occurs once per process
                print(f"[Process {pid}] Re-encountered. This is unexpected.")

            device_id = _process_to_gpu_map[pid]
            device = torch.device(f"cuda:{device_id}")

        print(
            f"[Process {pid}] Initializing model '{model_name}' on device {device}..."
        )
        try:
            module = importlib.import_module("cheap.pretrained")
            model_func = getattr(module, model_name)
            pipeline = model_func(return_pipeline=True).to(device)
            _model_pipeline_per_process[pid] = pipeline
            _device_per_process[pid] = device
            print(f"[Process {pid}] Model initialized on {device}.")
        except Exception as e:  # Catch-all for initialization errors
            print(f"[Process {pid}] Error initializing model on {device}: {e}")
            # Clean up map if initialization failed? Maybe not necessary.
            raise e  # Re-raise the exception

    # Return the stored model and device for this PID
    return _model_pipeline_per_process[pid], _device_per_process[pid]


def process_and_pad_batch(batch):
    """
    Applies the model pipeline (on the process-specific GPU) to a batch,
    pads the results, and returns them.
    """
    pid = os.getpid()  # Get PID to retrieve correct model/device
    # print(f"[Process {pid}] Processing batch.") # Optional: verbose logging

    sequences = batch["text"]

    # Get the model and device assigned to this specific worker process (via PID)
    try:
        model_pipeline, device = get_model_and_device(MODEL_NAME)
    except Exception as e:
        print(f"[Process {pid}] Failed to get model/device. Error: {e}")
        # Handle error appropriately, maybe return an empty/error structure
        # For now, re-raise to see the error clearly
        raise e

    # --- Run Inference ---
    try:
        with torch.no_grad():
            # Optional: Move sequences to device if pipeline requires it
            # sequences_on_device = [s.to(device) for s in sequences]
            # raw_embeddings, raw_masks = model_pipeline(sequences_on_device)
            raw_embeddings, raw_masks = model_pipeline(
                sequences
            )  # Assuming pipeline handles device internally

        # --- Move results back to CPU ---
        raw_embeddings = raw_embeddings.cpu().numpy()
        raw_masks = raw_masks.cpu().numpy()
    except Exception as e:
        print(f"[Process {pid}] Error during model inference or CPU transfer: {e}")
        # Handle error, maybe skip batch or return error indicator
        raise e  # Re-raise for now

    # --- Padding ---
    try:
        current_len = raw_embeddings.shape[1]
        pad_width_embeddings = (
            (0, 0),
            (0, max(0, PADDED_LENGTH - current_len)),
            (0, 0),
        )
        padded_embeddings = np.pad(
            raw_embeddings,
            pad_width=pad_width_embeddings,
            mode="constant",
            constant_values=0.0,
        )

        pad_width_masks = ((0, 0), (0, max(0, PADDED_LENGTH - current_len)))
        padded_masks = np.pad(
            raw_masks, pad_width=pad_width_masks, mode="constant", constant_values=False
        )
    except Exception as e:
        print(f"[Process {pid}] Error during padding: {e}")
        raise e  # Re-raise for now

    return {"embeddings": padded_embeddings, "masks": padded_masks}


# --- Load and Process Dataset (Main part remains similar) ---
# ... (rest of your script: loading dataset, defining features, calling .map, saving) ...

print(f"Loading dataset from {INPUT_DATASET_DIR}...")
if not os.path.exists(INPUT_DATASET_DIR):
    raise FileNotFoundError(
        f"Input directory not found: {INPUT_DATASET_DIR}. Make sure you ran the previous script successfully."
    )

start_time = time.time()
original_dataset = load_from_disk(INPUT_DATASET_DIR)
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")
print("Original dataset structure:")
print(original_dataset)

processed_features = Features(
    {
        "embeddings": Sequence(
            feature=Sequence(feature=Value(dtype="float32"), length=EMBED_DIM),
            length=PADDED_LENGTH,
        ),
        "masks": Sequence(feature=Value(dtype="bool"), length=PADDED_LENGTH),
    }
)

print(f"\nProcessing dataset with batch size {BATCH_SIZE} and {NUM_PROC} processes...")
processing_start_time = time.time()

# Make sure the dataset object exists and has a 'train' split before accessing column_names
if "train" not in original_dataset:
    raise ValueError("Dataset does not contain a 'train' split.")
if not original_dataset["train"].column_names:
    raise ValueError("'train' split has no columns.")


processed_dataset = original_dataset.map(
    process_and_pad_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=NUM_PROC,
    remove_columns=original_dataset[
        "train"
    ].column_names,  # Ensure 'train' split exists
    features=processed_features,
)

print(
    f"Dataset processing finished in {time.time() - processing_start_time:.2f} seconds."
)
print("Processed dataset structure:")
print(processed_dataset)

# --- Save Processed Dataset ---
print(f"\nSaving processed dataset to {OUTPUT_DATASET_DIR} (Arrow format)...")
save_start_time = time.time()
processed_dataset.save_to_disk(
    OUTPUT_DATASET_DIR, num_proc=NUM_PROC
)  # Use multiple processes for saving too
print(f"Processed dataset saved in {time.time() - save_start_time:.2f} seconds.")
total_time = time.time() - start_time
print(f"\nTotal script execution time: {total_time:.2f} seconds.")
