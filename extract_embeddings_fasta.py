import importlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cheap.datasets.custom import FastaDataset

os.environ["CHEAP_CACHE"] = "cache"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings from FASTA file.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="../counterfactual-proteomics/data/fluorescence/v0.1/fluorescence.fasta",
        help="Path to the input FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../counterfactual-proteomics/data/fluorescence/v0.5",
        help="Path to save the extracted embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing sequences.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CHEAP_shorten_1_dim_16",
        help=(
            "Name of the model to use for embedding extraction. "
            "Example: CHEAP_shorten_1_dim_16, CHEAP_shorten_2_dim_256, etc."
        ),
    )
    return parser.parse_args()


def get_model(model_name: str):
    """Dynamically imports and returns the specified model.

    Args:
        model_name: Name of the model to import.

    Returns:
        The model function.

    Raises:
        ValueError: If the model is not found.
    """
    try:
        # Dynamically import the model from the "cheap.pretrained" module
        module = importlib.import_module("cheap.pretrained")
        model_func = getattr(module, model_name)
        return model_func(return_pipeline=True)

    except (ImportError, AttributeError) as e:
        raise ValueError(f"Model '{model_name}' not found in 'cheap.pretrained'") from e


def run(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Dynamically load the model
    pipeline = get_model(args.model_name)
    print(f"Pipeline loaded for model: {args.model_name}")

    # Create Dataset and DataLoader
    dataset = FastaDataset(input_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Ensure the order of sequences is preserved
    )

    embeddings = []
    masks = []

    for batch in tqdm(dataloader):
        emb, mask = pipeline(batch)
        embeddings.append(emb.cpu())
        masks.append(mask.cpu())

    # Concatenate all embeddings and masks
    embeddings = torch.cat(embeddings, dim=0)
    masks = torch.cat(masks, dim=0)

    print(embeddings.shape)
    print(masks.shape)

    # Save the embeddings and mask
    torch.save(embeddings, output_path / "embeddings.pt")
    torch.save(masks, output_path / "mask.pt")
    print(f"Embeddings saved to {output_path / 'embeddings.pt'}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
