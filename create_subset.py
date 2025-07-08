"""Script to remove sequences longer than MAX_LENGTH characters from the 'agemagician/uniref50' dataset."""
from datasets import load_dataset
import os

MAX_LENGTH = 256

output_dir = "./uniref50_subset_leq256"

print("Loading dataset 'agemagician/uniref50'...")
full_dataset = load_dataset("agemagician/uniref50")
print(f"Dataset loaded. Splits found: {list(full_dataset.keys())}")
print("Original dataset structure:")
print(full_dataset)


def filter_by_length(example):
    """Returns True if the sequence length is less than or equal to MAX_LENGTH."""
    return len(example["text"]) <= MAX_LENGTH


print(f"\nFiltering dataset to keep sequences with length <= {MAX_LENGTH}...")
num_processors = os.cpu_count()
print(f"Using {num_processors} processors for filtering.")

filtered_dataset = full_dataset.filter(filter_by_length, num_proc=num_processors)

print("Filtering complete.")
print("Filtered dataset structure:")
print(filtered_dataset)

print(f"\nSaving filtered dataset to '{output_dir}'...")
filtered_dataset.save_to_disk(output_dir)
print(f"Filtered dataset successfully saved to '{output_dir}'.")

