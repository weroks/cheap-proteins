from datasets import IterableDataset
import torch
from torch.utils.data import Dataset
from pathlib import Path


class HFDataset(Dataset):
    """Wrapper for Hugging Face datasets to conform to __getitem__ returning sequence."""

    def __init__(self, hf_dataset, sequence_column="sequence"):
        self.hf_dataset = hf_dataset
        self.sequence_column = sequence_column

        if self.sequence_column not in self.hf_dataset.column_names:
            raise ValueError(
                f"Sequence column '{self.sequence_column}' not found in dataset features: {self.hf_dataset.column_names}"
            )

    def __len__(self):
        if isinstance(self.hf_dataset, IterableDataset):
            raise TypeError("Cannot get length of IterableDataset.")
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        return item[self.sequence_column]


class FastaDataset(Dataset):
    """Custom PyTorch Dataset for lists of protein sequences."""

    def __init__(self, filepath: str) -> None:
        self.sequences = self.load_fasta(filepath)

    def load_fasta(self, filepath: str) -> list[str]:
        """Reads a FASTA file and returns a list of protein sequences.

        Args:
            filepath (str): Path to the FASTA file.

        Returns:
            list: A list of protein sequences as strings.
        """
        sequences = []

        with Path(filepath).open() as file:
            current_sequence = []

            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if current_sequence:
                        sequences.append("".join(current_sequence))
                        current_sequence = []
                else:
                    current_sequence.append(line)

            if current_sequence:
                sequences.append("".join(current_sequence))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]
