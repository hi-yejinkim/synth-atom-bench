"""PyTorch dataset for polymer sequence/global geometry samples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Dataset of polymer configurations loaded from .npz files.

    Positions are shifted to [0, box_size] for training code compatibility.
    Contact pairs and fragment metadata are preserved as numpy arrays.
    """

    def __init__(self, path: str, max_samples: int | None = None):
        data = np.load(path)
        positions = data["positions"].astype(np.float32)
        if max_samples is not None:
            positions = positions[:max_samples]
        self.box_size = float(data["box_size"])
        self.radius = float(data["radius"])
        self.bond_length = float(data["bond_length"])
        self.n_fragments = int(data["n_fragments"])
        self.fragment_size = int(data["fragment_size"])
        self.fragment_ids = data["fragment_ids"]           # (N,) int32
        self.contact_pairs = data["contact_pairs"]         # (n_contacts, 2) int32
        self.contact_distance = float(data["contact_distance"])
        self.polymer_type = str(
            data["polymer_type"].item().decode()
            if hasattr(data["polymer_type"].item(), "decode")
            else data["polymer_type"].item()
        )

        # Shift from origin-centered to [0, box_size] for training compatibility
        self.positions = torch.from_numpy(positions + self.box_size / 2)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions": self.positions[idx],
            "radius": self.radius,
            "box_size": self.box_size,
            "bond_length": self.bond_length,
        }
