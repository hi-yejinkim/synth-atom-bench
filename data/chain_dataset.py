"""PyTorch dataset for self-avoiding chain samples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ChainDataset(Dataset):
    """Dataset of chain configurations loaded from .npz files.

    Loads entire dataset into memory as float32 tensors at init time.
    Positions are shifted to [0, box_size] for training code compatibility.
    """

    def __init__(self, path: str):
        data = np.load(path)
        positions = data["positions"].astype(np.float32)
        self.box_size = float(data["box_size"])
        self.radius = float(data["radius"])
        self.bond_length = float(data["bond_length"])

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
