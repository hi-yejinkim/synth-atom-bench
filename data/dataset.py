"""PyTorch dataset for hard sphere packing samples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class HardSphereDataset(Dataset):
    """Dataset of hard sphere configurations loaded from .npz files.

    Loads entire dataset into memory as float32 tensors at init time.
    """

    def __init__(self, path: str):
        data = np.load(path)
        self.positions = torch.from_numpy(data["positions"].astype(np.float32))
        self.radius = float(data["radius"])
        self.box_size = float(data["box_size"])

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions": self.positions[idx],
            "radius": self.radius,
            "box_size": self.box_size,
        }
