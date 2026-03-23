"""PyTorch dataset for VSEPR local geometry samples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class VSEPRDataset(Dataset):
    """Dataset of VSEPR molecule configurations loaded from .npz files.

    Positions are centered at the molecular centroid at generation time.
    For training compatibility, we shift them to [0, box_size] here.
    """

    def __init__(self, path: str, max_samples: int | None = None):
        data = np.load(path)
        positions = data["positions"].astype(np.float32)
        if max_samples is not None:
            positions = positions[:max_samples]
        self.box_size = float(data["box_size"])
        self.radius = float(data["radius"])
        self.orbital_type = str(data["orbital_type"].item().decode()
                                 if hasattr(data["orbital_type"].item(), "decode")
                                 else data["orbital_type"].item())
        self.n_lonepairs = int(data["n_lonepairs"])
        self.has_pi = bool(data["has_pi"])
        self.bond_range = tuple(float(x) for x in data["bond_range"])
        self.target_angle = float(data["target_angle"])

        # Shift from origin-centered to [0, box_size] for training compatibility
        self.positions = torch.from_numpy(positions + self.box_size / 2)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions": self.positions[idx],
            "radius": self.radius,
            "box_size": self.box_size,
        }
