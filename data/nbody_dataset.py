"""PyTorch dataset for n-body MCMC energy distribution samples."""

import numpy as np
import torch
from torch.utils.data import Dataset


class NBodyDataset(Dataset):
    """Dataset of n-body configurations from MCMC sampling.

    Loads positions and energy metadata from .npz files produced by
    data/generate_nbody.py.
    """

    def __init__(self, path: str, max_samples: int | None = None):
        data = np.load(path, allow_pickle=True)
        positions = data["positions"].astype(np.float32)
        if max_samples is not None:
            positions = positions[:max_samples]
        self.positions = torch.from_numpy(positions)
        self.box_size = float(data["box_size"])
        self.radius = float(data["sigma"]) / 2.0  # clash radius = σ/2

        # Energy reference for W2 metric
        energies = data["energies"].astype(np.float32)
        if max_samples is not None:
            energies = energies[:max_samples]
        self.energies = torch.from_numpy(energies)

        # Potential parameters (needed to compute energies of generated samples)
        self.body = int(data["body"])
        self.T = float(data["T"])
        self.sigma = float(data["sigma"])
        self.epsilon = float(data["epsilon"])
        self.nu = float(data.get("nu", 1.0))
        self.mu = float(data.get("mu", 0.2))
        self.bc = str(data.get("boundary", "pbc"))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "positions": self.positions[idx],
            "box_size": self.box_size,
        }
