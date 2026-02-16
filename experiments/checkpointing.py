"""Checkpoint save/load with atomic writes."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointState:
    """All state needed to resume training."""

    epoch: int
    step: int
    best_clash_rate: float
    best_gr_distance: float
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    config: dict[str, Any]
    best_bond_violation_rate: float = float("inf")
    best_nonbonded_clash_rate: float = float("inf")


def save_checkpoint(state: CheckpointState, path: str | Path) -> None:
    """Atomic save: write to temp file then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        torch.save(
            {
                "epoch": state.epoch,
                "step": state.step,
                "best_clash_rate": state.best_clash_rate,
                "best_gr_distance": state.best_gr_distance,
                "best_bond_violation_rate": state.best_bond_violation_rate,
                "best_nonbonded_clash_rate": state.best_nonbonded_clash_rate,
                "model_state_dict": state.model_state_dict,
                "optimizer_state_dict": state.optimizer_state_dict,
                "config": state.config,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
    except BaseException:
        os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    else:
        os.close(fd)


def load_checkpoint(path: str | Path, device: str = "cpu") -> CheckpointState:
    """Load checkpoint from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = torch.load(path, map_location=device, weights_only=False)
    return CheckpointState(
        epoch=data["epoch"],
        step=data["step"],
        best_clash_rate=data["best_clash_rate"],
        best_gr_distance=data.get("best_gr_distance", float("inf")),
        model_state_dict=data["model_state_dict"],
        optimizer_state_dict=data["optimizer_state_dict"],
        config=data["config"],
        best_bond_violation_rate=data.get("best_bond_violation_rate", float("inf")),
        best_nonbonded_clash_rate=data.get("best_nonbonded_clash_rate", float("inf")),
    )


class CheckpointManager:
    """Manages best.pt and latest.pt in a directory."""

    def __init__(self, checkpoint_dir: str | Path):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._best_clash_rate = float("inf")
        self._best_gr_distance = float("inf")
        self._best_bond_violation_rate = float("inf")
        self._best_nonbonded_clash_rate = float("inf")
        # Restore best from existing checkpoint if present
        best_path = self._dir / "best.pt"
        if best_path.exists():
            state = load_checkpoint(best_path)
            self._best_clash_rate = state.best_clash_rate
            self._best_gr_distance = state.best_gr_distance
            self._best_bond_violation_rate = state.best_bond_violation_rate
            self._best_nonbonded_clash_rate = state.best_nonbonded_clash_rate

    @property
    def best_clash_rate(self) -> float:
        return self._best_clash_rate

    @property
    def best_gr_distance(self) -> float:
        return self._best_gr_distance

    @property
    def best_bond_violation_rate(self) -> float:
        return self._best_bond_violation_rate

    @property
    def best_nonbonded_clash_rate(self) -> float:
        return self._best_nonbonded_clash_rate

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        clash_rate: float,
        config: dict[str, Any],
        gr_distance: float = float("inf"),
        bond_violation_rate: float = float("inf"),
        nonbonded_clash_rate: float = float("inf"),
    ) -> None:
        best_cr = min(clash_rate, self._best_clash_rate)
        best_grd = min(gr_distance, self._best_gr_distance)
        best_bvr = min(bond_violation_rate, self._best_bond_violation_rate)
        best_ncr = min(nonbonded_clash_rate, self._best_nonbonded_clash_rate)
        state = CheckpointState(
            epoch=epoch,
            step=step,
            best_clash_rate=best_cr,
            best_gr_distance=best_grd,
            best_bond_violation_rate=best_bvr,
            best_nonbonded_clash_rate=best_ncr,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            config=config,
        )
        # Always save latest
        save_checkpoint(state, self._dir / "latest.pt")
        # Save best if g(r) distance improved (primary metric)
        if gr_distance < self._best_gr_distance:
            save_checkpoint(state, self._dir / "best.pt")
        # Always update running minimums
        self._best_clash_rate = best_cr
        self._best_gr_distance = best_grd
        self._best_bond_violation_rate = best_bvr
        self._best_nonbonded_clash_rate = best_ncr

    def load_latest(self, device: str = "cpu") -> CheckpointState | None:
        path = self._dir / "latest.pt"
        if not path.exists():
            return None
        return load_checkpoint(path, device)

    def load_best(self, device: str = "cpu") -> CheckpointState | None:
        path = self._dir / "best.pt"
        if not path.exists():
            return None
        return load_checkpoint(path, device)
