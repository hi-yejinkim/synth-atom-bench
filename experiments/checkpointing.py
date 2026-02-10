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
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    config: dict[str, Any]


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
        model_state_dict=data["model_state_dict"],
        optimizer_state_dict=data["optimizer_state_dict"],
        config=data["config"],
    )


class CheckpointManager:
    """Manages best.pt and latest.pt in a directory."""

    def __init__(self, checkpoint_dir: str | Path):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._best_clash_rate = float("inf")
        # Restore best from existing checkpoint if present
        best_path = self._dir / "best.pt"
        if best_path.exists():
            state = load_checkpoint(best_path)
            self._best_clash_rate = state.best_clash_rate

    @property
    def best_clash_rate(self) -> float:
        return self._best_clash_rate

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        clash_rate: float,
        config: dict[str, Any],
    ) -> None:
        state = CheckpointState(
            epoch=epoch,
            step=step,
            best_clash_rate=min(clash_rate, self._best_clash_rate),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            config=config,
        )
        # Always save latest
        save_checkpoint(state, self._dir / "latest.pt")
        # Save best if improved
        if clash_rate < self._best_clash_rate:
            self._best_clash_rate = clash_rate
            state.best_clash_rate = clash_rate
            save_checkpoint(state, self._dir / "best.pt")

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
