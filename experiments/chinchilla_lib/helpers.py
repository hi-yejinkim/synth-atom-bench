"""Path helpers, gradient accumulation, FLOPs measurement, and completion checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

from experiments.chinchilla_lib.config import (
    BATCH_SIZE, D_STEPS, EVAL_EVERY, EVAL_N_SAMPLES, LR_NAMES, _GA_FALLBACK_N50,
)


def load_lower_bound(data_dir: str, metric: str = "W2_energy", n_eval: int = 10000) -> dict | None:
    """Load precomputed lower bound for a dataset.

    Looks for lower_bound.json in data_dir (written by chain_metric_lower_bound.py).
    Returns {"mean": float, "std": float} or None if not found.

    Args:
        data_dir : path to the dataset directory (containing train.npz)
        metric   : which metric key to load (e.g. "W2_E_total", "W2_bond_len")
        n_eval   : evaluation sample count to look up (default 10000)
    """
    p = Path(data_dir) / "lower_bound.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        by_n = payload.get("by_n", {})
        n_str = str(n_eval)
        # fall back to nearest available n if exact match missing
        if n_str not in by_n:
            available = sorted(int(k) for k in by_n)
            if not available:
                return None
            closest = min(available, key=lambda x: abs(x - n_eval))
            n_str = str(closest)
        entry = by_n[n_str].get(metric)
        if entry is None:
            return None
        return {
            **entry,                                      # mean, std
            "ref_mode": payload.get("ref_mode", "?"),
            "n_actual": int(n_str),                       # actual n used (may differ from n_eval)
        }
    except Exception:
        return None


def task_id_to_data_dir(task_id: str, base: str = "outputs/data") -> str:
    """Derive the dataset directory path from a task_id.

    Convention: task_id == directory name under outputs/data/.
    e.g. "nbody_chain_N15_b4_T1.0" → "outputs/data/nbody_chain_N15_b4_T1.0"
    """
    return os.path.join(base, task_id)


def _metric_key(task_id: str) -> str:
    """Return the trajectory.jsonl metric key for a given task.

    Most tasks use 'violation_rate' (bounded [0,1]).
    N-body chain tasks use 'energy_w2' (W2 matches lower bound baseline).
    Plain N-body tasks also use 'energy_w2' for consistency (both are logged).
    """
    if task_id.startswith("nbody_"):  # covers both nbody_ and nbody_chain_
        return "energy_w2"
    return "violation_rate"


def _metric_is_bounded(task_id: str) -> bool:
    """Return True if the task's metric is bounded in [0,1]."""
    return not task_id.startswith("nbody_")


def _lr_name(lr: float) -> str:
    return LR_NAMES.get(lr, f"lr{lr:.0e}")


def _ckpt_dir(chinchilla_dir: str, task: str, arch: str, size: str, lr: float,
              d_name: str = "") -> str:
    """Return checkpoint directory.

    Directory layout (D-budget-separated design):
        outputs/chinchilla/{task}/{arch}/{size}/{lr_name}/{d_name}/

    Each D budget gets its own run with T_max = D_k_steps and a schedule
    that anneals to 0.1×LR exactly at step D_k.  This ensures snapshots
    at different data scales are all schedule-optimal (no cosine mismatch).
    """
    base = os.path.join(chinchilla_dir, task, arch, size, _lr_name(lr))
    return os.path.join(base, d_name) if d_name else base


def _traj_path(chinchilla_dir: str, task: str, arch: str, size: str, lr: float,
               d_name: str = "") -> str:
    return os.path.join(_ckpt_dir(chinchilla_dir, task, arch, size, lr, d_name),
                        "trajectory.jsonl")


def _grid_meta_path(chinchilla_dir: str, task: str) -> str:
    return os.path.join(chinchilla_dir, task, "grid_meta.json")


def _results_path(chinchilla_dir: str, task: str) -> str:
    return os.path.join(chinchilla_dir, task, "results.json")


def _fits_path(chinchilla_dir: str, task: str) -> str:
    return os.path.join(chinchilla_dir, task, "fits.json")


def _fits_approach1_path(chinchilla_dir: str, task: str) -> str:
    return os.path.join(chinchilla_dir, task, "fits_approach1.json")


def _csv_path(chinchilla_dir: str, task: str) -> str:
    return os.path.join(chinchilla_dir, task, "scaling_results.csv")


def _get_grad_accum(arch: str, size: str, n_atoms: int,
                    batch_size: int = BATCH_SIZE) -> int:
    """Determine grad_accum_steps for a given (arch, size, n_atoms) combo.

    Returns 1 (no accumulation) if the full batch fits in memory.
    """
    if n_atoms < 30:
        return 1  # small-N tasks fit easily
    size_idx = int(size.split("_")[1]) if "_" in size else 0
    fallback = _GA_FALLBACK_N50.get(arch, [])
    ga = 1
    for min_idx, accum in fallback:
        if size_idx >= min_idx:
            ga = accum
    return min(ga, batch_size)  # never exceed batch_size


def _measure_flops(arch: str, size: str, n_atoms: int, batch_size: int,
                   aux_dist: bool = False, use_chain_pe: bool = False) -> tuple[int, int]:
    """Instantiate model, count params and FLOPs per step.

    Args:
        aux_dist: If True, add auxiliary pairwise distance loss FLOPs
            (two cdist calls + MSE on N×N matrices). Used for n-body tasks.
        use_chain_pe: If True, instantiate transformer with chain positional
            encoding enabled (nbody_chain tasks). Ensures param count in
            grid_meta.json matches the actual trained model.

    Returns:
        (n_params, flops_per_step)
    """
    import torch
    from experiments.model_registry import MODEL_REGISTRY, SIZE_PRESETS, MODEL_DEFAULTS

    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch: {arch}")
    if size not in SIZE_PRESETS.get(arch, {}):
        raise ValueError(f"Size '{size}' not in SIZE_PRESETS['{arch}']")

    kwargs = dict(MODEL_DEFAULTS.get(arch, {}))
    kwargs.update(SIZE_PRESETS[arch][size])
    if use_chain_pe and arch == "transformer":
        kwargs["use_chain_pe"] = True
    model = MODEL_REGISTRY[arch](**kwargs)
    n_params = sum(p.numel() for p in model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    try:
        from torch.utils.flop_counter import FlopCounterMode
        # Measure forward-only FLOPs with progressively smaller batch sizes to
        # avoid OOM (FLOPs scale linearly with batch_size so we can rescale).
        # We do NOT call backward() inside FlopCounterMode; instead we multiply
        # by 3 afterward (forward=1x, backward≈2x) to get per-step training FLOPs.
        flops_fwd_1 = None
        for probe_batch in [1, 4, 16]:
            if probe_batch > batch_size:
                continue
            try:
                x_probe = torch.randn(probe_batch, n_atoms, 3, device=device)
                t_probe = torch.rand(probe_batch, device=device)
                with torch.no_grad():
                    with FlopCounterMode(display=False) as counter:
                        model(x_probe, t_probe)
                flops_fwd_1 = int(counter.get_total_flops()) / probe_batch
                break
            except Exception:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
        if flops_fwd_1 is not None:
            # Add auxiliary pairwise distance loss FLOPs for n-body tasks:
            # Two cdist calls on (1, N, 3) → 2 × N × N × 3 multiply-adds each
            if aux_dist:
                flops_fwd_1 += 2 * (2 * n_atoms * n_atoms * 3)
            # Scale to full batch_size and account for forward + backward
            flops_per_step = int(flops_fwd_1 * batch_size * 3)
        else:
            flops_per_step = 6 * n_params * batch_size
    except Exception:
        flops_per_step = 6 * n_params * batch_size

    return n_params, flops_per_step


def _is_complete(traj_path: str, expected_steps: int | None = None,
                 eval_every: int = EVAL_EVERY) -> bool:
    """Check if a D-budget run has a terminal trajectory point.

    Args:
        traj_path: path to trajectory.jsonl
        expected_steps: the max_steps of the corresponding run (D_k_steps).
                        A run is complete if its last logged step is within
                        eval_every of expected_steps.
        eval_every: eval frequency used for this run (default: module-level EVAL_EVERY).
    """
    if not os.path.exists(traj_path):
        return False
    try:
        last = None
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        if last is None:
            return False
        if expected_steps is not None:
            # When eval_every >= expected_steps (tiny D-budgets like D1=3 steps),
            # the slack formula gives threshold=0, accepting any partial run as complete.
            # Require exact match instead.
            if eval_every >= expected_steps:
                threshold = expected_steps
            else:
                threshold = expected_steps - eval_every
        else:
            threshold = D_STEPS[-1] - eval_every
        return last.get("step", 0) >= threshold
    except Exception:
        return False
