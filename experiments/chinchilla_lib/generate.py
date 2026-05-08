"""Generate subcommand: measure FLOPs/params and emit training commands."""

from __future__ import annotations

import argparse
import json
import os
import sys

import re
from pathlib import Path

from experiments.chinchilla_lib.config import BATCH_SIZE, D_TARGETS, EVAL_N_SAMPLES
from experiments.chinchilla_lib.helpers import (
    _ckpt_dir, _get_grad_accum, _grid_meta_path, _lr_name,
    _measure_flops, _traj_path,
)


def _ensure_nbody_chain_config(task_id: str, config_dir: str = "configs/data") -> None:
    """Auto-generate a Hydra data config YAML for an nbody_chain task if missing.

    Pattern: nbody_chain_N{N}_b{body}_T{T}
    Written to: {config_dir}/{task_id}.yaml
    """
    m = re.match(r"nbody_chain_N(\d+)_b(\d+)_T([\d.]+)", task_id)
    if not m:
        return
    n, body, T = int(m.group(1)), int(m.group(2)), m.group(3)
    config_path = Path(config_dir) / f"{task_id}.yaml"
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"data_dir: outputs/data/{task_id}\n"
        f"n_atoms: {n}\n"
        f"body: {body}\n"
        f"T: {float(T)}\n"
        f"nbody_chain: true\n"
    )
    print(f"  [auto-config] Created {config_path}", file=sys.stderr)


def generate(args: argparse.Namespace) -> None:
    """Measure FLOPs/params and emit grid_meta.json + shell training commands."""
    from experiments.task_registry import TASK_REGISTRY

    tasks = [t.strip() for t in args.tasks.split(",")]
    archs = [a.strip() for a in args.archs.split(",")]
    sizes = [s.strip() for s in args.sizes.split(",")]
    lrs   = [float(lr) for lr in args.lrs.split(",")]
    chinchilla_dir = args.chinchilla_dir

    from experiments.task_registry import (
        _register_nbody_task, _register_nbody_chain_task,
    )

    for task_id in tasks:
        if task_id not in TASK_REGISTRY:
            if task_id.startswith("nbody_chain_"):
                _register_nbody_chain_task(task_id)
                _ensure_nbody_chain_config(task_id)
            elif task_id.startswith("nbody_"):
                _register_nbody_task(task_id)
            else:
                print(f"[WARN] Unknown task '{task_id}', skipping", file=sys.stderr)
                continue
        spec = TASK_REGISTRY[task_id]
        n_atoms = spec.n_atoms
        grid_meta: dict[str, dict] = {}

        is_nbody = task_id.startswith("nbody_")
        is_chain = task_id.startswith("nbody_chain_")
        print(f"\n=== Task: {task_id} ({spec.description}) ===", file=sys.stderr)

        for arch in archs:
            for size in sizes:
                key = f"{arch}/{size}"
                try:
                    n_params, fps = _measure_flops(arch, size, n_atoms, BATCH_SIZE,
                                                   aux_dist=is_nbody,
                                                   use_chain_pe=is_chain)
                except Exception as e:
                    print(f"[WARN] {key}: FLOPs failed: {e}", file=sys.stderr)
                    continue

                total_flops_D4 = fps * (D_TARGETS[-1] // BATCH_SIZE)
                print(
                    f"  {key:<30} params={n_params:>9,}  fps={fps:.2e}  "
                    f"total_FLOPs(D4)={total_flops_D4:.2e}",
                    file=sys.stderr,
                )
                grid_meta[key] = {
                    "arch": arch, "size": size,
                    "n_params": n_params, "flops_per_step": fps,
                }

        # Save grid_meta
        os.makedirs(os.path.join(chinchilla_dir, task_id), exist_ok=True)
        meta_path = _grid_meta_path(chinchilla_dir, task_id)
        with open(meta_path, "w") as f:
            json.dump(grid_meta, f, indent=2)
        print(f"  Grid meta saved: {meta_path}", file=sys.stderr)

        # Emit training commands to stdout.
        # One command per (arch, size, lr, D_budget) — 4 commands per (arch, size, lr).
        # Each command trains to exactly D_k steps with T_max=D_k so the cosine
        # schedule anneals properly to 0.1×LR by the end of that budget.
        #
        # D-budget data config resolution:
        #   If spec.chinchilla_data_configs has a key for this D (e.g. "D1"), use that
        #   data config (unique-structure dataset = true Chinchilla, 1 epoch).
        #   Otherwise fall back to spec.data_config (multi-epoch mode).
        # Resolve D-budget grid: --d_targets overrides module-level D_TARGETS.
        # Each D budget trains for exactly D/batch_size steps (1 epoch, no repetition).
        if getattr(args, "d_targets", None):
            _d_targets = [int(x) for x in args.d_targets.split(",")]
        else:
            _d_targets = D_TARGETS
        _epochs = getattr(args, "epochs", 1)
        _eval_n_samples = getattr(args, "eval_n_samples", EVAL_N_SAMPLES)
        _d_names = [f"D{i+1}" for i in range(len(_d_targets))]
        _d_steps = [_epochs * d // BATCH_SIZE for d in _d_targets]
        _base_eval_every = _d_steps[0]  # eval at every D1-sized chunk so all budgets are captured

        dc_map = spec.chinchilla_data_configs or {}
        for arch in archs:
            for size in sizes:
                key = f"{arch}/{size}"
                if key not in grid_meta:
                    continue
                ga = _get_grad_accum(arch, size, n_atoms)
                micro_bs = BATCH_SIZE // ga
                for lr in lrs:
                    for d_name, total_steps, d_target in zip(_d_names, _d_steps, _d_targets):
                        data_cfg = dc_map.get(d_name, spec.data_config)
                        ckpt = _ckpt_dir(chinchilla_dir, task_id, arch, size, lr, d_name)
                        traj = _traj_path(chinchilla_dir, task_id, arch, size, lr, d_name)
                        eval_every = min(_base_eval_every, total_steps)
                        cmd = (
                            f"uv run python experiments/train.py"
                            f" data={data_cfg}"
                            f" model={arch}"
                            f" model.size={size}"
                            f" train.max_steps={total_steps}"
                            f" train.max_train_samples={d_target}"
                            f" chinchilla.D_nominal={d_target}"
                            f" train.lr={lr}"
                            f" train.batch_size={BATCH_SIZE}"
                            f" train.grad_accum_steps={ga}"
                            f" train.warmup_fraction=0"
                            f" train.min_lr_ratio=0.01"
                            f" eval.every_n_steps={eval_every}"
                            f" eval.n_samples={_eval_n_samples}"
                            f" checkpoint.dir={ckpt}"
                            f" chinchilla.enabled=true"
                            f" chinchilla.task_id={task_id}"
                            f" chinchilla.trajectory_path={traj}"
                            f" chinchilla.size={size}"
                            f" logging.enabled={'true' if args.wandb else 'false'}"
                            f" hydra.run.dir={ckpt}"
                        )
                        if ga > 1:
                            print(f"# grad_accum={ga}, micro_batch={micro_bs}, effective_batch={BATCH_SIZE}", file=sys.stderr)
                        print(cmd)
