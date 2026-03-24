"""Run subcommand: execute training commands with optional GPU parallelism."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading

from experiments.chinchilla_lib.config import BATCH_SIZE, D_TARGETS
from experiments.chinchilla_lib.helpers import (
    _ckpt_dir, _get_grad_accum, _grid_meta_path, _is_complete,
    _lr_name, _traj_path,
)


def run(args: argparse.Namespace) -> None:
    """Execute training commands from generate, with optional GPU parallelism."""
    from experiments.task_registry import TASK_REGISTRY, _register_nbody_task

    tasks = [t.strip() for t in args.tasks.split(",")]
    archs = [a.strip() for a in args.archs.split(",")]
    sizes = [s.strip() for s in args.sizes.split(",")]
    lrs   = [float(lr) for lr in args.lrs.split(",")]
    chinchilla_dir = args.chinchilla_dir
    n_gpus = args.n_gpus

    # Build command list, skip already-complete runs
    commands: list[tuple[str, str]] = []  # (cmd, traj_path)
    for task_id in tasks:
        if task_id not in TASK_REGISTRY:
            if task_id.startswith("nbody_"):
                _register_nbody_task(task_id)
            else:
                continue
        spec = TASK_REGISTRY[task_id]
        meta_path = _grid_meta_path(chinchilla_dir, task_id)
        if not os.path.exists(meta_path):
            print(f"[WARN] grid_meta.json not found for {task_id}. Run generate first.", file=sys.stderr)
            continue
        with open(meta_path) as f:
            grid_meta = json.load(f)

        if getattr(args, "d_targets", None):
            _d_targets = [int(x) for x in args.d_targets.split(",")]
        else:
            _d_targets = D_TARGETS
        _epochs = getattr(args, "epochs", 1)
        _d_names = [f"D{i+1}" for i in range(len(_d_targets))]
        _d_steps = [_epochs * d // BATCH_SIZE for d in _d_targets]
        _base_eval_every = _d_steps[0]

        n_atoms = spec.n_atoms
        dc_map = spec.chinchilla_data_configs or {}
        for arch in archs:
            for size in sizes:
                if f"{arch}/{size}" not in grid_meta:
                    continue
                ga = _get_grad_accum(arch, size, n_atoms)
                for lr in lrs:
                    for d_name, total_steps, d_target in zip(_d_names, _d_steps, _d_targets):
                        data_cfg = dc_map.get(d_name, spec.data_config)
                        traj = _traj_path(chinchilla_dir, task_id, arch, size, lr, d_name)
                        ckpt = _ckpt_dir(chinchilla_dir, task_id, arch, size, lr, d_name)
                        eval_every = min(_base_eval_every, total_steps)
                        if _is_complete(traj, total_steps, eval_every=eval_every):
                            print(f"[SKIP] {task_id}/{arch}/{size}/{_lr_name(lr)}/{d_name}")
                            continue
                        cmd = (
                            f"uv run python experiments/train.py"
                            f" data={data_cfg}"
                            f" model={arch}"
                            f" model.size={size}"
                            f" train.max_steps={total_steps}"
                            f" train.max_train_samples={d_target}"
                            f" train.lr={lr}"
                            f" train.batch_size={BATCH_SIZE}"
                            f" train.grad_accum_steps={ga}"
                            f" train.warmup_fraction=0"
                            f" train.min_lr_ratio=0.01"
                            f" eval.every_n_steps={eval_every}"
                            f" eval.n_samples=1000"
                            f" checkpoint.dir={ckpt}"
                            f" chinchilla.enabled=true"
                            f" chinchilla.task_id={task_id}"
                            f" chinchilla.trajectory_path={traj}"
                            f" chinchilla.size={size}"
                            f" logging.enabled={'true' if args.wandb else 'false'}"
                            f" hydra.run.dir={ckpt}"
                        )
                        commands.append((cmd, traj))

    print(f"Total runs to execute: {len(commands)}", file=sys.stderr)
    if not commands:
        return

    if n_gpus <= 1:
        for i, (cmd, _) in enumerate(commands):
            print(f"\n[{i+1}/{len(commands)}] {cmd[:80]}...", file=sys.stderr)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            subprocess.run(cmd, shell=True, env=env)
    else:
        # Parallel: one worker per GPU
        q: queue.Queue = queue.Queue()
        for item in commands:
            q.put(item)

        def worker(gpu_id: int) -> None:
            while True:
                try:
                    cmd, _ = q.get(timeout=1)
                except queue.Empty:
                    break
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                print(f"[GPU {gpu_id}] {cmd[:80]}...", file=sys.stderr)
                subprocess.run(cmd, shell=True, env=env)
                q.task_done()

        threads = [threading.Thread(target=worker, args=(i % n_gpus,), daemon=True)
                   for i in range(n_gpus)]
        for t in threads:
            t.start()
        q.join()
