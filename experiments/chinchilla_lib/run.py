"""Run subcommand: execute training commands with optional GPU parallelism."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading

from experiments.chinchilla_lib.config import BATCH_SIZE, D_TARGETS, EVAL_N_SAMPLES
from experiments.chinchilla_lib.helpers import (
    _ckpt_dir, _get_grad_accum, _grid_meta_path, _is_complete,
    _lr_name, _metric_key, _traj_path,
)


def _run_commands(commands: list[tuple[str, str]], n_gpus: int, wandb: bool) -> None:
    """Execute (cmd, label) list across n_gpus GPUs."""
    if not commands:
        return
    if n_gpus <= 1:
        for i, (cmd, _) in enumerate(commands):
            print(f"\n[{i+1}/{len(commands)}] {cmd[:100]}...", file=sys.stderr)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            subprocess.run(cmd, shell=True, env=env)
    else:
        q: queue.Queue = queue.Queue()
        for item in commands:
            q.put(item)

        def worker(gpu_id: int) -> None:
            while True:
                try:
                    cmd, label = q.get(timeout=1)
                except queue.Empty:
                    break
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                print(f"[GPU {gpu_id}] {label}", file=sys.stderr)
                subprocess.run(cmd, shell=True, env=env)
                q.task_done()

        threads = [threading.Thread(target=worker, args=(i % n_gpus,), daemon=True)
                   for i in range(n_gpus)]
        for t in threads:
            t.start()
        q.join()


def _best_lr_from_d1(chinchilla_dir: str, task_id: str, arch: str, size: str,
                     lrs: list[float], d1_name: str) -> float | None:
    """Read D1 trajectories for all LRs, return the LR with best terminal metric."""
    mkey = _metric_key(task_id)
    best_lr, best_val = None, float("inf")
    for lr in lrs:
        traj_path = _traj_path(chinchilla_dir, task_id, arch, size, lr, d1_name)
        if not os.path.exists(traj_path):
            continue
        try:
            with open(traj_path) as f:
                points = [json.loads(l) for l in f if l.strip()]
            if not points:
                continue
            val = float(points[-1].get(mkey, float("inf")))
            if val < best_val:
                best_val, best_lr = val, lr
        except Exception:
            pass
    return best_lr


def _build_cmd(
    data_cfg: str, arch: str, size: str, total_steps: int, d_target: int,
    lr: float, ga: int, eval_every: int, eval_n_samples: int,
    ckpt: str, task_id: str, traj: str, wandb: bool,
) -> str:
    return (
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
        f" eval.n_samples={eval_n_samples}"
        f" checkpoint.dir={ckpt}"
        f" chinchilla.enabled=true"
        f" chinchilla.task_id={task_id}"
        f" chinchilla.D_nominal={d_target}"
        f" chinchilla.trajectory_path={traj}"
        f" chinchilla.size={size}"
        f" logging.enabled={'true' if wandb else 'false'}"
        f" hydra.run.dir={ckpt}"
    )


def run(args: argparse.Namespace) -> None:
    """Execute training commands from generate, with optional GPU parallelism.

    LR search strategy (--lr_warmup, default=True):
      Phase 1 — run all LRs for D1 only (cheapest budget).
      Phase 2 — for each (arch, size), pick best LR from D1 results,
                 then run that single LR for D2+ budgets.
      Savings: (n_lrs-1) × (n_D-1) runs skipped, e.g. 2LR×2D → 3 runs instead of 4.
    """
    from experiments.task_registry import TASK_REGISTRY, _register_nbody_task, _register_nbody_chain_task

    tasks = [t.strip() for t in args.tasks.split(",")]
    archs = [a.strip() for a in args.archs.split(",")]
    sizes = [s.strip() for s in args.sizes.split(",")]
    lrs   = [float(lr) for lr in args.lrs.split(",")]
    chinchilla_dir = args.chinchilla_dir
    n_gpus = args.n_gpus
    lr_warmup = getattr(args, "lr_warmup", True)
    eval_n_samples = getattr(args, "eval_n_samples", EVAL_N_SAMPLES)

    # ── Shared setup per task ─────────────────────────────────────────────
    task_configs: list[dict] = []
    for task_id in tasks:
        if task_id not in TASK_REGISTRY:
            if task_id.startswith("nbody_chain_"):
                _register_nbody_chain_task(task_id)
            elif task_id.startswith("nbody_"):
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

        task_configs.append(dict(
            task_id=task_id, spec=spec, grid_meta=grid_meta,
            d_targets=_d_targets, d_names=_d_names, d_steps=_d_steps,
            base_eval_every=_base_eval_every,
        ))

    if not task_configs:
        return

    # ── Phase 1: D1 with all LRs ──────────────────────────────────────────
    phase1: list[tuple[str, str]] = []
    for tc in task_configs:
        task_id = tc["task_id"]
        spec = tc["spec"]
        dc_map = spec.chinchilla_data_configs or {}
        d1_name = tc["d_names"][0]
        d1_steps = tc["d_steps"][0]
        d1_target = tc["d_targets"][0]
        eval_every = min(tc["base_eval_every"], d1_steps)
        data_cfg = dc_map.get(d1_name, spec.data_config)

        for arch in archs:
            for size in sizes:
                if f"{arch}/{size}" not in tc["grid_meta"]:
                    continue
                ga = _get_grad_accum(arch, size, spec.n_atoms)
                for lr in lrs:
                    traj = _traj_path(chinchilla_dir, task_id, arch, size, lr, d1_name)
                    ckpt = _ckpt_dir(chinchilla_dir, task_id, arch, size, lr, d1_name)
                    if _is_complete(traj, d1_steps, eval_every=eval_every):
                        print(f"[SKIP] {task_id}/{arch}/{size}/{_lr_name(lr)}/{d1_name}")
                        continue
                    cmd = _build_cmd(data_cfg, arch, size, d1_steps, d1_target, lr, ga,
                                     eval_every, eval_n_samples, ckpt, task_id, traj, args.wandb)
                    phase1.append((cmd, f"{task_id}/{arch}/{size}/{_lr_name(lr)}/{d1_name}"))

    if phase1:
        print(f"\n=== Phase 1: D1 LR search — {len(phase1)} runs ===", file=sys.stderr)
        _run_commands(phase1, n_gpus, args.wandb)

    # ── Phase 2: D2+ with best LR only ───────────────────────────────────
    if len(lrs) <= 1 or not lr_warmup:
        # No LR search: run all D budgets with all LRs directly
        phase2: list[tuple[str, str]] = []
        for tc in task_configs:
            task_id = tc["task_id"]
            spec = tc["spec"]
            dc_map = spec.chinchilla_data_configs or {}
            for arch in archs:
                for size in sizes:
                    if f"{arch}/{size}" not in tc["grid_meta"]:
                        continue
                    ga = _get_grad_accum(arch, size, spec.n_atoms)
                    for lr in lrs:
                        for d_name, total_steps, d_target in zip(
                            tc["d_names"][1:], tc["d_steps"][1:], tc["d_targets"][1:]
                        ):
                            eval_every = min(tc["base_eval_every"], total_steps)
                            data_cfg = dc_map.get(d_name, spec.data_config)
                            traj = _traj_path(chinchilla_dir, task_id, arch, size, lr, d_name)
                            ckpt = _ckpt_dir(chinchilla_dir, task_id, arch, size, lr, d_name)
                            if _is_complete(traj, total_steps, eval_every=eval_every):
                                print(f"[SKIP] {task_id}/{arch}/{size}/{_lr_name(lr)}/{d_name}")
                                continue
                            cmd = _build_cmd(data_cfg, arch, size, total_steps, d_target, lr, ga,
                                             eval_every, eval_n_samples, ckpt, task_id, traj, args.wandb)
                            phase2.append((cmd, f"{task_id}/{arch}/{size}/{_lr_name(lr)}/{d_name}"))
        if phase2:
            print(f"\n=== Phase 2: D2+ — {len(phase2)} runs ===", file=sys.stderr)
            _run_commands(phase2, n_gpus, args.wandb)
        return

    # LR warmup: pick best LR from D1 results for each (arch, size) and run D2+ with it
    phase2_warmup: list[tuple[str, str]] = []
    for tc in task_configs:
        task_id = tc["task_id"]
        spec = tc["spec"]
        dc_map = spec.chinchilla_data_configs or {}
        d1_name = tc["d_names"][0]

        for arch in archs:
            for size in sizes:
                if f"{arch}/{size}" not in tc["grid_meta"]:
                    continue
                ga = _get_grad_accum(arch, size, spec.n_atoms)
                best_lr = _best_lr_from_d1(chinchilla_dir, task_id, arch, size, lrs, d1_name)
                if best_lr is None:
                    print(f"[WARN] Could not select LR for {task_id}/{arch}/{size} — "
                          f"D1 incomplete? Running all LRs.", file=sys.stderr)
                    selected_lrs = lrs
                else:
                    print(f"[LR] {task_id}/{arch}/{size}: best LR={best_lr:.0e}", file=sys.stderr)
                    selected_lrs = [best_lr]

                for lr in selected_lrs:
                    for d_name, total_steps, d_target in zip(
                        tc["d_names"][1:], tc["d_steps"][1:], tc["d_targets"][1:]
                    ):
                        eval_every = min(tc["base_eval_every"], total_steps)
                        data_cfg = dc_map.get(d_name, spec.data_config)
                        traj = _traj_path(chinchilla_dir, task_id, arch, size, lr, d_name)
                        ckpt = _ckpt_dir(chinchilla_dir, task_id, arch, size, lr, d_name)
                        if _is_complete(traj, total_steps, eval_every=eval_every):
                            print(f"[SKIP] {task_id}/{arch}/{size}/{_lr_name(lr)}/{d_name}")
                            continue
                        cmd = _build_cmd(data_cfg, arch, size, total_steps, d_target, lr, ga,
                                         eval_every, eval_n_samples, ckpt, task_id, traj, args.wandb)
                        phase2_warmup.append((cmd, f"{task_id}/{arch}/{size}/{_lr_name(lr)}/{d_name}"))

    if phase2_warmup:
        print(f"\n=== Phase 2: D2+ with best LR — {len(phase2_warmup)} runs ===", file=sys.stderr)
        _run_commands(phase2_warmup, n_gpus, args.wandb)
