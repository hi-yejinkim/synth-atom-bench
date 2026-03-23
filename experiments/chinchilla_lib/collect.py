"""Collect subcommand: aggregate trajectory.jsonl files into results.json + CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from experiments.chinchilla_lib.config import ALL_ARCHS, D_NAMES, D_STEPS
from experiments.chinchilla_lib.helpers import _grid_meta_path, _results_path


def collect(args: argparse.Namespace) -> None:
    """Walk trajectory.jsonl files and aggregate into results.json per task."""
    chinchilla_dir = args.chinchilla_dir
    tasks = ([t.strip() for t in args.tasks.split(",")] if args.tasks else
             [d for d in os.listdir(chinchilla_dir)
              if os.path.isdir(os.path.join(chinchilla_dir, d))])

    for task_id in tasks:
        task_dir = os.path.join(chinchilla_dir, task_id)
        if not os.path.isdir(task_dir):
            print(f"[WARN] No directory for task '{task_id}'", file=sys.stderr)
            continue

        # Load grid_meta once per task for flops lookup
        _gm: dict = {}
        grid_meta_path = _grid_meta_path(chinchilla_dir, task_id)
        if os.path.exists(grid_meta_path):
            try:
                with open(grid_meta_path) as _gf:
                    _gm = json.load(_gf)
            except Exception:
                pass

        trajectories: list[dict] = []
        # Walk: task_dir/{arch}/{size}/{lr_name}/{d_name}/trajectory.jsonl
        # Each (arch, size, lr, d_name) is one completed D-budget run.
        # The terminal point of each trajectory = the measurement for (N, D, L).
        for arch in sorted(os.listdir(task_dir)):
            arch_dir = os.path.join(task_dir, arch)
            if not os.path.isdir(arch_dir) or arch not in ALL_ARCHS:
                continue
            for size in sorted(os.listdir(arch_dir)):
                size_dir = os.path.join(arch_dir, size)
                if not os.path.isdir(size_dir):
                    continue
                for lr_name in sorted(os.listdir(size_dir)):
                    lr_dir = os.path.join(size_dir, lr_name)
                    if not os.path.isdir(lr_dir):
                        continue
                    # Parse LR from lr_name
                    try:
                        lr_val = float(lr_name.replace("lr", ""))
                    except ValueError:
                        lr_val = float("nan")

                    # Each d_name subdir is one D-budget run
                    for d_name in sorted(os.listdir(lr_dir)):
                        run_dir = os.path.join(lr_dir, d_name)
                        traj_file = os.path.join(run_dir, "trajectory.jsonl")
                        if not os.path.exists(traj_file):
                            continue
                        points: list[dict] = []
                        try:
                            with open(traj_file) as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        points.append(json.loads(line))
                        except Exception as e:
                            print(f"[WARN] {traj_file}: {e}", file=sys.stderr)
                            continue
                        if not points:
                            continue

                        n_params = points[0].get("n_params", 0)
                        meta_key = f"{arch}/{size}"
                        if meta_key in _gm and _gm[meta_key].get("flops_per_step", 0) > 0:
                            flops_per_step = int(_gm[meta_key]["flops_per_step"])
                        elif points[0].get("step", 0) > 0 and points[0].get("total_flops", 0) > 0:
                            flops_per_step = int(points[0]["total_flops"] / points[0]["step"])
                        else:
                            flops_per_step = 0

                        # Terminal point = final eval of this D-budget run
                        # Skip incomplete runs: verify terminal step ≥ 95% of expected
                        terminal = points[-1]
                        d_idx = D_NAMES.index(d_name) if d_name in D_NAMES else -1
                        if d_idx >= 0:
                            exp_steps = D_STEPS[d_idx]
                            term_step = terminal.get("step", 0)
                            if term_step < exp_steps * 0.95:
                                print(f"[WARN] {traj_file}: terminal step {term_step} "
                                      f"< 95% of expected {exp_steps}, skipping",
                                      file=sys.stderr)
                                continue

                        trajectories.append({
                            "task": task_id,
                            "arch": arch,
                            "size": size,
                            "lr_name": lr_name,
                            "lr": lr_val,
                            "d_name": d_name,
                            "n_params": n_params,
                            "flops_per_step": flops_per_step,
                            # points = all evals in this run (for monitoring)
                            # terminal = the (N, D, L) data point for fitting
                            "points": points,
                            "terminal": terminal,
                        })

        if not trajectories:
            print(f"[WARN] No trajectory data found for task '{task_id}'", file=sys.stderr)
            continue

        # For each (arch, size, d_name): select best LR by terminal violation_rate.
        # Key = (arch, size, d_name) — one winner per D budget per model.
        best: dict[tuple, dict] = {}
        for traj in trajectories:
            key = (traj["arch"], traj["size"], traj["d_name"])
            vr = traj["terminal"].get("violation_rate", float("inf"))
            if key not in best or vr < best[key]["terminal_vr"]:
                best[key] = {**traj, "terminal_vr": vr}

        results = {
            "task": task_id,
            "all_trajectories": trajectories,
            # best_by_size_d: keyed by "arch/size/d_name", one (N,D,L) point per cell
            "best_by_size_d": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in best.items()},
        }
        out_path = _results_path(chinchilla_dir, task_id)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{task_id}] Collected {len(trajectories)} runs "
              f"({len(best)} unique (arch,size,D) cells) → {out_path}")

        # ── CSV export ────────────────────────────────────────────────────
        # One row per (arch, size, D_budget) with best LR.
        # Each row is a clean (N, D, L) data point for Approach 1 + 3.
        csv_path = os.path.join(os.path.dirname(out_path), "scaling_results.csv")
        csv_rows = []
        for (arch, size, d_name), traj in best.items():
            pt = traj["terminal"]
            fps = traj["flops_per_step"]
            step = pt.get("step", 0)
            total_flops = fps * step if fps and step else pt.get("total_flops", 0)
            csv_rows.append({
                "Architecture": arch,
                "Parameters_N": traj["n_params"],
                "Data_Tokens_D": pt.get("D_seen") or (
                    print(f"[WARN] D_seen missing for {arch}/{size}/{d_name}, CSV row may be wrong", file=sys.stderr)
                    or 0
                ),
                "Total_FLOPs_C": int(total_flops),
                "Final_Violation_Rate": pt.get("violation_rate", float("nan")),
                "task": task_id,
                "size": size,
                "d_name": d_name,
                "lr": traj["lr"],
            })
        if csv_rows:
            fieldnames = ["Architecture", "Parameters_N", "Data_Tokens_D",
                          "Total_FLOPs_C", "Final_Violation_Rate",
                          "task", "size", "d_name", "lr"]
            with open(csv_path, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[{task_id}] CSV exported ({len(csv_rows)} rows) → {csv_path}")
