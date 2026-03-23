"""Scaling experiment: generate grid, run, collect results, fit scaling laws."""

import argparse
import json
import math
import os
import subprocess
import sys
from itertools import product

import numpy as np
import torch

from experiments.model_registry import MODEL_DEFAULTS, MODEL_REGISTRY, SIZE_PRESETS

BUDGETS = [1e15, 4e15, 1.6e16, 6.4e16, 2.56e17]
LEARNING_RATES = [1e-4, 1e-3]
ALL_SIZES = ["xs", "small", "medium", "large", "xl"]

MIN_STEPS = 2000
MAX_STEPS = 1_000_000

# Default data config per task (used when --data is not specified)
TASK_DEFAULT_DATA = {
    "hard_sphere": "easy_small",
    "chain": "chain_N10",
    "vsepr": "vsepr_sp3",
    "sequence": "sequence_linear",
}


def _read_n_atoms_from_data_config(data_name: str) -> int:
    """Read n_atoms from configs/data/{data_name}.yaml."""
    import yaml

    config_path = os.path.join("configs", "data", f"{data_name}.yaml")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get("n_atoms", 10))
    return 10


def measure_flops(arch: str, size: str, batch_size: int = 256, n_atoms: int = 10) -> tuple[int, int]:
    """Instantiate model, measure FLOPs per step, return (flops_per_step, n_params)."""
    from experiments.train import count_flops

    kwargs = {**MODEL_DEFAULTS[arch], **SIZE_PRESETS[arch][size]}
    model = MODEL_REGISTRY[arch](**kwargs)
    n_params = sum(p.numel() for p in model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    flops = count_flops(model, n_atoms, batch_size, device)

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return flops, n_params


def generate_grid(args):
    """Measure FLOPs per config and print viable training commands."""
    archs = args.archs.split(",") if args.archs else list(SIZE_PRESETS.keys())
    sizes = args.sizes.split(",") if args.sizes else ALL_SIZES
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LEARNING_RATES
    budgets = [float(x) for x in args.budgets.split(",")] if args.budgets else BUDGETS
    batch_size = args.batch_size

    # Resolve data config: explicit --data > task default > None
    task = getattr(args, "task", None)
    data_config = getattr(args, "data", None)
    if not data_config and task and task in TASK_DEFAULT_DATA:
        data_config = TASK_DEFAULT_DATA[task]
        print(f"Task '{task}' → using default data config: {data_config}", file=sys.stderr)

    if data_config:
        n_atoms = _read_n_atoms_from_data_config(data_config)
        print(f"Data config: {data_config} (n_atoms={n_atoms})", file=sys.stderr)
    else:
        n_atoms = args.n_atoms

    # Measure FLOPs for all (arch, size) combos
    print("Measuring FLOPs per configuration...", file=sys.stderr)
    flops_table = {}
    params_table = {}
    for arch in archs:
        for size in sizes:
            flops, n_params = measure_flops(arch, size, batch_size, n_atoms)
            flops_table[(arch, size)] = flops
            params_table[(arch, size)] = n_params
            print(f"  {arch:>12} {size:>6}: {n_params:>10,} params, {flops:.2e} FLOPs/step", file=sys.stderr)

    # Generate commands
    commands = []
    skipped = 0
    for budget, arch, size, lr in product(budgets, archs, sizes, lrs):
        flops_per_step = flops_table[(arch, size)]
        max_steps = int(budget / flops_per_step)

        if max_steps < MIN_STEPS:
            skipped += 1
            continue
        if max_steps > MAX_STEPS:
            skipped += 1
            continue

        run_name = f"{arch}_{size}_lr{lr:.0e}_budget{budget:.2e}"
        ckpt_dir = os.path.join(args.scaling_dir, run_name)

        # Build Hydra overrides
        preset = SIZE_PRESETS[arch][size]
        model_overrides = " ".join(f"model.model_kwargs.{k}={v}" for k, v in preset.items())

        data_override = f"data={data_config} " if data_config else ""
        cmd = (
            f"uv run python experiments/train.py "
            f"{data_override}"
            f"model={arch} "
            f"model.size={size} "
            f"train.lr={lr} "
            f"train.max_steps={max_steps} "
            f"{model_overrides} "
            f"checkpoint.dir={ckpt_dir} "
            f"logging.enabled={str(args.wandb).lower()} "
            f"hydra.run.dir={ckpt_dir}"
        )
        commands.append((run_name, cmd, arch, size, lr, budget, max_steps, flops_per_step))

    # Print commands
    for run_name, cmd, arch, size, lr, budget, max_steps, flops_per_step in commands:
        print(cmd)

    # Summary
    print(f"\n# Total: {len(commands)} runs ({skipped} skipped)", file=sys.stderr)
    print(f"# Budgets: {[f'{b:.0e}' for b in budgets]}", file=sys.stderr)
    print(f"# Sizes: {sizes}", file=sys.stderr)
    print(f"# LRs: {lrs}", file=sys.stderr)

    # Print grid overview table
    print(f"\n# Grid overview (max_steps per budget):", file=sys.stderr)
    header = f"# {'arch':>12} {'size':>6} |" + "".join(f" {b:>10.0e}" for b in budgets)
    print(header, file=sys.stderr)
    print(f"# {'-'*len(header)}", file=sys.stderr)
    for arch in archs:
        for size in sizes:
            flops = flops_table[(arch, size)]
            cells = []
            for b in budgets:
                steps = int(b / flops)
                if steps < MIN_STEPS:
                    cells.append(f" {'skip':>10}")
                elif steps > MAX_STEPS:
                    cells.append(f" {'skip':>10}")
                else:
                    cells.append(f" {steps:>10,}")
            print(f"# {arch:>12} {size:>6} |" + "".join(cells), file=sys.stderr)

    # Save grid metadata for collect
    meta_path = os.path.join(args.scaling_dir, "grid_meta.json")
    os.makedirs(args.scaling_dir, exist_ok=True)
    meta = {
        "flops_table": {f"{a}_{s}": v for (a, s), v in flops_table.items()},
        "params_table": {f"{a}_{s}": v for (a, s), v in params_table.items()},
        "runs": [
            {
                "name": name,
                "arch": arch,
                "size": size,
                "lr": lr,
                "budget": budget,
                "max_steps": max_steps,
                "flops_per_step": fps,
            }
            for name, _, arch, size, lr, budget, max_steps, fps in commands
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n# Grid metadata saved to {meta_path}", file=sys.stderr)


def _extract_run_name_from_cmd(cmd: str) -> str:
    """Extract run name from checkpoint.dir in a command string."""
    for part in cmd.split():
        if part.startswith("checkpoint.dir="):
            return os.path.basename(part.split("=", 1)[1])
    return ""


def _execute_commands(remaining: list[str], scaling_dir: str, n_gpus: int) -> None:
    """Execute a list of training commands sequentially or in parallel across GPUs."""
    total = len(remaining)

    if n_gpus <= 1:
        for i, cmd in enumerate(remaining):
            print(f"\n{'='*60}")
            print(f"Job {i+1}/{total}: {cmd}")
            print(f"{'='*60}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"Warning: job {i+1} exited with code {result.returncode}", file=sys.stderr)
        return

    # Parallel: n_gpus jobs at a time, each pinned to a GPU
    import time
    active: dict[int, tuple[subprocess.Popen, str, int]] = {}
    job_queue = list(enumerate(remaining))
    completed = 0
    failed = 0

    def _launch(gpu_id: int, job_idx: int, cmd: str) -> None:
        run_name = _extract_run_name_from_cmd(cmd)
        log_path = os.path.join(scaling_dir, run_name, "train.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_f = open(log_path, "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        proc = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        proc._log_file = log_f
        active[gpu_id] = (proc, run_name, job_idx)
        print(f"  [GPU {gpu_id}] Started job {job_idx+1}/{total}: {run_name}")

    for gpu_id in range(min(n_gpus, len(job_queue))):
        job_idx, cmd = job_queue.pop(0)
        _launch(gpu_id, job_idx, cmd)

    while active:
        time.sleep(5)
        for gpu_id in list(active.keys()):
            proc, run_name, job_idx = active[gpu_id]
            ret = proc.poll()
            if ret is not None:
                proc._log_file.close()
                del active[gpu_id]
                completed += 1
                if ret != 0:
                    failed += 1
                status = "Completed" if ret == 0 else f"Failed (exit {ret})"
                print(f"  [GPU {gpu_id}] {status} ({completed}/{total}): {run_name}")
                if job_queue:
                    next_idx, next_cmd = job_queue.pop(0)
                    _launch(gpu_id, next_idx, next_cmd)

    print(f"\nAll done: {completed - failed} succeeded, {failed} failed out of {total}")


def run_grid(args):
    """Execute scaling grid commands, optionally in parallel across GPUs."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        generate_grid(args)

    commands = [
        line.strip()
        for line in buf.getvalue().strip().split("\n")
        if line.strip() and not line.startswith("#")
    ]

    # Skip already-completed runs
    remaining = []
    for cmd in commands:
        run_name = _extract_run_name_from_cmd(cmd)
        run_dir = os.path.join(args.scaling_dir, run_name)
        latest_pt = os.path.join(run_dir, "latest.pt")
        best_pt = os.path.join(run_dir, "best.pt")

        if os.path.isfile(latest_pt):
            try:
                data = torch.load(latest_pt, map_location="cpu", weights_only=False)
                saved_step = data.get("step", 0)
                config = data.get("config", {})
                max_steps = config.get("train", {}).get("max_steps", 0)
                if max_steps > 0 and saved_step >= max_steps:
                    print(f"Skipping (completed {saved_step}/{max_steps}): {run_name}")
                    continue
                else:
                    print(f"Resuming (incomplete {saved_step}/{max_steps}): {run_name}")
            except Exception as e:
                print(f"Warning: could not read {latest_pt}: {e}, will re-run")
        elif os.path.isfile(best_pt):
            print(f"Resuming (no latest.pt): {run_name}")

        remaining.append(cmd)

    print(f"\n{len(commands)} total, {len(commands) - len(remaining)} done, {len(remaining)} remaining")
    _execute_commands(remaining, args.scaling_dir, args.n_gpus)


def _count_params(arch: str, model_kwargs: dict) -> int:
    """Instantiate model to count parameters (no GPU needed)."""
    try:
        kwargs = {**MODEL_DEFAULTS.get(arch, {}), **model_kwargs}
        model = MODEL_REGISTRY[arch](**kwargs)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def collect_results(args):
    """Walk scaling directory, load latest.pt from each run, save results.json.

    Uses latest.pt because it tracks the running-best metrics across all
    evaluation steps.  best.pt only saves weights when g(r) distance improves,
    so its best_clash_rate may miss better values achieved at other steps.
    """
    scaling_dir = args.scaling_dir
    if not os.path.isdir(scaling_dir):
        print(f"Scaling directory not found: {scaling_dir}", file=sys.stderr)
        sys.exit(1)

    # Load grid metadata for budget/flops info (not stored in checkpoints)
    meta_path = os.path.join(scaling_dir, "grid_meta.json")
    grid_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for run in meta.get("runs", []):
            grid_meta[run["name"]] = run

    # Cache param counts to avoid repeated instantiation
    param_cache = {}

    results = []
    for run_name in sorted(os.listdir(scaling_dir)):
        run_dir = os.path.join(scaling_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        # Prefer latest.pt (tracks running-best metrics across all evals)
        # Fall back to best.pt if latest.pt is missing
        ckpt_path = os.path.join(run_dir, "latest.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(run_dir, "best.pt")
        if not os.path.isfile(ckpt_path):
            continue

        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = data.get("config", {})
            arch = config.get("model", {}).get("arch", "unknown")
            model_kwargs = config.get("model", {}).get("model_kwargs", {})
            size = config.get("model", {}).get("size", "unknown")
            lr = config.get("train", {}).get("lr", 0)
            # Prefer grid metadata for budget/flops (not stored in checkpoints)
            gm = grid_meta.get(run_name, {})
            budget = gm.get("budget", 0) or config.get("train", {}).get("budget", 0) or 0
            max_steps = config.get("train", {}).get("max_steps", 0)
            cr = data.get("best_clash_rate", float("inf"))
            grd = data.get("best_gr_distance", float("inf"))
            bvr = data.get("best_bond_violation_rate", float("inf"))
            ncr = data.get("best_nonbonded_clash_rate", float("inf"))
            # VSEPR metrics
            angle_jsd = data.get("best_angle_jsd", float("inf"))
            bl_in_peak = data.get("best_bond_length_in_peak_ratio", float("inf"))
            torsion_obr = data.get("best_torsional_out_of_bin_rate", float("inf"))
            valence_ocr = data.get("best_valence_overcoord_rate", float("inf"))
            # Sequence metrics
            contact_recall = data.get("best_contact_recall", float("inf"))
            rg_error = data.get("best_rg_error", float("inf"))
            seq_bvr = data.get("best_seq_bond_violation_rate", float("inf"))
            step = data.get("step", 0)

            # Count params (cached)
            cache_key = f"{arch}_{size}"
            if cache_key not in param_cache:
                param_cache[cache_key] = _count_params(arch, model_kwargs)
            n_params = param_cache[cache_key]

            # Use measured flops_per_step from grid metadata if available
            flops_per_step = gm.get("flops_per_step", 0) or (budget / max_steps if (budget and max_steps) else 0)
            total_flops = flops_per_step * step if flops_per_step else 0

            result_entry = {
                "run": run_name,
                "arch": arch,
                "size": size,
                "lr": lr,
                "budget": float(budget),
                "best_clash_rate": cr,
                "best_gr_distance": grd,
                "step": step,
                "n_params": n_params,
                "flops_per_step": flops_per_step,
                "total_flops": total_flops,
                "model_kwargs": model_kwargs,
            }
            if bvr < float("inf"):
                result_entry["best_bond_violation_rate"] = bvr
            if ncr < float("inf"):
                result_entry["best_nonbonded_clash_rate"] = ncr
            # VSEPR metrics
            if angle_jsd < float("inf"):
                result_entry["best_angle_jsd"] = angle_jsd
            if bl_in_peak < float("inf"):
                result_entry["best_bond_length_in_peak_ratio"] = bl_in_peak
            if torsion_obr < float("inf"):
                result_entry["best_torsional_out_of_bin_rate"] = torsion_obr
            if valence_ocr < float("inf"):
                result_entry["best_valence_overcoord_rate"] = valence_ocr
            # Sequence metrics
            if contact_recall < float("inf"):
                result_entry["best_contact_recall"] = contact_recall
            if rg_error < float("inf"):
                result_entry["best_rg_error"] = rg_error
            if seq_bvr < float("inf"):
                result_entry["best_seq_bond_violation_rate"] = seq_bvr
            results.append(result_entry)
        except Exception as e:
            print(f"Warning: failed to load {best_pt}: {e}", file=sys.stderr)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Print all results
    results.sort(key=lambda r: (r["arch"], r["budget"], r["best_gr_distance"]))
    print(f"\n{'Run':<45} {'Arch':<12} {'Size':<6} {'LR':<8} {'Budget':>10} {'CR':>8} {'g(r)':>8} {'Step':>8}")
    print("-" * 110)
    for r in results:
        grd_str = f"{r['best_gr_distance']:>8.4f}" if r["best_gr_distance"] < float("inf") else "     n/a"
        print(
            f"{r['run']:<45} {r['arch']:<12} {r['size']:<6} {r['lr']:<8.0e} "
            f"{r['budget']:>10.0e} {r['best_clash_rate']:>8.4f} {grd_str} {r['step']:>8}"
        )

    # Best per (arch, budget): select by lowest clash rate (primary), fall back to g(r) distance
    best_per_budget = {}
    for r in results:
        key = (r["arch"], r["budget"])
        if key not in best_per_budget:
            best_per_budget[key] = r
        else:
            prev = best_per_budget[key]
            if r["best_clash_rate"] < prev["best_clash_rate"]:
                best_per_budget[key] = r
            elif r["best_clash_rate"] == prev["best_clash_rate"] and r["best_gr_distance"] < prev["best_gr_distance"]:
                best_per_budget[key] = r

    print(f"\nBest per (architecture, budget):")
    print("-" * 95)
    print(f"{'Arch':<12} {'Budget':>10} {'Best CR':>10} {'Best g(r)':>10} {'Size':<6} {'LR':<8} {'Params':>10}")
    print("-" * 95)
    for (arch, budget), r in sorted(best_per_budget.items()):
        grd_str = f"{r['best_gr_distance']:>10.4f}" if r["best_gr_distance"] < float("inf") else "       n/a"
        print(
            f"{arch:<12} {budget:>10.0e} {r['best_clash_rate']:>10.4f} "
            f"{grd_str} {r['size']:<6} {r['lr']:<8.0e} {r['n_params']:>10,}"
        )

    # Save results
    out = {
        "all_results": results,
        "best_per_budget": {
            f"{arch}_{budget:.0e}": r
            for (arch, budget), r in best_per_budget.items()
        },
    }
    results_path = os.path.join(scaling_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def fit_scaling(args):
    """Fit scaling laws and generate plots."""
    from scipy.optimize import curve_fit

    from viz import save_figure, synthbench_style
    from viz.scaling import fit_scaling_law, plot_scaling_curves
    from viz.style import ARCH_COLORS, ARCH_MARKERS, DOUBLE_COL

    import matplotlib.pyplot as plt

    scaling_dir = args.scaling_dir
    results_path = os.path.join(scaling_dir, "results.json")
    if not os.path.isfile(results_path):
        print(f"Results file not found: {results_path}", file=sys.stderr)
        print("Run 'collect' first.", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    best_per_budget = data["best_per_budget"]

    # Organize by architecture (skip budget=0 entries from orphaned runs)
    arch_data = {}
    for key, r in best_per_budget.items():
        if r["budget"] <= 0:
            continue
        arch = r["arch"]
        if arch not in arch_data:
            arch_data[arch] = {
                "flops": [], "clash_rate": [], "gr_distance": [],
                "bond_violation_rate": [], "nonbonded_clash_rate": [],
                # VSEPR
                "angle_jsd": [], "bond_length_in_peak_ratio": [],
                "torsional_out_of_bin_rate": [], "valence_overcoord_rate": [],
                # Sequence
                "contact_recall": [], "rg_error": [], "seq_bond_violation_rate": [],
            }
        arch_data[arch]["flops"].append(r["budget"])
        arch_data[arch]["clash_rate"].append(r["best_clash_rate"])
        arch_data[arch]["gr_distance"].append(r.get("best_gr_distance", float("inf")))
        arch_data[arch]["bond_violation_rate"].append(r.get("best_bond_violation_rate", float("inf")))
        arch_data[arch]["nonbonded_clash_rate"].append(r.get("best_nonbonded_clash_rate", float("inf")))
        arch_data[arch]["angle_jsd"].append(r.get("best_angle_jsd", float("inf")))
        arch_data[arch]["bond_length_in_peak_ratio"].append(r.get("best_bond_length_in_peak_ratio", float("inf")))
        arch_data[arch]["torsional_out_of_bin_rate"].append(r.get("best_torsional_out_of_bin_rate", float("inf")))
        arch_data[arch]["valence_overcoord_rate"].append(r.get("best_valence_overcoord_rate", float("inf")))
        arch_data[arch]["contact_recall"].append(r.get("best_contact_recall", float("inf")))
        arch_data[arch]["rg_error"].append(r.get("best_rg_error", float("inf")))
        arch_data[arch]["seq_bond_violation_rate"].append(r.get("best_seq_bond_violation_rate", float("inf")))

    # Sort by budget within each arch; clip clash_rate floor at 1/n_eval_samples
    eval_floor = 1.0 / 1000  # n_eval_samples from configs/train.yaml
    _scalar_keys = [
        "gr_distance", "bond_violation_rate", "nonbonded_clash_rate",
        "angle_jsd", "bond_length_in_peak_ratio", "torsional_out_of_bin_rate",
        "valence_overcoord_rate", "contact_recall", "rg_error", "seq_bond_violation_rate",
    ]
    for arch in arch_data:
        order = np.argsort(arch_data[arch]["flops"])
        arch_data[arch]["flops"] = np.array(arch_data[arch]["flops"])[order]
        arch_data[arch]["clash_rate"] = np.clip(
            np.array(arch_data[arch]["clash_rate"])[order], eval_floor, None
        )
        for key in _scalar_keys:
            arch_data[arch][key] = np.array(arch_data[arch][key])[order]

    # Output directories (defined here so all plotting blocks below can use them)
    plots_dir = "outputs/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Capitalize arch names for plotting (matches ARCH_COLORS keys)
    arch_name_map = {"painn": "PaiNN", "transformer": "Transformer", "pairformer": "Pairformer"}
    plot_data = {}
    for arch, d in arch_data.items():
        display_name = arch_name_map.get(arch, arch)
        plot_data[display_name] = d

    # Fit and report
    print("\nScaling Law Fits:")
    print("=" * 60)
    print(f"{'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
    print("-" * 60)
    fits = {}
    for arch, d in arch_data.items():
        flops = np.array(d["flops"], dtype=float)
        cr = np.array(d["clash_rate"], dtype=float)
        if len(flops) < 3:
            print(f"{arch:<15} insufficient data ({len(flops)} points)", file=sys.stderr)
            continue
        try:
            a, alpha, floor = fit_scaling_law(flops, cr)
            fits[arch] = {"a": a, "alpha": alpha, "floor": floor}
            print(f"{arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
        except RuntimeError as e:
            print(f"{arch:<15} fit failed: {e}", file=sys.stderr)

    # Fit g(r) distance scaling laws
    has_gr = any(
        np.isfinite(d["gr_distance"]).any() for d in arch_data.values()
    )
    gr_fits = {}
    if has_gr:
        print("\nScaling Law Fits (g(r) distance):")
        print("=" * 60)
        print(f"{'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
        print("-" * 60)
        for arch, d in arch_data.items():
            flops = np.array(d["flops"], dtype=float)
            grd = np.array(d["gr_distance"], dtype=float)
            valid = np.isfinite(grd)
            if valid.sum() < 3:
                print(f"{arch:<15} insufficient data ({valid.sum()} finite points)", file=sys.stderr)
                continue
            try:
                a, alpha, floor = fit_scaling_law(flops[valid], grd[valid])
                gr_fits[arch] = {"a": a, "alpha": alpha, "floor": floor}
                print(f"{arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
            except RuntimeError as e:
                print(f"{arch:<15} fit failed: {e}", file=sys.stderr)
        fits["gr_distance"] = gr_fits

    # Fit chain-specific scaling laws (bond violation rate)
    has_bvr = any(
        np.isfinite(d["bond_violation_rate"]).any() for d in arch_data.values()
    )
    bvr_fits = {}
    if has_bvr:
        print("\nScaling Law Fits (bond violation rate):")
        print("=" * 60)
        print(f"{'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
        print("-" * 60)
        for arch, d in arch_data.items():
            flops = np.array(d["flops"], dtype=float)
            bvr = np.array(d["bond_violation_rate"], dtype=float)
            valid = np.isfinite(bvr)
            if valid.sum() < 3:
                print(f"{arch:<15} insufficient data ({valid.sum()} finite points)", file=sys.stderr)
                continue
            try:
                a, alpha, floor = fit_scaling_law(flops[valid], bvr[valid])
                bvr_fits[arch] = {"a": a, "alpha": alpha, "floor": floor}
                print(f"{arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
            except RuntimeError as e:
                print(f"{arch:<15} fit failed: {e}", file=sys.stderr)
        fits["bond_violation_rate"] = bvr_fits

    # Fit chain-specific scaling laws (nonbonded clash rate)
    has_ncr = any(
        np.isfinite(d["nonbonded_clash_rate"]).any() for d in arch_data.values()
    )
    ncr_fits = {}
    if has_ncr:
        print("\nScaling Law Fits (nonbonded clash rate):")
        print("=" * 60)
        print(f"{'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
        print("-" * 60)
        for arch, d in arch_data.items():
            flops = np.array(d["flops"], dtype=float)
            ncr = np.array(d["nonbonded_clash_rate"], dtype=float)
            valid = np.isfinite(ncr)
            if valid.sum() < 3:
                print(f"{arch:<15} insufficient data ({valid.sum()} finite points)", file=sys.stderr)
                continue
            try:
                a, alpha, floor = fit_scaling_law(flops[valid], ncr[valid])
                ncr_fits[arch] = {"a": a, "alpha": alpha, "floor": floor}
                print(f"{arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
            except RuntimeError as e:
                print(f"{arch:<15} fit failed: {e}", file=sys.stderr)
        fits["nonbonded_clash_rate"] = ncr_fits

    # ---- VSEPR metrics -------------------------------------------------------
    _vsepr_metrics = [
        ("angle_jsd", "Angle distribution JSD"),
        ("torsional_out_of_bin_rate", "Torsional out-of-bin rate"),
        ("valence_overcoord_rate", "Valence overcoordination rate"),
    ]
    for metric_key, metric_label in _vsepr_metrics:
        has_metric = any(
            np.isfinite(d[metric_key]).any() for d in arch_data.values()
        )
        if not has_metric:
            continue
        metric_fits = {}
        print(f"\nScaling Law Fits ({metric_label}):")
        print("=" * 60)
        print(f"{'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
        print("-" * 60)
        for arch, d in arch_data.items():
            flops = np.array(d["flops"], dtype=float)
            vals = np.array(d[metric_key], dtype=float)
            valid = np.isfinite(vals)
            if valid.sum() < 3:
                continue
            try:
                a, alpha, floor = fit_scaling_law(flops[valid], vals[valid])
                metric_fits[arch] = {"a": a, "alpha": alpha, "floor": floor}
                print(f"{arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
            except RuntimeError as e:
                print(f"{arch:<15} fit failed: {e}", file=sys.stderr)
        fits[metric_key] = metric_fits

    # bond_length_in_peak_ratio: higher = better, invert for scaling law
    has_blip = any(
        np.isfinite(d["bond_length_in_peak_ratio"]).any() for d in arch_data.values()
    )
    if has_blip:
        blip_plot_data = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d["bond_length_in_peak_ratio"])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                # Invert so that "error" decreases with compute (1 - ratio)
                blip_plot_data[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": 1.0 - d["bond_length_in_peak_ratio"][valid],
                }
        if blip_plot_data:
            with synthbench_style():
                fig = plot_scaling_curves(blip_plot_data, fit_curves=True,
                                          ylabel="Bond length out-of-peak rate (1 - in-peak)")
                save_figure(fig, os.path.join(plots_dir, "scaling_curves_bl_in_peak"))
                print(f"Saved {plots_dir}/scaling_curves_bl_in_peak.png")

    # ---- Sequence metrics ----------------------------------------------------
    _seq_metrics = [
        ("rg_error", "Radius of gyration error"),
        ("seq_bond_violation_rate", "Bond violation rate (sequence)"),
    ]
    for metric_key, metric_label in _seq_metrics:
        has_metric = any(
            np.isfinite(d[metric_key]).any() for d in arch_data.values()
        )
        if not has_metric:
            continue
        metric_fits = {}
        print(f"\nScaling Law Fits ({metric_label}):")
        print("=" * 60)
        print(f"{'Architecture':<15} {'alpha':>8} {'prefactor':>12} {'floor':>10}")
        print("-" * 60)
        for arch, d in arch_data.items():
            flops = np.array(d["flops"], dtype=float)
            vals = np.array(d[metric_key], dtype=float)
            valid = np.isfinite(vals)
            if valid.sum() < 3:
                continue
            try:
                a, alpha, floor = fit_scaling_law(flops[valid], vals[valid])
                metric_fits[arch] = {"a": a, "alpha": alpha, "floor": floor}
                print(f"{arch:<15} {alpha:>8.3f} {a:>12.4f} {floor:>10.5f}")
            except RuntimeError as e:
                print(f"{arch:<15} fit failed: {e}", file=sys.stderr)
        fits[metric_key] = metric_fits

    # contact_recall: higher = better; invert for scaling law
    has_cr_seq = any(
        np.isfinite(d["contact_recall"]).any() for d in arch_data.values()
    )
    if has_cr_seq:
        cr_seq_plot_data = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d["contact_recall"])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                cr_seq_plot_data[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": 1.0 - d["contact_recall"][valid],
                }
        if cr_seq_plot_data:
            with synthbench_style():
                fig = plot_scaling_curves(cr_seq_plot_data, fit_curves=True,
                                          ylabel="Contact miss rate (1 - recall)")
                save_figure(fig, os.path.join(plots_dir, "scaling_curves_contact_recall"))
                print(f"Saved {plots_dir}/scaling_curves_contact_recall.png")

    # Save fits
    fits_path = os.path.join(scaling_dir, "scaling_fits.json")
    with open(fits_path, "w") as f:
        json.dump(fits, f, indent=2)
    print(f"\nFits saved to {fits_path}")

    # Plot 1: Scaling curves (clash rate)
    with synthbench_style():
        fig = plot_scaling_curves(plot_data, fit_curves=True)
        save_figure(fig, os.path.join(plots_dir, "scaling_curves"))
        print(f"Saved {plots_dir}/scaling_curves.png")

    # Plot 1b: Scaling curves (g(r) distance)
    if has_gr:
        gr_plot_data = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d["gr_distance"])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                gr_plot_data[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": d["gr_distance"][valid],  # reuse clash_rate key for plot_scaling_curves
                }
        if gr_plot_data:
            with synthbench_style():
                fig = plot_scaling_curves(gr_plot_data, fit_curves=True, ylabel="g(r) L1 distance")
                save_figure(fig, os.path.join(plots_dir, "scaling_curves_gr_distance"))
                print(f"Saved {plots_dir}/scaling_curves_gr_distance.png")

    # Plot 1c: Scaling curves (bond violation rate)
    if has_bvr:
        bvr_plot_data = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d["bond_violation_rate"])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                bvr_plot_data[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": d["bond_violation_rate"][valid],
                }
        if bvr_plot_data:
            with synthbench_style():
                fig = plot_scaling_curves(bvr_plot_data, fit_curves=True, ylabel="Bond violation rate")
                save_figure(fig, os.path.join(plots_dir, "scaling_curves_bond_violation"))
                print(f"Saved {plots_dir}/scaling_curves_bond_violation.png")

    # Plot 1d: Scaling curves (nonbonded clash rate)
    if has_ncr:
        ncr_plot_data = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d["nonbonded_clash_rate"])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                ncr_plot_data[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": d["nonbonded_clash_rate"][valid],
                }
        if ncr_plot_data:
            with synthbench_style():
                fig = plot_scaling_curves(ncr_plot_data, fit_curves=True, ylabel="Non-bonded clash rate")
                save_figure(fig, os.path.join(plots_dir, "scaling_curves_nonbonded_clash"))
                print(f"Saved {plots_dir}/scaling_curves_nonbonded_clash.png")

    # Plot VSEPR metric scaling curves
    _vsepr_plot_specs = [
        ("angle_jsd", "Angle distribution JSD", "scaling_curves_angle_jsd"),
        ("torsional_out_of_bin_rate", "Torsional out-of-bin rate", "scaling_curves_torsion"),
        ("valence_overcoord_rate", "Valence overcoordination rate", "scaling_curves_valence"),
    ]
    for metric_key, ylabel, fname in _vsepr_plot_specs:
        plot_data_metric = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d[metric_key])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                plot_data_metric[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": d[metric_key][valid],
                }
        if plot_data_metric:
            with synthbench_style():
                fig = plot_scaling_curves(plot_data_metric, fit_curves=True, ylabel=ylabel)
                save_figure(fig, os.path.join(plots_dir, fname))
                print(f"Saved {plots_dir}/{fname}.png")

    # Plot sequence metric scaling curves
    _seq_plot_specs = [
        ("rg_error", "Radius of gyration error", "scaling_curves_rg_error"),
        ("seq_bond_violation_rate", "Bond violation rate (sequence)", "scaling_curves_seq_bond_viol"),
    ]
    for metric_key, ylabel, fname in _seq_plot_specs:
        plot_data_metric = {}
        for arch, d in arch_data.items():
            valid = np.isfinite(d[metric_key])
            if valid.any():
                display_name = arch_name_map.get(arch, arch)
                plot_data_metric[display_name] = {
                    "flops": d["flops"][valid],
                    "clash_rate": d[metric_key][valid],
                }
        if plot_data_metric:
            with synthbench_style():
                fig = plot_scaling_curves(plot_data_metric, fit_curves=True, ylabel=ylabel)
                save_figure(fig, os.path.join(plots_dir, fname))
                print(f"Saved {plots_dir}/{fname}.png")

    # Plot 2: Isoflop profiles (clash_rate vs model_size at each budget)
    all_results = data["all_results"]
    budgets_seen = sorted(set(r["budget"] for r in all_results if r["budget"] > 0))
    archs_seen = sorted(set(r["arch"] for r in all_results))
    size_order = ["xs", "small", "medium", "large", "xl"]

    if budgets_seen and archs_seen:
        with synthbench_style():
            n_budgets = len(budgets_seen)
            ncols = min(n_budgets, 3)
            nrows = math.ceil(n_budgets / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
            if n_budgets == 1:
                axes = np.array([axes])
            axes = np.atleast_2d(axes)

            for idx, budget in enumerate(budgets_seen):
                row, col = divmod(idx, ncols)
                ax = axes[row, col]

                for arch in archs_seen:
                    runs = [
                        r for r in all_results
                        if r["arch"] == arch and r["budget"] == budget
                    ]
                    if not runs:
                        continue

                    # Group by size, take best LR per size
                    best_by_size = {}
                    for r in runs:
                        s = r["size"]
                        if s not in best_by_size or r["best_clash_rate"] < best_by_size[s]["best_clash_rate"]:
                            best_by_size[s] = r

                    # Sort by size order
                    sizes_present = [s for s in size_order if s in best_by_size]
                    params = [best_by_size[s]["n_params"] for s in sizes_present]
                    crs = [best_by_size[s]["best_clash_rate"] for s in sizes_present]

                    display_name = arch_name_map.get(arch, arch)
                    color = ARCH_COLORS.get(display_name, "gray")
                    marker = ARCH_MARKERS.get(display_name, "x")
                    ax.plot(params, crs, marker=marker, color=color, label=display_name)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Parameters")
                ax.set_ylabel("Clash rate")
                ax.set_title(f"Budget = {budget:.0e} FLOPs")
                if idx == 0:
                    ax.legend(frameon=False, fontsize=8)

            # Hide unused axes
            for idx in range(n_budgets, nrows * ncols):
                row, col = divmod(idx, ncols)
                axes[row, col].set_visible(False)

            save_figure(fig, os.path.join(plots_dir, "isoflop_profiles"))
            print(f"Saved {plots_dir}/isoflop_profiles.png")


DATA_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]


def generate_data_grid(args):
    """Print training commands for data scaling experiment.

    Fixes model size and compute (max_steps), sweeps training set size.
    Strategy: generate one large dataset (100K), subsample at train time via
    train.max_train_samples — no need to re-run MCMC for each size.
    """
    archs = args.archs.split(",") if args.archs else list(SIZE_PRESETS.keys())
    sizes = args.sizes.split(",") if args.sizes else ["medium"]
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LEARNING_RATES
    data_sizes = [int(x) for x in args.data_sizes.split(",")] if args.data_sizes else DATA_SIZES
    max_steps = args.max_steps
    data_config = args.data or "medium_large"

    # Auto-select batch_size based on n_atoms in data config (N>=50 needs smaller batch)
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        cfg_path = os.path.join("configs", "data", f"{data_config}.yaml")
        try:
            import yaml
            with open(cfg_path) as f:
                data_cfg = yaml.safe_load(f)
            n_atoms = data_cfg.get("n_atoms", 10)
        except Exception:
            n_atoms = 10
        batch_size = 64 if n_atoms >= 50 else 256

    commands = []
    meta_runs = []
    for arch, size, lr, n_train in product(archs, sizes, lrs, data_sizes):
        if arch not in SIZE_PRESETS or size not in SIZE_PRESETS[arch]:
            continue
        preset = SIZE_PRESETS[arch][size]
        model_overrides = " ".join(f"model.model_kwargs.{k}={v}" for k, v in preset.items())
        run_name = f"{arch}_{size}_lr{lr:.0e}_N{n_train}"
        ckpt_dir = os.path.join(args.scaling_dir, run_name)
        cmd = (
            f"uv run python experiments/train.py "
            f"data={data_config} "
            f"model={arch} "
            f"model.size={size} "
            f"train.lr={lr} "
            f"train.max_steps={max_steps} "
            f"train.max_train_samples={n_train} "
            f"train.batch_size={batch_size} "
            f"{model_overrides} "
            f"checkpoint.dir={ckpt_dir} "
            f"logging.enabled={str(args.wandb).lower()} "
            f"hydra.run.dir={ckpt_dir}"
        )
        commands.append(cmd)
        meta_runs.append({"name": run_name, "arch": arch, "size": size, "lr": lr, "n_train": n_train})

    for cmd in commands:
        print(cmd)

    print(f"\n# Total: {len(commands)} runs", file=sys.stderr)
    print(f"# Data sizes: {data_sizes}", file=sys.stderr)
    print(f"# Archs: {archs}, Sizes: {sizes}, LRs: {lrs}", file=sys.stderr)
    print(f"# max_steps: {max_steps} (fixed — only N_train varies)", file=sys.stderr)

    meta_path = os.path.join(args.scaling_dir, "data_grid_meta.json")
    os.makedirs(args.scaling_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({"runs": meta_runs}, f, indent=2)
    print(f"\n# Grid metadata saved to {meta_path}", file=sys.stderr)


def run_data_grid(args):
    """Execute data scaling commands, skipping completed runs."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        generate_data_grid(args)

    commands = [
        line.strip()
        for line in buf.getvalue().strip().split("\n")
        if line.strip() and not line.startswith("#")
    ]

    remaining = []
    for cmd in commands:
        run_name = _extract_run_name_from_cmd(cmd)
        run_dir = os.path.join(args.scaling_dir, run_name)
        latest_pt = os.path.join(run_dir, "latest.pt")
        if os.path.isfile(latest_pt):
            try:
                data = torch.load(latest_pt, map_location="cpu", weights_only=False)
                saved_step = data.get("step", 0)
                max_steps = data.get("config", {}).get("train", {}).get("max_steps", 0)
                if max_steps > 0 and saved_step >= max_steps:
                    print(f"Skipping (completed {saved_step}/{max_steps}): {run_name}")
                    continue
                print(f"Resuming (incomplete {saved_step}/{max_steps}): {run_name}")
            except Exception as e:
                print(f"Warning: could not read {latest_pt}: {e}, will re-run")
        remaining.append(cmd)

    print(f"\n{len(commands)} total, {len(commands) - len(remaining)} done, {len(remaining)} remaining")
    _execute_commands(remaining, args.scaling_dir, args.n_gpus)


def collect_data_results(args):
    """Walk data scaling directory, collect best results per (arch, n_train)."""
    scaling_dir = args.scaling_dir
    if not os.path.isdir(scaling_dir):
        print(f"Directory not found: {scaling_dir}", file=sys.stderr)
        sys.exit(1)

    meta_path = os.path.join(scaling_dir, "data_grid_meta.json")
    run_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            for run in json.load(f).get("runs", []):
                run_meta[run["name"]] = run

    results = []
    for run_name in sorted(os.listdir(scaling_dir)):
        run_dir = os.path.join(scaling_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        ckpt_path = os.path.join(run_dir, "latest.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(run_dir, "best.pt")
        if not os.path.isfile(ckpt_path):
            continue
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = data.get("config", {})
            arch = config.get("model", {}).get("arch", "unknown")
            size = config.get("model", {}).get("size", "unknown")
            lr = config.get("train", {}).get("lr", 0)
            n_train = (
                run_meta.get(run_name, {}).get("n_train")
                or config.get("train", {}).get("max_train_samples")
                or 0
            )
            cr = data.get("best_clash_rate", float("inf"))
            grd = data.get("best_gr_distance", float("inf"))
            step = data.get("step", 0)
            results.append({
                "run": run_name,
                "arch": arch,
                "size": size,
                "lr": lr,
                "n_train": int(n_train) if n_train else 0,
                "best_clash_rate": cr,
                "best_gr_distance": grd,
                "step": step,
            })
        except Exception as e:
            print(f"Warning: failed to load {ckpt_path}: {e}", file=sys.stderr)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda r: (r["arch"], r["n_train"]))
    print(f"\n{'Run':<45} {'Arch':<12} {'N_train':>10} {'LR':<8} {'CR':>8} {'g(r)':>8}")
    print("-" * 95)
    for r in results:
        grd_str = f"{r['best_gr_distance']:>8.4f}" if r["best_gr_distance"] < float("inf") else "     n/a"
        print(f"{r['run']:<45} {r['arch']:<12} {r['n_train']:>10,} {r['lr']:<8.0e} "
              f"{r['best_clash_rate']:>8.4f} {grd_str}")

    # Best per (arch, n_train) — pick lowest clash rate across LRs
    best_per_size: dict = {}
    for r in results:
        key = (r["arch"], r["n_train"])
        if key not in best_per_size or r["best_clash_rate"] < best_per_size[key]["best_clash_rate"]:
            best_per_size[key] = r

    print(f"\nBest per (architecture, N_train):")
    print("-" * 65)
    print(f"{'Arch':<12} {'N_train':>10} {'Best CR':>10} {'Best g(r)':>10} {'LR':<8}")
    print("-" * 65)
    for (arch, n_train), r in sorted(best_per_size.items()):
        grd_str = f"{r['best_gr_distance']:>10.4f}" if r["best_gr_distance"] < float("inf") else "       n/a"
        print(f"{arch:<12} {n_train:>10,} {r['best_clash_rate']:>10.4f} {grd_str} {r['lr']:<8.0e}")

    out = {
        "all_results": results,
        "best_per_size": {f"{arch}_{n_train}": r for (arch, n_train), r in best_per_size.items()},
    }
    results_path = os.path.join(scaling_dir, "data_results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def fit_data_scaling(args):
    """Fit data scaling laws (clash_rate ∝ N^-β) and generate plots."""
    from viz import save_figure, synthbench_style
    from viz.scaling import fit_scaling_law, plot_data_scaling_curves

    scaling_dir = args.scaling_dir
    results_path = os.path.join(scaling_dir, "data_results.json")
    if not os.path.isfile(results_path):
        print(f"Results file not found: {results_path}", file=sys.stderr)
        print("Run 'data_collect' first.", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    arch_data: dict = {}
    eval_floor = 1.0 / 1000
    for r in data["best_per_size"].values():
        if not r["n_train"]:
            continue
        arch = r["arch"]
        if arch not in arch_data:
            arch_data[arch] = {"n_train": [], "clash_rate": []}
        arch_data[arch]["n_train"].append(r["n_train"])
        arch_data[arch]["clash_rate"].append(r["best_clash_rate"])

    arch_name_map = {"painn": "PaiNN", "transformer": "Transformer", "pairformer": "Pairformer"}
    for arch in arch_data:
        order = np.argsort(arch_data[arch]["n_train"])
        arch_data[arch]["n_train"] = np.array(arch_data[arch]["n_train"])[order]
        arch_data[arch]["clash_rate"] = np.clip(
            np.array(arch_data[arch]["clash_rate"])[order], eval_floor, None
        )

    print("\nData Scaling Law Fits:  clash_rate(N) = a * N^(-β) + floor")
    print("=" * 60)
    print(f"{'Architecture':<15} {'beta (β)':>10} {'prefactor':>12} {'floor':>10}")
    print("-" * 60)
    fits: dict = {}
    for arch, d in arch_data.items():
        n = np.array(d["n_train"], dtype=float)
        cr = np.array(d["clash_rate"], dtype=float)
        if len(n) < 3:
            print(f"{arch:<15} insufficient data ({len(n)} points)", file=sys.stderr)
            continue
        try:
            a, beta, floor = fit_scaling_law(n, cr)
            fits[arch] = {"a": a, "beta": beta, "floor": floor}
            print(f"{arch:<15} {beta:>10.3f} {a:>12.4f} {floor:>10.5f}")
        except RuntimeError as e:
            print(f"{arch:<15} fit failed: {e}", file=sys.stderr)

    fits_path = os.path.join(scaling_dir, "data_scaling_fits.json")
    with open(fits_path, "w") as f:
        json.dump(fits, f, indent=2)
    print(f"\nFits saved to {fits_path}")

    plots_dir = "outputs/plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_data = {arch_name_map.get(a, a): d for a, d in arch_data.items()}
    with synthbench_style():
        fig = plot_data_scaling_curves(plot_data, fit_curves=bool(fits))
        save_figure(fig, os.path.join(plots_dir, "data_scaling_curves"))
        print(f"Saved {plots_dir}/data_scaling_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Scaling law experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scaling_dir", default="outputs/scaling", help="Base directory for scaling outputs")
    common.add_argument("--archs", default=None, help="Comma-separated architectures (default: all)")
    common.add_argument("--sizes", default=None, help="Comma-separated sizes (default: all)")
    common.add_argument("--lrs", default=None, help="Comma-separated learning rates")
    common.add_argument("--budgets", default=None, help="Comma-separated FLOP budgets")
    common.add_argument("--batch_size", type=int, default=256, help="Batch size for FLOPs measurement")
    common.add_argument("--n_atoms", type=int, default=10, help="Number of atoms (auto-detected if --data set)")
    common.add_argument("--data", default=None, help="Hydra data config name (e.g. medium_large)")
    common.add_argument("--task", default=None,
                        choices=["hard_sphere", "chain", "vsepr", "sequence"],
                        help="Task type — sets default data config and primary metric")
    common.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    common.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs for parallel execution")

    # Compute scaling subcommands
    subparsers.add_parser("generate", parents=[common], help="Measure FLOPs and print grid commands")
    subparsers.add_parser("run", parents=[common], help="Execute scaling grid sequentially")

    collect_parser = subparsers.add_parser("collect", help="Collect results from completed runs")
    collect_parser.add_argument("--scaling_dir", default="outputs/scaling", help="Scaling directory")

    fit_parser = subparsers.add_parser("fit", help="Fit scaling laws and generate plots")
    fit_parser.add_argument("--scaling_dir", default="outputs/scaling", help="Scaling directory")

    # Data scaling subcommands (vary N_train, fix compute + model size)
    data_common = argparse.ArgumentParser(add_help=False)
    data_common.add_argument("--scaling_dir", default="outputs/data_scaling",
                             help="Base directory for data scaling outputs")
    data_common.add_argument("--archs", default=None, help="Comma-separated architectures (default: all)")
    data_common.add_argument("--sizes", default="medium", help="Comma-separated model sizes (default: medium)")
    data_common.add_argument("--lrs", default=None, help="Comma-separated learning rates (default: 1e-4,1e-3)")
    data_common.add_argument("--data", default=None, help="Hydra data config name (default: medium_large)")
    data_common.add_argument("--data_sizes", default=None,
                             help=f"Comma-separated N_train values (default: {DATA_SIZES})")
    data_common.add_argument("--max_steps", type=int, default=50_000,
                             help="Training steps per run — fixed across all data sizes (default: 50000)")
    data_common.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    data_common.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs for parallel execution")
    data_common.add_argument("--batch_size", type=int, default=None,
                             help="Batch size override (default: 64 for N>=50, 256 otherwise)")

    subparsers.add_parser("data_generate", parents=[data_common],
                          help="Print data scaling commands (vary N_train, fix compute)")
    subparsers.add_parser("data_run", parents=[data_common],
                          help="Execute data scaling commands")

    data_collect_parser = subparsers.add_parser("data_collect",
                                                help="Collect data scaling results")
    data_collect_parser.add_argument("--scaling_dir", default="outputs/data_scaling")

    data_fit_parser = subparsers.add_parser("data_fit",
                                            help="Fit data scaling laws and plot")
    data_fit_parser.add_argument("--scaling_dir", default="outputs/data_scaling")

    args = parser.parse_args()

    if args.command == "generate":
        generate_grid(args)
    elif args.command == "run":
        run_grid(args)
    elif args.command == "collect":
        collect_results(args)
    elif args.command == "fit":
        fit_scaling(args)
    elif args.command == "data_generate":
        generate_data_grid(args)
    elif args.command == "data_run":
        run_data_grid(args)
    elif args.command == "data_collect":
        collect_data_results(args)
    elif args.command == "data_fit":
        fit_data_scaling(args)


if __name__ == "__main__":
    main()
