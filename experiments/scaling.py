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

from models.painn import PaiNNVelocityNetwork
from models.pairformer import PairformerVelocityNetwork
from models.transformer import TransformerVelocityNetwork

MODEL_REGISTRY = {
    "painn": PaiNNVelocityNetwork,
    "transformer": TransformerVelocityNetwork,
    "pairformer": PairformerVelocityNetwork,
}

SIZE_PRESETS = {
    "painn": {
        "xs": {"hidden_dim": 16, "n_layers": 2},
        "small": {"hidden_dim": 32, "n_layers": 3},
        "medium": {"hidden_dim": 128, "n_layers": 5},
        "large": {"hidden_dim": 256, "n_layers": 8},
        "xl": {"hidden_dim": 512, "n_layers": 10},
    },
    "transformer": {
        "xs": {"hidden_dim": 32, "num_layers": 2, "num_heads": 2},
        "small": {"hidden_dim": 64, "num_layers": 3, "num_heads": 4},
        "medium": {"hidden_dim": 128, "num_layers": 6, "num_heads": 8},
        "large": {"hidden_dim": 256, "num_layers": 8, "num_heads": 8},
        "xl": {"hidden_dim": 384, "num_layers": 10, "num_heads": 8},
    },
    "pairformer": {
        "xs": {"hidden_dim": 32, "pair_dim": 16, "num_layers": 1, "num_heads": 2},
        "small": {"hidden_dim": 64, "pair_dim": 32, "num_layers": 2, "num_heads": 4},
        "medium": {"hidden_dim": 128, "pair_dim": 64, "num_layers": 4, "num_heads": 8},
        "large": {"hidden_dim": 256, "pair_dim": 128, "num_layers": 6, "num_heads": 8},
        "xl": {"hidden_dim": 384, "pair_dim": 192, "num_layers": 8, "num_heads": 8},
    },
}

# Default configs for model kwargs not in SIZE_PRESETS
MODEL_DEFAULTS = {
    "painn": {"n_rbf": 20, "cutoff": 10.0},
    "transformer": {"num_rbf": 64, "cutoff": 10.0, "mlp_ratio": 4.0},
    "pairformer": {"num_rbf": 64, "cutoff": 10.0, "expansion_factor": 4.0},
}

BUDGETS = [1e15, 4e15, 1.6e16, 6.4e16, 2.56e17]
LEARNING_RATES = [3e-4, 1e-3]
ALL_SIZES = ["xs", "small", "medium", "large", "xl"]

MIN_STEPS = 2000
MAX_STEPS = 1_000_000


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
    for arch, size, budget, lr in product(archs, sizes, budgets, lrs):
        flops_per_step = flops_table[(arch, size)]
        max_steps = int(budget / flops_per_step)

        if max_steps < MIN_STEPS:
            skipped += 1
            continue
        if max_steps > MAX_STEPS:
            skipped += 1
            continue

        run_name = f"{arch}_{size}_lr{lr:.0e}_budget{budget:.0e}"
        ckpt_dir = os.path.join(args.scaling_dir, run_name)

        # Build Hydra overrides
        preset = SIZE_PRESETS[arch][size]
        model_overrides = " ".join(f"model.model_kwargs.{k}={v}" for k, v in preset.items())

        cmd = (
            f"uv run python experiments/train.py "
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


def run_grid(args):
    """Execute scaling grid commands sequentially."""
    import io
    from contextlib import redirect_stdout

    # Generate commands
    buf = io.StringIO()
    with redirect_stdout(buf):
        generate_grid(args)

    commands = [
        line.strip()
        for line in buf.getvalue().strip().split("\n")
        if line.strip() and not line.startswith("#")
    ]

    print(f"Running {len(commands)} scaling jobs...")
    for i, cmd in enumerate(commands):
        print(f"\n{'='*60}")
        print(f"Job {i+1}/{len(commands)}: {cmd}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Warning: job {i+1} exited with code {result.returncode}", file=sys.stderr)


def collect_results(args):
    """Walk scaling directory, load best.pt from each run, save results.json."""
    scaling_dir = args.scaling_dir
    if not os.path.isdir(scaling_dir):
        print(f"Scaling directory not found: {scaling_dir}", file=sys.stderr)
        sys.exit(1)

    # Load grid metadata if available
    meta_path = os.path.join(scaling_dir, "grid_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Build lookup from run name to metadata
    run_meta = {}
    for r in meta.get("runs", []):
        run_meta[r["name"]] = r

    results = []
    for run_name in sorted(os.listdir(scaling_dir)):
        run_dir = os.path.join(scaling_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        # Find best.pt
        best_pt = None
        for candidate in [
            os.path.join(run_dir, "best.pt"),
        ]:
            if os.path.isfile(candidate):
                best_pt = candidate
                break
        if best_pt is None:
            for root, dirs, files in os.walk(run_dir):
                if "best.pt" in files:
                    best_pt = os.path.join(root, "best.pt")
                    break
        if best_pt is None:
            continue

        try:
            data = torch.load(best_pt, map_location="cpu", weights_only=False)
            config = data.get("config", {})
            arch = config.get("model", {}).get("arch", "unknown")
            model_kwargs = config.get("model", {}).get("model_kwargs", {})
            lr = config.get("train", {}).get("lr", 0)
            cr = data.get("best_clash_rate", float("inf"))
            step = data.get("step", 0)

            # Get metadata from grid
            rm = run_meta.get(run_name, {})
            budget = rm.get("budget", 0)
            flops_per_step = rm.get("flops_per_step", 0)
            size = rm.get("size", "unknown")
            n_params = meta.get("params_table", {}).get(f"{arch}_{size}", 0)

            # Compute actual total FLOPs
            total_flops = flops_per_step * step if flops_per_step else 0

            results.append({
                "run": run_name,
                "arch": arch,
                "size": size,
                "lr": lr,
                "budget": budget,
                "best_clash_rate": cr,
                "step": step,
                "n_params": n_params,
                "flops_per_step": flops_per_step,
                "total_flops": total_flops,
                "model_kwargs": model_kwargs,
            })
        except Exception as e:
            print(f"Warning: failed to load {best_pt}: {e}", file=sys.stderr)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Print all results
    results.sort(key=lambda r: (r["arch"], r["budget"], r["best_clash_rate"]))
    print(f"\n{'Run':<45} {'Arch':<12} {'Size':<6} {'LR':<8} {'Budget':>10} {'CR':>8} {'Step':>8}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['run']:<45} {r['arch']:<12} {r['size']:<6} {r['lr']:<8.0e} "
            f"{r['budget']:>10.0e} {r['best_clash_rate']:>8.4f} {r['step']:>8}"
        )

    # Best per (arch, budget): select lowest clash rate
    best_per_budget = {}
    for r in results:
        key = (r["arch"], r["budget"])
        if key not in best_per_budget or r["best_clash_rate"] < best_per_budget[key]["best_clash_rate"]:
            best_per_budget[key] = r

    print(f"\nBest per (architecture, budget):")
    print("-" * 80)
    print(f"{'Arch':<12} {'Budget':>10} {'Best CR':>10} {'Size':<6} {'LR':<8} {'Params':>10}")
    print("-" * 80)
    for (arch, budget), r in sorted(best_per_budget.items()):
        print(
            f"{arch:<12} {budget:>10.0e} {r['best_clash_rate']:>10.4f} "
            f"{r['size']:<6} {r['lr']:<8.0e} {r['n_params']:>10,}"
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

    # Organize by architecture
    arch_data = {}
    for key, r in best_per_budget.items():
        arch = r["arch"]
        if arch not in arch_data:
            arch_data[arch] = {"flops": [], "clash_rate": []}
        arch_data[arch]["flops"].append(r["budget"])
        arch_data[arch]["clash_rate"].append(r["best_clash_rate"])

    # Sort by budget within each arch
    for arch in arch_data:
        order = np.argsort(arch_data[arch]["flops"])
        arch_data[arch]["flops"] = np.array(arch_data[arch]["flops"])[order]
        arch_data[arch]["clash_rate"] = np.array(arch_data[arch]["clash_rate"])[order]

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

    # Save fits
    fits_path = os.path.join(scaling_dir, "scaling_fits.json")
    with open(fits_path, "w") as f:
        json.dump(fits, f, indent=2)
    print(f"\nFits saved to {fits_path}")

    # Plot 1: Scaling curves
    plots_dir = "outputs/plots"
    os.makedirs(plots_dir, exist_ok=True)

    with synthbench_style():
        fig = plot_scaling_curves(plot_data, fit_curves=True)
        save_figure(fig, os.path.join(plots_dir, "scaling_curves"))
        print(f"Saved {plots_dir}/scaling_curves.png")

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
    common.add_argument("--n_atoms", type=int, default=10, help="Number of atoms")
    common.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # Subcommands
    subparsers.add_parser("generate", parents=[common], help="Measure FLOPs and print grid commands")
    subparsers.add_parser("run", parents=[common], help="Execute scaling grid sequentially")

    collect_parser = subparsers.add_parser("collect", help="Collect results from completed runs")
    collect_parser.add_argument("--scaling_dir", default="outputs/scaling", help="Scaling directory")

    fit_parser = subparsers.add_parser("fit", help="Fit scaling laws and generate plots")
    fit_parser.add_argument("--scaling_dir", default="outputs/scaling", help="Scaling directory")

    args = parser.parse_args()

    if args.command == "generate":
        generate_grid(args)
    elif args.command == "run":
        run_grid(args)
    elif args.command == "collect":
        collect_results(args)
    elif args.command == "fit":
        fit_scaling(args)


if __name__ == "__main__":
    main()
