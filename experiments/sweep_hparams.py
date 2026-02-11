"""Hyperparameter sweep orchestrator: generate, run, and summarize experiments."""

import argparse
import json
import os
import subprocess
import sys
from itertools import product

import numpy as np

# Model size presets: (name, kwargs_overrides)
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

LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def build_hydra_overrides(arch: str, size: str, lr: float, max_steps: int, output_dir: str) -> list[str]:
    """Build Hydra CLI override strings for a sweep run."""
    overrides = [
        f"model={arch}",
        f"model.size={size}",
        f"train.lr={lr}",
        f"train.max_steps={max_steps}",
    ]
    # Model size overrides
    preset = SIZE_PRESETS[arch][size]
    for key, val in preset.items():
        overrides.append(f"model.model_kwargs.{key}={val}")
    return overrides


def generate_commands(args):
    """Print training commands for the sweep grid."""
    archs = args.archs.split(",") if args.archs else list(SIZE_PRESETS.keys())
    sizes = args.sizes.split(",") if args.sizes else ["small", "medium", "large"]
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LEARNING_RATES
    max_steps = args.max_steps

    commands = []
    for arch, size, lr in product(archs, sizes, lrs):
        if arch not in SIZE_PRESETS:
            print(f"Warning: skipping unknown arch {arch}", file=sys.stderr)
            continue
        if size not in SIZE_PRESETS[arch]:
            print(f"Warning: skipping unknown size {size} for {arch}", file=sys.stderr)
            continue

        overrides = build_hydra_overrides(arch, size, lr, max_steps, args.sweep_dir)
        run_name = f"{arch}_{size}_lr{lr:.0e}"
        override_str = " ".join(overrides)

        # Override checkpoint dir to keep runs separate
        ckpt_dir = os.path.join(args.sweep_dir, run_name)
        cmd = (
            f"uv run python experiments/train.py {override_str} "
            f"checkpoint.dir={ckpt_dir} "
            f"logging.enabled={str(args.wandb).lower()} "
            f"hydra.run.dir={ckpt_dir}"
        )
        commands.append((run_name, cmd))

    for name, cmd in commands:
        print(cmd)

    print(f"\n# Total: {len(commands)} runs", file=sys.stderr)


def run_sweep(args):
    """Execute sweep commands sequentially."""
    # Capture commands from generate
    args_copy = argparse.Namespace(**vars(args))
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        generate_commands(args_copy)

    commands = [line.strip() for line in buf.getvalue().strip().split("\n") if line.strip() and not line.startswith("#")]

    print(f"Running {len(commands)} sweep jobs...")
    for i, cmd in enumerate(commands):
        print(f"\n{'='*60}")
        print(f"Job {i+1}/{len(commands)}: {cmd}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Warning: job {i+1} exited with code {result.returncode}", file=sys.stderr)


def summarize_sweep(args):
    """Walk sweep directory, collect results from best.pt checkpoints."""
    import torch

    sweep_dir = args.sweep_dir
    if not os.path.isdir(sweep_dir):
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for run_name in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, run_name)
        # Look for checkpoint in outputs/checkpoints/{arch}/ or directly
        best_pt = None
        for candidate in [
            os.path.join(run_dir, "best.pt"),
            os.path.join(run_dir, "outputs", "checkpoints", "best.pt"),
        ]:
            if os.path.isfile(candidate):
                best_pt = candidate
                break

        # Also search subdirectories for best.pt
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

            # Estimate param count from model_kwargs
            results.append({
                "run": run_name,
                "arch": arch,
                "lr": lr,
                "best_clash_rate": cr,
                "step": step,
                "model_kwargs": model_kwargs,
            })
        except Exception as e:
            print(f"Warning: failed to load {best_pt}: {e}", file=sys.stderr)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    results.sort(key=lambda r: (r["arch"], r["best_clash_rate"]))
    print(f"\n{'Run':<35} {'Arch':<12} {'LR':<10} {'Clash Rate':<12} {'Step':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['run']:<35} {r['arch']:<12} {r['lr']:<10.1e} {r['best_clash_rate']:<12.4f} {r['step']:<8}")

    # Best per architecture
    print(f"\nBest per architecture:")
    print("-" * 50)
    seen = set()
    for r in results:
        if r["arch"] not in seen:
            seen.add(r["arch"])
            print(f"  {r['arch']:<12} clash_rate={r['best_clash_rate']:.4f} (lr={r['lr']:.1e})")

    # Save summary JSON
    summary_path = os.path.join(sweep_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for velocity networks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sweep_dir", default="outputs/sweep", help="Base directory for sweep outputs")
    common.add_argument("--max_steps", type=int, default=100000, help="Training steps per run")
    common.add_argument("--archs", default=None, help="Comma-separated architectures (default: all)")
    common.add_argument("--sizes", default=None, help="Comma-separated sizes (default: small,medium,large)")
    common.add_argument("--lrs", default=None, help="Comma-separated learning rates")
    common.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # Subcommands
    gen_parser = subparsers.add_parser("generate", parents=[common], help="Print sweep commands")
    run_parser = subparsers.add_parser("run", parents=[common], help="Execute sweep sequentially")
    sum_parser = subparsers.add_parser("summarize", help="Summarize sweep results")
    sum_parser.add_argument("--sweep_dir", default="outputs/sweep", help="Sweep directory")

    args = parser.parse_args()

    if args.command == "generate":
        generate_commands(args)
    elif args.command == "run":
        run_sweep(args)
    elif args.command == "summarize":
        summarize_sweep(args)


if __name__ == "__main__":
    main()
