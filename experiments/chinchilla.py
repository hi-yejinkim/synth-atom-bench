"""Chinchilla-style scaling law study: Approach 1 (isoFLOP) + Approach 3 (parametric fit).

Experimental grid:
  Architectures : PaiNN, Transformer, Pairformer
  Model sizes   : 5 sizes per arch (log-spaced, calibrated for 50K-sample datasets)
                  (keys: chinchilla_1, chinchilla_3, chinchilla_5, chinchilla_7, chinchilla_9)
                  Param range: ~2K–11M (spans underfit→optimal→overfit for L(N,D) fitting)
  Data budgets  : 4 snapshots — 50K, 100K, 500K, 1M total training samples
  LR sweep      : 2 values per run (best selected at collect time)
  Total runs    : 3 × 5 × 2 = 30 training commands per task (each runs to D4, snapshots at D1–D4)

Training constraints (per Chinchilla paper):
  - Exactly 1 epoch over assigned D budget (max_steps = D / batch_size)
  - CosineAnnealingLR with T_max = total_steps, min LR = 0.1 × initial LR (warmup_fraction=0)
  - FLOPs tracked via torch.utils.flop_counter.FlopCounterMode

Data collection:
  - trajectory.jsonl: one JSON line per eval step (D_seen, violation_rate, total_flops, n_params)
  - results.json: aggregated per-task trajectories + best-LR selection
  - scaling_results.csv: final CSV with [arch, n_params, D_tokens, total_flops, violation_rate]

Post-processing:
  - Approach 1: IsoFLOP envelope — best violation_rate vs total FLOPs per arch
  - Approach 3: Parametric fit L(N,D) = E + A/N^α + B/D^β per arch
    → Optimal allocation exponents: a = β/(α+β), b = α/(α+β)

Subcommands:
  generate  -- measure FLOPs/params per (arch, size), emit grid_meta.json + shell commands
  run       -- execute training commands from generate (sequential or parallel across GPUs)
  collect   -- aggregate trajectory.jsonl files into results.json + CSV per task
  fit       -- fit L(N,D) = E + A/N^alpha + B/D^beta per (task, arch), write fits.json
  plot      -- generate all Approach 1 + 3 figures

Usage:
  uv run python experiments/chinchilla.py generate --tasks vsepr_sp3 --archs painn,transformer,pairformer
  uv run python experiments/chinchilla.py run      --tasks vsepr_sp3 --n_gpus 4
  uv run python experiments/chinchilla.py collect  --tasks vsepr_sp3
  uv run python experiments/chinchilla.py fit      --tasks vsepr_sp3
  uv run python experiments/chinchilla.py plot     --tasks vsepr_sp3

Recommended entry tasks: sphere_N50 and chain_N50 (N=50 atoms)
  - N=50 gives O(N²)=2500 attention pairs → FLOPs scale meaningfully with model size
  - Architectural biases (local/global attention, pair representation) matter at N=50
  - sphere_N50: pure clash avoidance (simplest constraint, cleanest scaling signal)
  - chain_N50:  bond + clash constraints (one step harder, tests connectivity learning)
  - Unified violation_rate: clash_rate (sphere) / bond_violation_rate (chain)

Why NOT N=5 (vsepr_sp3) for Chinchilla:
  - O(N²)=25 attention pairs → trivially fast for any model size
  - 100M-param model trains in ~1 min: compute budget differences are negligible
  - Architectural inductive biases don't differentiate at this scale
"""

from __future__ import annotations

import argparse

from experiments.chinchilla_lib.config import (
    ALL_ARCHS, CHINCHILLA_5_SIZES, CHINCHILLA_SIZES, D_TARGETS, LRS,
)
from experiments.chinchilla_lib.generate import generate
from experiments.chinchilla_lib.run import run
from experiments.chinchilla_lib.collect import collect
from experiments.chinchilla_lib.fit import fit, fit_approach1
from experiments.chinchilla_lib.plot import plot, plot_T


# ── Helpers ───────────────────────────────────────────────────────────────

def _expand_nbody_temps(args: argparse.Namespace) -> None:
    """Expand --nbody_temps into concrete task IDs and merge into --tasks.

    If --nbody_temps is given (e.g. "0.5,1.0,2.0") with --nbody_base
    (e.g. "nbody_n15_b2"), generate task IDs like nbody_n15_b2_T0.5, etc.
    These are appended to --tasks (or replace the default if --tasks was
    not explicitly provided).
    """
    temps = getattr(args, "nbody_temps", None)
    if not temps:
        return
    base = getattr(args, "nbody_base", "nbody_n15_b2")
    t_values = [t.strip() for t in temps.split(",")]
    nbody_tasks = [f"{base}_T{t}" for t in t_values]

    existing = [t.strip() for t in args.tasks.split(",")]
    # If tasks is still the default, replace; otherwise append
    default_tasks = {"sphere_N50", "chain_N50"}
    if set(existing) == default_tasks:
        args.tasks = ",".join(nbody_tasks)
    else:
        args.tasks = ",".join(existing + nbody_tasks)


# ── CLI ───────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chinchilla-style scaling law study (Approaches 1 & 3)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--tasks",
        default="sphere_N50,chain_N50",  # large-N by default
        help=(
            "Comma-separated task IDs. "
            "Large-N primary tasks: sphere_N50,chain_N50. "
            "Diagnostic small-N tasks: sphere_easy,chain_N10,vsepr_sp3,sequence_linear."
        ),
    )
    shared.add_argument("--archs",         default=",".join(ALL_ARCHS))
    shared.add_argument(
        "--sizes",
        default=",".join(CHINCHILLA_5_SIZES),
        help=(
            "Comma-separated size keys. "
            f"5-size preset (default): {','.join(CHINCHILLA_5_SIZES)}. "
            f"10-size fine-grained: {','.join(CHINCHILLA_SIZES)}."
        ),
    )
    shared.add_argument("--lrs",           default=",".join(str(lr) for lr in LRS))
    shared.add_argument("--chinchilla_dir", default="outputs/chinchilla")
    shared.add_argument("--wandb",         action="store_true")
    shared.add_argument(
        "--nbody_temps", default=None,
        help=(
            "Comma-separated temperature values for n-body T sweep. "
            "Generates task IDs {nbody_base}_T{t} for each T. "
            "Example: --nbody_temps 0.5,1.0,2.0"
        ),
    )
    shared.add_argument(
        "--nbody_base", default="nbody_n15_b2",
        help=(
            "Base name for n-body tasks (used with --nbody_temps). "
            "Default: nbody_n15_b2. "
            "Example: nbody_n20_b2_hw for hard-wall boundary."
        ),
    )

    # generate
    p_gen = sub.add_parser("generate", parents=[shared], help="Emit training commands")
    p_gen.add_argument(
        "--d_targets", default=None,
        help=(
            "Comma-separated data budgets (samples) to sweep. "
            "Each budget trains for exactly epochs*D/batch_size steps. "
            f"Default: {','.join(str(d) for d in D_TARGETS)}. "
            "Example for small-data sweep: --d_targets 10000,20000,50000,100000"
        ),
    )
    p_gen.add_argument(
        "--epochs", type=int, default=1,
        help=(
            "Number of passes over each D-budget dataset. "
            "total_steps = epochs * D / batch_size. "
            "T_max for cosine annealing = total_steps (schedule is still per-run). "
            "Use --epochs 50 for convergence on bounded tasks like sphere_N50. "
            "Default: 1 (original 1-epoch Chinchilla regime)."
        ),
    )
    p_gen.set_defaults(func=generate)

    # run
    p_run = sub.add_parser("run", parents=[shared], help="Execute training grid")
    p_run.add_argument("--n_gpus", type=int, default=1)
    p_run.add_argument(
        "--d_targets", default=None,
        help="Comma-separated data budgets (must match the generate call). "
             "Example: --d_targets 10000,20000,50000,100000",
    )
    p_run.add_argument(
        "--epochs", type=int, default=1,
        help="Number of passes over each D-budget (must match the generate call).",
    )
    p_run.set_defaults(func=run)

    # collect
    p_col = sub.add_parser("collect", parents=[shared], help="Aggregate trajectory.jsonl")
    p_col.set_defaults(func=collect)

    # fit (Approach 3: parametric L(N,D) model)
    p_fit = sub.add_parser("fit", parents=[shared], help="Fit L(N,D) parametric law (Approach 3)")
    p_fit.set_defaults(func=fit)

    # fit_approach1 (Approach 1: isoFLOP envelope → empirical power laws)
    p_fit1 = sub.add_parser(
        "fit_approach1", parents=[shared],
        help="Approach 1: isoFLOP envelope → N*(C), D*(C) power-law fit",
    )
    p_fit1.set_defaults(func=fit_approach1)

    # plot
    p_plt = sub.add_parser("plot", parents=[shared], help="Generate all figures")
    p_plt.add_argument("--plots_dir", default="outputs/plots/chinchilla")
    p_plt.set_defaults(func=plot)

    # plot_T — T-axis comparison across temperatures (n-body)
    p_plt_t = sub.add_parser(
        "plot_T", parents=[shared],
        help="Compare n-body scaling across temperatures (T sweep)",
    )
    p_plt_t.add_argument("--plots_dir", default="outputs/plots/chinchilla")
    p_plt_t.set_defaults(func=plot_T)

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    _expand_nbody_temps(args)
    args.func(args)


if __name__ == "__main__":
    main()
