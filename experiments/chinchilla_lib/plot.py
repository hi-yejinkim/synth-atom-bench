"""Plot subcommand: generate all Approach 1 + Approach 3 figures."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from experiments.chinchilla_lib.helpers import (
    _fits_approach1_path, _fits_path, _results_path,
)


def _infer_d_unique_labels(chinchilla_dir: str, task_id: str) -> dict[str, str]:
    """Infer unique dataset size labels from run configs (overrides.yaml).

    Reads max_train_samples from a sample run to reconstruct
    D1=10K, D2=20K, etc. labels. Falls back to empty dict.
    """
    import glob
    import yaml
    task_dir = os.path.join(chinchilla_dir, task_id)
    # Find any overrides.yaml to read max_train_samples per D level
    labels: dict[str, str] = {}
    for d_name in ["D1", "D2", "D3", "D4"]:
        pattern = os.path.join(task_dir, "**", d_name, ".hydra", "overrides.yaml")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            continue
        try:
            with open(matches[0]) as f:
                overrides = yaml.safe_load(f)
            for entry in overrides:
                if "max_train_samples=" in str(entry):
                    val = int(str(entry).split("=")[1])
                    if val >= 1_000_000:
                        labels[d_name] = f"{val/1e6:.0f}M"
                    elif val >= 1_000:
                        labels[d_name] = f"{val//1000}K"
                    else:
                        labels[d_name] = str(val)
                    break
        except Exception:
            continue
    return labels


def plot(args: argparse.Namespace) -> None:
    """Generate all Approach 1 + Approach 3 figures.

    Output organization (per task):
        {plots_dir}/{task_id}/
        +-- a1_isoflop_{arch}.png          # IsoFLOP profiles: VR vs N per D budget
        +-- a1_compute_frontier.png        # Compute-perf envelope (raw progressive-min)
        +-- a1_compute_frontier_fit.png    # Compute-perf envelope (parametric fit)
        +-- a1_optimal_allocation.png      # N*(C) and D*(C) from envelope
        +-- a1_vr_by_data.png              # VR vs FLOPs colored by data budget
        +-- a1_vr_by_params.png            # VR vs FLOPs colored by model size
        +-- a1_nd_regime_{arch}.png        # N-D scatter with VR + regime boundary
        +-- a1_arch_comparison.png         # All archs on same axes per D budget
        +-- a3_loss_surface_{arch}.png     # L(N,D) contour with scatter overlay
        +-- a3_optimal_allocation.png      # N*(C) D*(C) from parametric fit
        +-- training_trajectories_{arch}.png  # D_seen vs VR during training

    Cross-task (in plots_dir root):
        +-- heatmap_{key}.png
        +-- cross_task_summary.png
    """
    from viz.chinchilla import (
        plot_isoflop_curves,
        plot_isoflop_envelope,
        plot_training_trajectories,
        plot_arch_comparison,
        plot_loss_surface,
        plot_scaling_exponent_heatmap,
        plot_optimal_allocation,
        plot_optimal_allocation_approach1,
        plot_N_D_regime_map,
        plot_cross_task_summary,
        plot_smooth_envelope,
        plot_optimal_ND_from_envelope,
        plot_vr_vs_flops_by_data,
        plot_vr_vs_flops_by_params,
    )
    from viz.style import save_figure, synthbench_style

    chinchilla_dir = args.chinchilla_dir
    plots_base = args.plots_dir
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Gather all fits for cross-task summary
    all_fits: dict[str, dict] = {}

    for task_id in tasks:
        res_path = _results_path(chinchilla_dir, task_id)
        fits_path = _fits_path(chinchilla_dir, task_id)
        if not os.path.exists(res_path):
            print(f"[SKIP] No results.json for '{task_id}'", file=sys.stderr)
            continue

        with open(res_path) as f:
            results = json.load(f)
        trajectories = results.get("all_trajectories", [])
        best_by_size = results.get("best_by_size_d", {})

        fits: dict[str, dict] = {}
        if os.path.exists(fits_path):
            with open(fits_path) as f:
                fits = json.load(f).get("fits", {})
            all_fits[task_id] = fits

        fits_a1: dict[str, dict] = {}
        fits_a1_path = _fits_approach1_path(chinchilla_dir, task_id)
        if os.path.exists(fits_a1_path):
            with open(fits_a1_path) as f:
                fits_a1 = json.load(f).get("fits", {})

        plots_dir = os.path.join(plots_base, task_id)
        os.makedirs(plots_dir, exist_ok=True)

        with synthbench_style():
            # ── Approach 1: IsoFLOP ──────────────────────────────────────

            # A1: IsoFLOP profiles -- VR vs N per D-budget (one fig per arch)
            figs = plot_isoflop_curves(best_by_size, task_id)
            for arch, fig in figs.items():
                save_figure(fig, os.path.join(plots_dir, f"a1_isoflop_{arch}"))
                print(f"  a1_isoflop_{arch}  ({task_id})")

            # A1: Compute-performance frontier -- raw progressive-min envelope
            if best_by_size:
                fig_env = plot_isoflop_envelope(best_by_size, task_id)
                save_figure(fig_env, os.path.join(plots_dir, "a1_compute_frontier"))
                print(f"  a1_compute_frontier  ({task_id})")

            # A1: Compute-performance frontier -- parametric VR(C) = E+A*C^(-alpha) fit
            if best_by_size:
                fig_se = plot_smooth_envelope(best_by_size, task_id)
                save_figure(fig_se, os.path.join(plots_dir, "a1_compute_frontier_fit"))
                print(f"  a1_compute_frontier_fit  ({task_id})")

            # A1: Optimal N*(C) and D*(C) from envelope
            if best_by_size:
                fig_nd = plot_optimal_ND_from_envelope(best_by_size, task_id)
                save_figure(fig_nd, os.path.join(plots_dir, "a1_optimal_allocation"))
                print(f"  a1_optimal_allocation  ({task_id})")

            # A1: VR vs FLOPs -- colored by data budget
            if best_by_size:
                _d_uniq = _infer_d_unique_labels(chinchilla_dir, task_id)
                fig_vrd = plot_vr_vs_flops_by_data(
                    best_by_size, task_id, d_unique_labels=_d_uniq)
                save_figure(fig_vrd, os.path.join(plots_dir, "a1_vr_by_data"))
                print(f"  a1_vr_by_data  ({task_id})")

            # A1: VR vs FLOPs -- colored by model size
            if best_by_size:
                fig_vrp = plot_vr_vs_flops_by_params(best_by_size, task_id)
                save_figure(fig_vrp, os.path.join(plots_dir, "a1_vr_by_params"))
                print(f"  a1_vr_by_params  ({task_id})")

            # A1: N-D regime map (scatter with VR coloring + regime boundary)
            if best_by_size:
                figs_nd = plot_N_D_regime_map(best_by_size, task_id, fits=fits or None)
                for arch, fig_nd in figs_nd.items():
                    save_figure(fig_nd, os.path.join(plots_dir, f"a1_nd_regime_{arch}"))
                    print(f"  a1_nd_regime_{arch}  ({task_id})")

            # A1: Arch comparison -- all archs on same axes per D budget
            if best_by_size:
                fig3 = plot_arch_comparison(best_by_size, task_id)
                save_figure(fig3, os.path.join(plots_dir, "a1_arch_comparison"))
                print(f"  a1_arch_comparison  ({task_id})")

            # A1: Optimal allocation from fit_approach1 (if available)
            if fits_a1:
                fits_a1_valid = {k: v for k, v in fits_a1.items()
                                 if v.get("fit_available", True) and "n_exp" in v}
                if fits_a1_valid:
                    compute_budgets = np.geomspace(1e12, 1e17, 200)
                    fig_a1 = plot_optimal_allocation_approach1(
                        fits_a1_valid, task_id, compute_budgets)
                    save_figure(fig_a1, os.path.join(plots_dir, "a1_optimal_allocation_fit"))
                    print(f"  a1_optimal_allocation_fit  ({task_id})")

            # ── Approach 3: Parametric L(N,D) fit ────────────────────────

            # A3: L(N,D) contour per arch
            for arch, fparams in fits.items():
                fig_s = plot_loss_surface(best_by_size, fparams, task_id, arch)
                save_figure(fig_s, os.path.join(plots_dir, f"a3_loss_surface_{arch}"))
                print(f"  a3_loss_surface_{arch}  ({task_id})")

            # A3: N*(C) and D*(C) from parametric fit
            if fits:
                compute_budgets = np.geomspace(1e12, 1e17, 200)
                fig_alloc = plot_optimal_allocation(fits, task_id, compute_budgets)
                save_figure(fig_alloc, os.path.join(plots_dir, "a3_optimal_allocation"))
                print(f"  a3_optimal_allocation  ({task_id})")

            # ── Training diagnostics ─────────────────────────────────────

            figs2 = plot_training_trajectories(trajectories, task_id)
            for arch, fig in figs2.items():
                save_figure(fig, os.path.join(plots_dir, f"training_trajectories_{arch}"))
                print(f"  training_trajectories_{arch}  ({task_id})")

    # ── Cross-task summary (requires >=2 tasks) ──────────────────────────
    if len(all_fits) >= 2:
        os.makedirs(plots_base, exist_ok=True)
        with synthbench_style():
            for exp_key in ["alpha", "beta", "N_exponent"]:
                fig_hm = plot_scaling_exponent_heatmap(all_fits, exponent_key=exp_key)
                save_figure(fig_hm, os.path.join(plots_base, f"heatmap_{exp_key}"))
                print(f"  heatmap_{exp_key}")

            fig_ct = plot_cross_task_summary(all_fits)
            save_figure(fig_ct, os.path.join(plots_base, "cross_task_summary"))
            print("  cross_task_summary")
