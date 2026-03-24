"""Plot subcommand + all Chinchilla visualization functions.

Approach 1 (isoFLOP profiles):
  plot_isoflop_curves        -- N vs violation_rate per D budget
  plot_training_trajectories -- D_seen vs violation_rate per model size
  plot_arch_comparison       -- all 3 archs on same axes per D budget
  plot_smooth_envelope       -- smooth compute-perf frontier (sigmoid fit)
  plot_optimal_ND_from_envelope -- N*(C) and D*(C) from frontier envelope
  plot_vr_vs_flops_by_data   -- VR vs FLOPs per arch, colored by data budget
  plot_vr_vs_flops_by_params -- VR vs FLOPs per arch, colored by model size

Approach 3 (parametric L(N,D) fit):
  plot_loss_surface          -- contour of L(N,D) with scatter overlay
  plot_scaling_exponent_heatmap -- α / β heatmap: archs × tasks
  plot_optimal_allocation    -- N*(C) and D*(C) curves per arch
  plot_cross_task_summary    -- α vs β scatter for all (task, arch)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from viz.style import ARCH_COLORS, ARCH_MARKERS, SINGLE_COL, DOUBLE_COL

from experiments.chinchilla_lib.helpers import (
    _fits_approach1_path, _fits_path, _results_path,
)


# ── Constants ─────────────────────────────────────────────────────────────

# Consistent floor for VR=0 in all log operations and plots
_VR_FLOOR = 1e-6

_ARCH_DISPLAY = {
    "painn": "PaiNN",
    "transformer": "Transformer",
    "pairformer": "Pairformer",
}

_D_NAMES = ["D1", "D2", "D3", "D4"]
_D_COLORS = ["#bdd7e7", "#6baed6", "#2171b5", "#084594"]   # light → dark blue
_SIZE_CMAP = "viridis"
_ARCHS_ORDERED = ["painn", "transformer", "pairformer"]


def _d_label(d_name: str, d_seen: float | None = None) -> str:
    if d_seen is not None:
        if d_seen >= 1e6:
            return f"{d_name} ({d_seen/1e6:.1f}M mol)"
        elif d_seen >= 1e3:
            return f"{d_name} ({d_seen/1e3:.0f}K mol)"
        return f"{d_name} ({int(d_seen)} mol)"
    return d_name


_TASK_LABELS = {
    "sphere_N50":      "Sphere N=50 η=0.3",
    "chain_N50":       "Chain N=50",
    "sphere_easy":     "Sphere η=0.1",
    "sphere_medium":   "Sphere η=0.3",
    "sphere_hard":     "Sphere η=0.5",
    "chain_N10":       "Chain N=10",
    "chain_N20":       "Chain N=20",
    "vsepr_sp3":       "VSEPR sp3",
    "sequence_linear": "Seq. linear",
}

_TASK_MARKERS = {
    "sphere_N50": "*", "chain_N50": "h",
    "sphere_easy": "o", "sphere_medium": "s", "sphere_hard": "D",
    "chain_N10": "^", "chain_N20": "v",
    "vsepr_sp3": "P", "sequence_linear": "X",
}

_TASK_COLORS = [
    "#006d2c", "#31a354",
    "#74c476", "#bae4b3", "#edf8e9",
    "#54278f", "#756bb1", "#bcbddc",
    "#e08214", "#d73027",
]
_TASK_COLOR_MAP = {
    tid: _TASK_COLORS[i]
    for i, tid in enumerate([
        "sphere_N50", "chain_N50",
        "sphere_easy", "sphere_medium", "sphere_hard",
        "chain_N10", "chain_N20",
        "vsepr_sp3", "sequence_linear",
    ])
}


def _arch_display(arch: str) -> str:
    return _ARCH_DISPLAY.get(arch, arch)


def _clip_vr(vr: float | np.ndarray) -> float | np.ndarray:
    """Clip VR to floor for consistent log-space handling."""
    return np.clip(vr, _VR_FLOOR, None)


def _extract_terminal_points(best_by_size_d: dict[str, dict]) -> list[dict]:
    """Extract flat list of terminal (arch, N, D_seen, C, VR, d_name, size)."""
    points = []
    for key, traj in best_by_size_d.items():
        arch = traj["arch"]
        pt = traj.get("terminal", {})
        N = traj.get("n_params", 0)
        fps = traj.get("flops_per_step", 0)
        step = pt.get("step", 0)
        vr = pt.get("violation_rate")
        D_seen = pt.get("D_seen", 0)
        d_name = traj.get("d_name", key.split("/")[-1])
        size = traj.get("size", "")
        if N <= 0 or fps <= 0 or step <= 0 or vr is None:
            continue
        C = fps * step
        points.append(dict(arch=arch, N=N, D_seen=D_seen, C=C,
                           VR=max(vr, _VR_FLOOR), d_name=d_name, size=size))
    return points


# ── Approach 1 plots ──────────────────────────────────────────────────────

def plot_isoflop_curves(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> dict[str, plt.Figure]:
    """N vs violation_rate at each D-budget level (one curve per D level)."""
    task_label = _TASK_LABELS.get(task_id, task_id)

    data: dict[str, dict[str, list]] = {}
    for cell_key, traj in best_by_size_d.items():
        arch = traj["arch"]
        d_name = traj.get("d_name") or cell_key.split("/")[-1]
        pt = traj.get("terminal", {})
        N = traj.get("n_params", 0)
        vr = pt.get("violation_rate")
        D_seen = pt.get("D_seen")
        if N <= 0 or vr is None:
            continue
        if arch not in data:
            data[arch] = {}
        data[arch].setdefault(d_name, []).append((N, max(vr, _VR_FLOOR), D_seen or 0))

    figs: dict[str, plt.Figure] = {}
    for arch, d_data in data.items():
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.set_title(f"{_arch_display(arch)} — {task_label}", fontsize=11)

        has_data = False
        d_names_sorted = sorted(d_data.keys())
        colors = _D_COLORS[:len(d_names_sorted)]
        while len(colors) < len(d_names_sorted):
            colors.append("#000000")

        for dname, color in zip(d_names_sorted, colors):
            pts = sorted(d_data[dname])
            if len(pts) < 2:
                continue
            Ns_d = np.array([p[0] for p in pts], dtype=float)
            VRs_d = np.array([p[1] for p in pts], dtype=float)
            d_seen_vals = [p[2] for p in pts if p[2] > 0]
            d_seen_med = float(np.median(d_seen_vals)) if d_seen_vals else None
            label = _d_label(dname, d_seen_med)

            ax.plot(Ns_d, VRs_d, "o-", color=color, label=label, linewidth=1.5,
                    markersize=5, zorder=3)
            best_idx = int(np.argmin(VRs_d))
            ax.axvline(Ns_d[best_idx], color=color, linestyle="--", linewidth=0.8,
                       alpha=0.6)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model parameters N")
        ax.set_ylabel("Violation rate")
        ax.legend(frameon=False, fontsize=8)
        figs[arch] = fig

    return figs


def plot_training_trajectories(
    trajectories: list[dict],
    task_id: str,
) -> dict[str, plt.Figure]:
    """Violation rate vs. D_seen during training, colored by model size."""
    task_label = _TASK_LABELS.get(task_id, task_id)

    best: dict[tuple, dict] = {}
    for traj in trajectories:
        arch, size = traj["arch"], traj["size"]
        if not traj["points"]:
            continue
        final_vr = traj["points"][-1]["violation_rate"]
        key = (arch, size)
        if key not in best or final_vr < best[key]["final_vr"]:
            best[key] = {**traj, "final_vr": final_vr}

    by_arch: dict[str, list[dict]] = {}
    for (arch, size), traj in best.items():
        by_arch.setdefault(arch, []).append(traj)

    figs: dict[str, plt.Figure] = {}
    for arch, trajs in by_arch.items():
        trajs = sorted(trajs, key=lambda t: t["n_params"])
        n = len(trajs)
        cmap = cm.get_cmap(_SIZE_CMAP, n)

        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.set_title(f"{_arch_display(arch)} — {task_label}", fontsize=11)

        for i, traj in enumerate(trajs):
            if not traj["points"]:
                continue
            D_seen = [pt["D_seen"] for pt in traj["points"]]
            VRs = [max(pt["violation_rate"], _VR_FLOOR) for pt in traj["points"]]
            color = cmap(i / max(n - 1, 1))
            ax.plot(D_seen, VRs, "-", color=color, linewidth=1.2, alpha=0.85,
                    label=f"N={traj['n_params']:,}" if i % 3 == 0 else None)

        terminal_Ds = sorted({t["points"][-1]["D_seen"] for t in trajs
                               if t["points"] and t["points"][-1].get("D_seen", 0) > 0})
        for D_val in terminal_Ds:
            ax.axvline(D_val, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Model size (small→large)", fontsize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training samples seen (D)")
        ax.set_ylabel("Violation rate")

        figs[arch] = fig

    return figs


def plot_arch_comparison(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """All 3 architectures on same axes, one subplot per D-budget level."""
    task_label = _TASK_LABELS.get(task_id, task_id)

    data: dict[str, dict[str, list]] = {}
    d_seen_by_name: dict[str, list] = {}
    for cell_key, traj in best_by_size_d.items():
        arch = traj["arch"]
        d_name = traj.get("d_name") or cell_key.split("/")[-1]
        pt = traj.get("terminal", {})
        N = traj.get("n_params", 0)
        vr = pt.get("violation_rate")
        D_seen = pt.get("D_seen", 0)
        if N <= 0 or vr is None:
            continue
        data.setdefault(d_name, {}).setdefault(arch, []).append((N, max(vr, _VR_FLOOR)))
        d_seen_by_name.setdefault(d_name, []).append(D_seen)

    d_names_sorted = sorted(data.keys())
    n_panels = len(d_names_sorted)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL[0], DOUBLE_COL[1] * 0.6 * nrows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    fig.suptitle(task_label, fontsize=12)

    for ax, dname in zip(axes, d_names_sorted):
        d_seen_vals = d_seen_by_name.get(dname, [])
        d_seen_med = float(np.median(d_seen_vals)) if d_seen_vals else None
        ax.set_title(_d_label(dname, d_seen_med), fontsize=9)
        for arch, pts in data[dname].items():
            pts_sorted = sorted(pts)
            if len(pts_sorted) < 2:
                continue
            Ns_d = np.array([p[0] for p in pts_sorted], dtype=float)
            VRs_d = np.array([p[1] for p in pts_sorted], dtype=float)
            disp = _arch_display(arch)
            ax.plot(Ns_d, VRs_d,
                    marker=ARCH_MARKERS.get(disp, "o"),
                    color=ARCH_COLORS.get(disp, "gray"),
                    label=disp, linewidth=1.5, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (params)", fontsize=8)
        ax.set_ylabel("Violation rate", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_isoflop_envelope(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """Compute-performance frontier: total FLOPs C vs best VR per arch."""
    task_label = _TASK_LABELS.get(task_id, task_id)

    arch_points: dict[str, list[tuple[float, float, float, float]]] = {}
    for cell_key, traj in best_by_size_d.items():
        arch = traj["arch"]
        pt = traj.get("terminal", {})
        fps = traj.get("flops_per_step", 0)
        step = pt.get("step", 0)
        vr = pt.get("violation_rate")
        n_p = traj.get("n_params", 0)
        D_seen = pt.get("D_seen", 0)
        if fps <= 0 or step <= 0 or vr is None:
            continue
        C = fps * step
        arch_points.setdefault(arch, []).append((C, n_p, D_seen, max(vr, _VR_FLOOR)))

    if not arch_points:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_title(f"Compute-Performance Frontier — {task_label}", fontsize=11)

    for arch, pts in arch_points.items():
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")
        marker = ARCH_MARKERS.get(disp, "o")

        pts_sorted = sorted(pts, key=lambda x: x[0])
        Cs = np.array([p[0] for p in pts_sorted], dtype=float)
        VRs = np.array([p[3] for p in pts_sorted], dtype=float)

        ax.scatter(Cs, VRs, color=color, alpha=0.35, s=20, zorder=2)

        env_C, env_VR = [], []
        running_min = float("inf")
        for C, VR in zip(Cs, VRs):
            if VR < running_min:
                running_min = VR
                env_C.append(C)
                env_VR.append(VR)

        if env_C:
            ax.plot(env_C, env_VR, "-", color=color, label=disp,
                    linewidth=2.0, marker=marker, markersize=5, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total training FLOPs C")
    ax.set_ylabel("Violation rate (lower = better)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_smooth_envelope(
    best_by_size_d: dict[str, dict],
    task_id: str,
    d_unique_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Chinchilla Approach 1: sigmoid envelope fit in log-compute space."""
    from scipy.optimize import curve_fit

    task_label = _TASK_LABELS.get(task_id, task_id)
    points = _extract_terminal_points(best_by_size_d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Compute–Performance Frontier — {task_label}", fontsize=12)

    for arch in _ARCHS_ORDERED:
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")
        marker = ARCH_MARKERS.get(disp, "o")

        pts = [p for p in points if p["arch"] == arch]
        if not pts:
            continue

        Cs = np.array([p["C"] for p in pts])
        VRs = np.array([p["VR"] for p in pts])

        ax.scatter(Cs, VRs, color=color, alpha=0.2, s=15, zorder=2)

        # Progressive-min envelope
        order = np.argsort(Cs)
        Cs_s, VRs_s = Cs[order], VRs[order]
        env_C, env_VR = [], []
        running_min = float("inf")
        for c, v in zip(Cs_s, VRs_s):
            if v < running_min:
                running_min = v
                env_C.append(c)
                env_VR.append(v)
        env_C = np.array(env_C)
        env_VR = np.array(env_VR)

        ax.scatter(env_C, env_VR, color=color, s=45, zorder=5, marker=marker,
                   edgecolors="white", linewidths=0.5)

        label = disp
        if len(env_C) >= 3:
            log_C_env = np.log10(env_C)
            try:
                def _sigmoid_vr(logC, a, b, E):
                    return E + (1.0 - E) / (1.0 + np.exp(b * (logC - a)))

                vr_min = float(env_VR.min())
                logC_mid = float(np.median(log_C_env))
                p0 = [logC_mid, 2.0, max(vr_min * 0.5, _VR_FLOOR)]
                bounds = (
                    [log_C_env.min() - 2, 0.1, 0],
                    [log_C_env.max() + 2, 20.0, min(vr_min + 0.1, 0.99)],
                )
                popt, _ = curve_fit(
                    _sigmoid_vr, log_C_env, env_VR,
                    p0=p0, bounds=bounds, maxfev=50000,
                )
                a_fit, b_fit, E_fit = popt

                C_smooth = np.geomspace(env_C.min() * 0.5, env_C.max() * 3, 400)
                VR_smooth = _sigmoid_vr(np.log10(C_smooth), a_fit, b_fit, E_fit)
                ax.plot(C_smooth, VR_smooth, "-", color=color, linewidth=2.5,
                        zorder=4, alpha=0.9)

                converged = env_VR < 0.95
                if converged.sum() >= 2:
                    coeffs = np.polyfit(
                        np.log10(env_C[converged]),
                        np.log10(_clip_vr(env_VR[converged])),
                        1,
                    )
                    alpha_slope = -coeffs[0]
                    label = f"{disp}  α={alpha_slope:.3f}"
                else:
                    label = f"{disp}  (floor={E_fit:.2f})"
            except Exception:
                ax.plot(env_C, env_VR, "-", color=color, linewidth=2.0,
                        zorder=4, alpha=0.9)
        elif len(env_C) >= 2:
            ax.plot(env_C, env_VR, "-", color=color, linewidth=2.0,
                    zorder=4, alpha=0.9)

        ax.plot([], [], "-", color=color, marker=marker, linewidth=2.0,
                label=label, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total training FLOPs C", fontsize=12)
    ax.set_ylabel("Violation rate (lower = better)", fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    fig.tight_layout()

    return fig


def plot_optimal_ND_from_envelope(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """N*(C) and D*(C) from compute-frontier envelope (Approach 1).

    Derives optimal allocation directly from the progressive-min envelope
    over all (C, N, D, VR) points — the same envelope shown in
    ``a1_compute_frontier``.  Each envelope point represents the best VR
    achieved at or below that compute budget, so its (N, D) is the
    compute-optimal allocation at that C.

    Power-law fits N*(C) ∝ C^a and D*(C) ∝ C^b are fitted on the
    converged (VR < 0.99) envelope points.
    """
    task_label = _TASK_LABELS.get(task_id, task_id)
    points = _extract_terminal_points(best_by_size_d)

    fig, (ax_N, ax_D) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Compute-Optimal Allocation (frontier) — {task_label}", fontsize=13)

    for arch in _ARCHS_ORDERED:
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")
        marker = ARCH_MARKERS.get(disp, "o")

        pts = [p for p in points if p["arch"] == arch]
        if not pts:
            continue

        # Sort all points by C and build progressive-min envelope
        # (same method as a1_compute_frontier)
        pts_sorted = sorted(pts, key=lambda p: p["C"])
        env_C, env_N, env_D, env_VR = [], [], [], []
        running_min = float("inf")
        for p in pts_sorted:
            if p["VR"] < running_min:
                running_min = p["VR"]
                env_C.append(p["C"])
                env_N.append(p["N"])
                env_D.append(p["D_seen"])
                env_VR.append(p["VR"])

        if not env_C:
            continue

        env_C = np.array(env_C)
        env_N = np.array(env_N)
        env_D = np.array(env_D)
        env_VR = np.array(env_VR)

        # Scatter: envelope-optimal points
        converged = env_VR < 0.99
        unconverged = ~converged
        if converged.any():
            ax_N.scatter(env_C[converged], env_N[converged], color=color, s=80,
                         zorder=5, marker=marker, edgecolors="white", linewidths=0.5)
            ax_D.scatter(env_C[converged], env_D[converged], color=color, s=80,
                         zorder=5, marker=marker, edgecolors="white", linewidths=0.5)
        if unconverged.any():
            ax_N.scatter(env_C[unconverged], env_N[unconverged], facecolors="none",
                         edgecolors=color, s=50, zorder=3, marker=marker,
                         linewidths=1.0, alpha=0.4)
            ax_D.scatter(env_C[unconverged], env_D[unconverged], facecolors="none",
                         edgecolors=color, s=50, zorder=3, marker=marker,
                         linewidths=1.0, alpha=0.4)

        # Connect envelope points
        ax_N.plot(env_C, env_N, "-", color=color, linewidth=1.0,
                  alpha=0.5, zorder=4)
        ax_D.plot(env_C, env_D, "-", color=color, linewidth=1.0,
                  alpha=0.5, zorder=4)

        # Fit power law on converged envelope points
        if converged.sum() >= 2:
            log_C = np.log10(env_C[converged])
            C_line = np.geomspace(env_C[converged].min() * 0.3,
                                  env_C[converged].max() * 3, 200)

            coeffs_N = np.polyfit(log_C, np.log10(env_N[converged]), 1)
            n_exp = coeffs_N[0]
            N_pred = 10 ** np.polyval(coeffs_N, log_C)
            ss_res = np.sum((np.log10(env_N[converged]) - np.log10(N_pred)) ** 2)
            ss_tot = np.sum((np.log10(env_N[converged]) - np.log10(env_N[converged]).mean()) ** 2)
            r2_N = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax_N.plot(C_line, 10 ** np.polyval(coeffs_N, np.log10(C_line)),
                      "--", color=color, linewidth=1.8,
                      label=f"{disp}  a={n_exp:.2f}  R²={r2_N:.2f}")

            coeffs_D = np.polyfit(log_C, np.log10(env_D[converged]), 1)
            d_exp = coeffs_D[0]
            D_pred = 10 ** np.polyval(coeffs_D, log_C)
            ss_res = np.sum((np.log10(env_D[converged]) - np.log10(D_pred)) ** 2)
            ss_tot = np.sum((np.log10(env_D[converged]) - np.log10(env_D[converged]).mean()) ** 2)
            r2_D = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax_D.plot(C_line, 10 ** np.polyval(coeffs_D, np.log10(C_line)),
                      "--", color=color, linewidth=1.8,
                      label=f"{disp}  b={d_exp:.2f}  R²={r2_D:.2f}")
        else:
            ax_N.plot([], [], "--", color=color, linewidth=1.8,
                      label=f"{disp}  (insufficient converged pts)")

    ax_N.set_title("N*(C) — frontier optimal params", fontsize=10)
    ax_D.set_title("D*(C) — frontier optimal data", fontsize=10)
    for ax, ylabel in [(ax_N, "Optimal N* (params)"),
                        (ax_D, "Optimal D* (samples seen)")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute budget C (FLOPs)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)

    fig.tight_layout()
    return fig


def plot_vr_vs_flops_by_data(
    best_by_size_d: dict[str, dict],
    task_id: str,
    d_unique_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Per-arch subplots: VR vs total FLOPs, one line per data budget."""
    task_label = _TASK_LABELS.get(task_id, task_id)
    points = _extract_terminal_points(best_by_size_d)
    d_unique_labels = d_unique_labels or {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Violation Rate vs. FLOPs by Data Budget — {task_label}", fontsize=13)

    for ax, arch in zip(axes, _ARCHS_ORDERED):
        disp = _arch_display(arch)
        ax.set_title(disp, fontsize=12, fontweight="bold")

        pts = [p for p in points if p["arch"] == arch]
        if not pts:
            continue

        # Group by d_name
        by_d: dict[str, list] = {}
        for p in pts:
            by_d.setdefault(p["d_name"], []).append(p)

        d_names_sorted = sorted(by_d.keys())
        colors = _D_COLORS[:len(d_names_sorted)]
        while len(colors) < len(d_names_sorted):
            colors.append("#000000")

        for dname, color in zip(d_names_sorted, colors):
            d_pts = sorted(by_d[dname], key=lambda p: p["C"])
            Cs = np.array([p["C"] for p in d_pts])
            VRs = np.array([p["VR"] for p in d_pts])

            # Label with data budget
            d_label = d_unique_labels.get(dname, dname)
            if d_label == dname:
                d_seen_vals = [p["D_seen"] for p in d_pts if p["D_seen"] > 0]
                if d_seen_vals:
                    d_med = float(np.median(d_seen_vals))
                    d_label = _d_label(dname, d_med)

            ax.plot(Cs, VRs, "o-", color=color, linewidth=1.8, markersize=5,
                    label=d_label, zorder=3)

            # Annotate model sizes
            for c, vr, p in zip(Cs, VRs, d_pts):
                size_str = p["size"].replace("chinchilla_", "c")
                ax.annotate(size_str, (c, vr), fontsize=5.5, color=color,
                            xytext=(2, 4), textcoords="offset points", alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total FLOPs C", fontsize=10)
        ax.set_ylabel("Violation rate", fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.grid(True, which="minor", alpha=0.1)

    return fig


def plot_vr_vs_flops_by_params(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """Per-arch subplots: VR vs total FLOPs, one line per model size."""
    task_label = _TASK_LABELS.get(task_id, task_id)
    points = _extract_terminal_points(best_by_size_d)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Violation Rate vs. FLOPs by Model Size — {task_label}", fontsize=13)

    _size_cmap = cm.get_cmap("viridis")

    for ax, arch in zip(axes, _ARCHS_ORDERED):
        disp = _arch_display(arch)
        ax.set_title(disp, fontsize=12, fontweight="bold")

        pts = [p for p in points if p["arch"] == arch]
        if not pts:
            continue

        by_size: dict[str, list] = {}
        for p in pts:
            by_size.setdefault(p["size"], []).append(p)

        sizes_sorted = sorted(by_size.keys(),
                               key=lambda s: int(s.split("_")[1]) if "_" in s else 0)
        n_sizes = len(sizes_sorted)

        for i, size in enumerate(sizes_sorted):
            s_pts = sorted(by_size[size], key=lambda p: p["C"])
            Cs = np.array([p["C"] for p in s_pts])
            VRs = np.array([p["VR"] for p in s_pts])
            N = s_pts[0]["N"]

            color = _size_cmap(i / max(n_sizes - 1, 1))

            if N >= 1e6:
                n_str = f"{N/1e6:.1f}M"
            elif N >= 1e3:
                n_str = f"{N/1e3:.0f}K"
            else:
                n_str = str(N)
            lbl = f"{size.replace('chinchilla_', 'c')}: {n_str}"

            ax.plot(Cs, VRs, "o-", color=color, linewidth=1.8, markersize=5,
                    label=lbl, zorder=3)

            for c, vr, p in zip(Cs, VRs, s_pts):
                ax.annotate(p["d_name"], (c, vr), fontsize=5.5, color=color,
                            xytext=(2, 4), textcoords="offset points", alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total FLOPs C", fontsize=10)
        ax.set_ylabel("Violation rate", fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.grid(True, which="minor", alpha=0.1)

    return fig


def plot_N_D_regime_map(
    best_by_size_d: dict[str, dict],
    task_id: str,
    fits: dict[str, dict] | None = None,
) -> dict[str, plt.Figure]:
    """N-D scatter colored by VR with regime boundary and optional L(N,D) contours."""
    task_label = _TASK_LABELS.get(task_id, task_id)

    arch_data: dict[str, list[tuple[float, float, float]]] = {}
    for key, traj in best_by_size_d.items():
        arch = traj["arch"]
        N = traj.get("n_params", 0)
        pt = traj.get("terminal", {})
        D = pt.get("D_seen", 0)
        vr = pt.get("violation_rate")
        if N <= 0 or D <= 0 or vr is None:
            continue
        arch_data.setdefault(arch, []).append((N, D, max(vr, _VR_FLOOR)))

    figs: dict[str, plt.Figure] = {}
    for arch, pts in arch_data.items():
        Ns = np.array([p[0] for p in pts])
        Ds = np.array([p[1] for p in pts])
        VRs = np.array([p[2] for p in pts])

        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.set_title(f"{_arch_display(arch)} — {task_label}\nN-D Regime Map", fontsize=10)

        vmin = max(VRs.min(), _VR_FLOOR)
        vmax = VRs.max()
        sc = ax.scatter(Ns, Ds, c=VRs, cmap="RdYlGn_r",
                        norm=LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin * 2 else None,
                        s=50, edgecolors="white", linewidths=0.3, zorder=4)
        fig.colorbar(sc, ax=ax, label="Violation rate", shrink=0.85)

        # Chinchilla boundary: D = 6N (below → data-limited)
        N_line = np.geomspace(Ns.min() * 0.5, Ns.max() * 2, 100)
        ax.plot(N_line, 6 * N_line, "k--", linewidth=0.8, alpha=0.5, label="D = 6N")

        # Optional contours from Approach 3 fit
        if fits and arch in fits:
            fp = fits[arch]
            E, A, alpha, B, beta = fp["E"], fp["A"], fp["alpha"], fp["B"], fp["beta"]
            N_grid = np.geomspace(Ns.min() * 0.5, Ns.max() * 2, 60)
            D_grid = np.geomspace(Ds.min() * 0.5, Ds.max() * 2, 60)
            NN, DD = np.meshgrid(N_grid, D_grid)
            LL = E + A * np.power(NN, -alpha) + B * np.power(DD, -beta)
            LL = np.clip(LL, _VR_FLOOR, 1.0)
            ax.contour(NN, DD, LL, levels=6, colors="gray", linewidths=0.6, alpha=0.4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model parameters N")
        ax.set_ylabel("Training molecules D")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        try:
            fig.tight_layout()
        except RuntimeError:
            pass
        figs[arch] = fig

    return figs


# ── Approach 3 plots ──────────────────────────────────────────────────────

def plot_loss_surface(
    best_by_size: dict[str, dict],
    fit_params: dict,
    task_id: str,
    arch: str,
) -> plt.Figure:
    """2D contour of fitted L(N,D) with scatter overlay."""
    task_label = _TASK_LABELS.get(task_id, task_id)
    E = fit_params["E"]
    A = fit_params["A"]
    alpha = fit_params["alpha"]
    B = fit_params["B"]
    beta = fit_params["beta"]
    r2 = fit_params.get("r_squared", float("nan"))

    scatter_N, scatter_D, scatter_L = [], [], []
    for key, traj in best_by_size.items():
        if traj["arch"] != arch:
            continue
        N = traj["n_params"]
        for pt in traj["points"]:
            scatter_N.append(N)
            scatter_D.append(pt["D_seen"])
            scatter_L.append(max(pt["violation_rate"], _VR_FLOOR))

    if not scatter_N:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    scatter_N = np.array(scatter_N, dtype=float)
    scatter_D = np.array(scatter_D, dtype=float)
    scatter_L = np.array(scatter_L, dtype=float)

    N_grid = np.geomspace(scatter_N.min() * 0.8, scatter_N.max() * 1.2, 80)
    D_grid = np.geomspace(scatter_D.min() * 0.8, scatter_D.max() * 1.2, 80)
    NN, DD = np.meshgrid(N_grid, D_grid)
    LL = E + A * np.power(NN, -alpha) + B * np.power(DD, -beta)
    LL = np.clip(LL, _VR_FLOOR, 1.0)

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    cf = ax.contourf(NN, DD, LL, levels=20, cmap="RdYlGn_r", alpha=0.85)
    fig.colorbar(cf, ax=ax, label="L(N,D) violation rate", shrink=0.85)

    sc = ax.scatter(scatter_N, scatter_D, c=scatter_L, cmap="RdYlGn_r",
                    norm=plt.Normalize(scatter_L.min(), scatter_L.max()),
                    s=30, edgecolors="white", linewidths=0.4, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters N")
    ax.set_ylabel("Training samples D")
    ax.set_title(
        f"{_arch_display(arch)} — {task_label}\n"
        f"L=E+A/N^α+B/D^β  α={alpha:.2f} β={beta:.2f}  R²={r2:.2f}",
        fontsize=9,
    )
    return fig


def plot_scaling_exponent_heatmap(
    all_fits: dict[str, dict],
    exponent_key: str = "alpha",
) -> plt.Figure:
    """Heatmap of scaling exponents: rows=architectures, cols=tasks."""
    from experiments.task_registry import TASKS_BY_COMPLEXITY

    tasks_ordered = [t.task_id for t in TASKS_BY_COMPLEXITY
                     if t.task_id in all_fits]
    archs = ["painn", "transformer", "pairformer"]
    arch_labels = [_arch_display(a) for a in archs]
    task_labels = [_TASK_LABELS.get(t, t) for t in tasks_ordered]

    matrix = np.full((len(archs), len(tasks_ordered)), np.nan)
    for j, task_id in enumerate(tasks_ordered):
        for i, arch in enumerate(archs):
            v = all_fits.get(task_id, {}).get(arch, {}).get(exponent_key)
            if v is not None:
                matrix[i, j] = float(v)

    key_labels = {
        "alpha": "α (model scaling exponent)",
        "beta":  "β (data scaling exponent)",
        "N_exponent": "β/(α+β)  [N* ∝ C^this]",
        "D_exponent": "α/(α+β)  [D* ∝ C^this]",
    }
    cbar_label = key_labels.get(exponent_key, exponent_key)

    fig, ax = plt.subplots(figsize=(max(DOUBLE_COL[0], len(tasks_ordered) * 1.2), 3.5))
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

    for i in range(len(archs)):
        for j in range(len(tasks_ordered)):
            val = matrix[i, j]
            if np.isfinite(val):
                rgba = im.cmap(im.norm(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = "white" if brightness < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=tc, fontsize=9)
            else:
                ax.text(j, i, "n/a", ha="center", va="center",
                        color="gray", fontsize=8)

    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(arch_labels)
    ax.set_xticks(range(len(tasks_ordered)))
    ax.set_xticklabels(task_labels, rotation=40, ha="right", fontsize=9)
    ax.set_title(f"Scaling exponent: {cbar_label}")

    return fig


def plot_optimal_allocation(
    fits: dict[str, dict],
    task_id: str,
    compute_budgets: np.ndarray,
) -> plt.Figure:
    """N*(C) and D*(C) curves per architecture (Approach 3)."""
    task_label = _TASK_LABELS.get(task_id, task_id)
    fig, (ax_N, ax_D) = plt.subplots(1, 2, figsize=DOUBLE_COL)
    fig.suptitle(f"{task_label} — Compute-optimal allocation (Approach 3)", fontsize=11)

    for arch, fparams in fits.items():
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")

        alpha = fparams["alpha"]
        beta  = fparams["beta"]
        A = fparams["A"]
        B = fparams["B"]
        r2 = fparams.get("r_squared", fparams.get("r_squared_raw", float("nan")))
        N_exp = fparams.get("N_exponent", beta / (alpha + beta))
        D_exp = fparams.get("D_exponent", alpha / (alpha + beta))

        K = 6.0
        ratio = (A * alpha) / (B * beta)
        N_star = (ratio ** (beta / (alpha + beta))) * ((compute_budgets / K) ** N_exp)
        D_star = compute_budgets / (K * np.maximum(N_star, 1))

        ax_N.plot(compute_budgets, N_star, color=color,
                  label=f"{disp}  a={N_exp:.2f}  α={alpha:.2f}  R²={r2:.2f}",
                  linewidth=1.8)
        ax_D.plot(compute_budgets, D_star, color=color,
                  label=f"{disp}  b={D_exp:.2f}  β={beta:.2f}  R²={r2:.2f}",
                  linewidth=1.8)

    ax_N.set_title("N*(C) ∝ C^a,  a = β/(α+β)", fontsize=10)
    ax_D.set_title("D*(C) ∝ C^b,  b = α/(α+β)", fontsize=10)
    for ax, ylabel in [(ax_N, "Optimal N* (params)"), (ax_D, "Optimal D* (samples)")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute budget C (FLOPs)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def plot_optimal_allocation_approach1(
    fits: dict[str, dict],
    task_id: str,
    compute_budgets: np.ndarray,
) -> plt.Figure:
    """Approach 1: Empirical N*(C) and D*(C) from isoFLOP envelope.

    Shows scatter of observed envelope points alongside the fitted
    power-law lines N*(C) = a_N × C^n_exp, D*(C) = a_D × C^d_exp.
    The fit line is drawn spanning the actual data range (not hardcoded).
    """
    task_label = _TASK_LABELS.get(task_id, task_id)
    fig, (ax_N, ax_D) = plt.subplots(1, 2, figsize=DOUBLE_COL)
    fig.suptitle(f"{task_label} — Compute-optimal allocation (Approach 1)", fontsize=11)

    for arch, fparams in fits.items():
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")
        marker = ARCH_MARKERS.get(disp, "o")

        n_exp = fparams["n_exp"]
        a_N   = fparams["a_N"]
        d_exp = fparams["d_exp"]
        a_D   = fparams["a_D"]
        r2_N  = fparams.get("r2_N", float("nan"))
        r2_D  = fparams.get("r2_D", float("nan"))

        env_C  = np.array(fparams["envelope_C"])
        env_N  = np.array(fparams["envelope_N"])
        env_D  = np.array(fparams["envelope_D"])

        # Scatter: empirical optimal points
        ax_N.scatter(env_C, env_N, color=color, s=50, zorder=4, marker=marker,
                     edgecolors="white", linewidths=0.5)
        ax_D.scatter(env_C, env_D, color=color, s=50, zorder=4, marker=marker,
                     edgecolors="white", linewidths=0.5)

        # Fit line spanning actual data range (with margins)
        C_range = np.geomspace(env_C.min() * 0.3, env_C.max() * 3, 200)
        N_fit = a_N * np.power(C_range, n_exp)
        D_fit = a_D * np.power(C_range, d_exp)
        ax_N.plot(C_range, N_fit, color=color, linewidth=1.8, linestyle="--",
                  label=f"{disp}  a={n_exp:.2f}  R²={r2_N:.2f}")
        ax_D.plot(C_range, D_fit, color=color, linewidth=1.8, linestyle="--",
                  label=f"{disp}  b={d_exp:.2f}  R²={r2_D:.2f}")

    ax_N.set_title("N*(C) = a_N · C^a", fontsize=10)
    ax_D.set_title("D*(C) = a_D · C^b", fontsize=10)
    for ax, ylabel in [(ax_N, "Optimal N* (params)"), (ax_D, "Optimal D* (samples)")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute budget C (FLOPs)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def plot_cross_task_summary(
    all_fits: dict[str, dict],
) -> plt.Figure:
    """α vs β scatter plot for all (task, arch) combinations."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    plotted_tasks: set[str] = set()
    plotted_archs: set[str] = set()

    for task_id, arch_fits in all_fits.items():
        task_marker = _TASK_MARKERS.get(task_id, "o")
        task_color = _TASK_COLOR_MAP.get(task_id, "gray")

        for arch, fparams in arch_fits.items():
            alpha = fparams.get("alpha")
            beta  = fparams.get("beta")
            if alpha is None or beta is None:
                continue

            disp = _arch_display(arch)
            arch_color = ARCH_COLORS.get(disp, "gray")

            ax.scatter(
                alpha, beta,
                marker=task_marker,
                color=arch_color,
                edgecolors=task_color,
                linewidths=1.5,
                s=80,
                zorder=4,
            )
            plotted_tasks.add(task_id)
            plotted_archs.add(arch)

    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.5)
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.4,
            label="α = β")

    for arch in sorted(plotted_archs):
        disp = _arch_display(arch)
        ax.scatter([], [], color=ARCH_COLORS.get(disp, "gray"),
                   label=disp, s=60, marker="o")
    for tid in sorted(plotted_tasks):
        ax.scatter([], [], color="gray",
                   marker=_TASK_MARKERS.get(tid, "o"),
                   label=_TASK_LABELS.get(tid, tid), s=60)

    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.set_xlabel("α  (model scaling exponent)")
    ax.set_ylabel("β  (data scaling exponent)")
    ax.set_title("Scaling strategy by task & architecture")
    ax.text(0.95, 0.05, "← data-limited\nmodel-limited ↓",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="gray", style="italic")

    return fig


# ── Orchestrator ──────────────────────────────────────────────────────────

def _infer_d_unique_labels(chinchilla_dir: str, task_id: str) -> dict[str, str]:
    """Infer unique dataset size labels from run configs (overrides.yaml)."""
    import glob
    import yaml
    task_dir = os.path.join(chinchilla_dir, task_id)
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
        +-- a1_isoflop_{arch}.png
        +-- a1_compute_frontier.png
        +-- a1_compute_frontier_fit.png
        +-- a1_optimal_allocation.png
        +-- a1_vr_by_data.png
        +-- a1_vr_by_params.png
        +-- a1_nd_regime_{arch}.png
        +-- a1_arch_comparison.png
        +-- a1_optimal_allocation_fit.png
        +-- a3_loss_surface_{arch}.png
        +-- a3_optimal_allocation.png
        +-- training_trajectories_{arch}.png

    Cross-task (in plots_dir root):
        +-- heatmap_{key}.png
        +-- cross_task_summary.png
    """
    from viz.style import save_figure, synthbench_style

    chinchilla_dir = args.chinchilla_dir
    plots_base = args.plots_dir
    tasks = [t.strip() for t in args.tasks.split(",")]

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

            figs = plot_isoflop_curves(best_by_size, task_id)
            for arch, fig in figs.items():
                save_figure(fig, os.path.join(plots_dir, f"a1_isoflop_{arch}"))
                print(f"  a1_isoflop_{arch}  ({task_id})")

            if best_by_size:
                fig_env = plot_isoflop_envelope(best_by_size, task_id)
                save_figure(fig_env, os.path.join(plots_dir, "a1_compute_frontier"))
                print(f"  a1_compute_frontier  ({task_id})")

            if best_by_size:
                fig_se = plot_smooth_envelope(best_by_size, task_id)
                save_figure(fig_se, os.path.join(plots_dir, "a1_compute_frontier_fit"))
                print(f"  a1_compute_frontier_fit  ({task_id})")

            if best_by_size:
                fig_nd = plot_optimal_ND_from_envelope(best_by_size, task_id)
                save_figure(fig_nd, os.path.join(plots_dir, "a1_optimal_allocation"))
                print(f"  a1_optimal_allocation  ({task_id})")

            if best_by_size:
                _d_uniq = _infer_d_unique_labels(chinchilla_dir, task_id)
                fig_vrd = plot_vr_vs_flops_by_data(
                    best_by_size, task_id, d_unique_labels=_d_uniq)
                save_figure(fig_vrd, os.path.join(plots_dir, "a1_vr_by_data"))
                print(f"  a1_vr_by_data  ({task_id})")

            if best_by_size:
                fig_vrp = plot_vr_vs_flops_by_params(best_by_size, task_id)
                save_figure(fig_vrp, os.path.join(plots_dir, "a1_vr_by_params"))
                print(f"  a1_vr_by_params  ({task_id})")

            if best_by_size:
                figs_nd = plot_N_D_regime_map(best_by_size, task_id, fits=fits or None)
                for arch, fig_nd in figs_nd.items():
                    save_figure(fig_nd, os.path.join(plots_dir, f"a1_nd_regime_{arch}"))
                    print(f"  a1_nd_regime_{arch}  ({task_id})")

            if best_by_size:
                fig3 = plot_arch_comparison(best_by_size, task_id)
                save_figure(fig3, os.path.join(plots_dir, "a1_arch_comparison"))
                print(f"  a1_arch_comparison  ({task_id})")

            # A1: Optimal allocation from fit_approach1 (if available)
            if fits_a1:
                fits_a1_valid = {k: v for k, v in fits_a1.items()
                                 if v.get("fit_available", True) and "n_exp" in v}
                if fits_a1_valid:
                    # Use actual data range, not hardcoded
                    all_C = []
                    for v in fits_a1_valid.values():
                        all_C.extend(v.get("envelope_C", []))
                    if all_C:
                        c_min = min(all_C) * 0.3
                        c_max = max(all_C) * 3
                    else:
                        c_min, c_max = 1e12, 1e17
                    compute_budgets = np.geomspace(c_min, c_max, 200)
                    fig_a1 = plot_optimal_allocation_approach1(
                        fits_a1_valid, task_id, compute_budgets)
                    save_figure(fig_a1, os.path.join(plots_dir, "a1_optimal_allocation_fit"))
                    print(f"  a1_optimal_allocation_fit  ({task_id})")

            # ── Approach 3: Parametric L(N,D) fit ────────────────────────

            for arch, fparams in fits.items():
                fig_s = plot_loss_surface(best_by_size, fparams, task_id, arch)
                save_figure(fig_s, os.path.join(plots_dir, f"a3_loss_surface_{arch}"))
                print(f"  a3_loss_surface_{arch}  ({task_id})")

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


# ── Temperature (T) axis comparison for n-body tasks ─────────────────────

def _parse_T_from_task(task_id: str) -> float | None:
    """Extract temperature T from task_id like nbody_n15_b2_T1.0."""
    import re
    m = re.search(r"_T([\d.]+)$", task_id)
    return float(m.group(1)) if m else None


def _T_color(T: float, T_values: list[float]) -> str:
    """Return color for temperature value from a blue→red gradient."""
    if len(T_values) <= 1:
        return "#2171b5"
    idx = sorted(T_values).index(T)
    frac = idx / (len(T_values) - 1)
    cmap = cm.get_cmap("coolwarm")
    rgba = cmap(frac)
    return "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


def _T_marker(T: float, T_values: list[float]) -> str:
    markers = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "<"]
    idx = sorted(T_values).index(T)
    return markers[idx % len(markers)]


def plot_T_compute_frontier(
    task_results: dict[str, dict],
    arch: str,
) -> plt.Figure:
    """Compute-performance frontier for one arch across multiple T values.

    Args:
        task_results: {task_id: results.json content} for nbody tasks with different T.
        arch: architecture to plot.
    """
    T_data: dict[float, list[tuple[float, float]]] = {}  # T → [(C, VR)]
    for task_id, results in task_results.items():
        T = _parse_T_from_task(task_id)
        if T is None:
            continue
        best = results.get("best_by_size_d", {})
        for key, traj in best.items():
            if traj["arch"] != arch:
                continue
            pt = traj.get("terminal", {})
            fps = traj.get("flops_per_step", 0)
            step = pt.get("step", 0)
            vr = pt.get("violation_rate")
            if fps <= 0 or step <= 0 or vr is None:
                continue
            C = fps * step
            T_data.setdefault(T, []).append((C, max(vr, _VR_FLOOR)))

    T_values = sorted(T_data.keys())
    if not T_values:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_title(f"{_arch_display(arch)} — Compute Frontier by Temperature", fontsize=11)

    for T in T_values:
        pts = sorted(T_data[T], key=lambda x: x[0])
        Cs = np.array([p[0] for p in pts], dtype=float)
        VRs = np.array([p[1] for p in pts], dtype=float)
        color = _T_color(T, T_values)
        marker = _T_marker(T, T_values)

        ax.scatter(Cs, VRs, color=color, alpha=0.35, s=20, zorder=2)

        # Envelope (running min)
        env_C, env_VR = [], []
        running_min = float("inf")
        for c, v in zip(Cs, VRs):
            if v < running_min:
                running_min = v
                env_C.append(c)
                env_VR.append(v)
        if env_C:
            ax.plot(env_C, env_VR, "-", color=color, label=f"T={T}",
                    linewidth=2.0, marker=marker, markersize=5, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total FLOPs (C)")
    ax.set_ylabel("Violation rate (energy W2 / σ)")
    ax.legend(frameon=False, fontsize=8, title="Temperature")
    fig.tight_layout()
    return fig


def plot_T_arch_comparison(
    task_results: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """For one task (one T), show all archs — same as plot_arch_comparison
    but called per-task in T-sweep context. Delegates to existing function."""
    results = task_results.get(task_id, {})
    best = results.get("best_by_size_d", {})
    return plot_arch_comparison(best, task_id)


def plot_T_isoflop_by_temp(
    task_results: dict[str, dict],
    arch: str,
) -> plt.Figure:
    """N (params) vs VR, one subplot per T — like isoflop but T axis instead of D.

    Each subplot shows curves for different D budgets at one temperature.
    """
    T_data: dict[float, dict[str, list]] = {}  # T → {d_name: [(N, VR)]}
    for task_id, results in task_results.items():
        T = _parse_T_from_task(task_id)
        if T is None:
            continue
        best = results.get("best_by_size_d", {})
        for key, traj in best.items():
            if traj["arch"] != arch:
                continue
            d_name = traj.get("d_name", key.split("/")[-1])
            pt = traj.get("terminal", {})
            N = traj.get("n_params", 0)
            vr = pt.get("violation_rate")
            if N <= 0 or vr is None:
                continue
            T_data.setdefault(T, {}).setdefault(d_name, []).append(
                (N, max(vr, _VR_FLOOR)))

    T_values = sorted(T_data.keys())
    if not T_values:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    n_panels = len(T_values)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL[0] * 1.2, DOUBLE_COL[1] * 0.6 * nrows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    fig.suptitle(f"{_arch_display(arch)} — IsoFLOP by Temperature", fontsize=12)

    for ax, T in zip(axes, T_values):
        ax.set_title(f"T = {T}", fontsize=9)
        d_data = T_data[T]
        d_names_sorted = sorted(d_data.keys())
        for i, dname in enumerate(d_names_sorted):
            pts = sorted(d_data[dname])
            if len(pts) < 2:
                continue
            Ns = np.array([p[0] for p in pts], dtype=float)
            VRs = np.array([p[1] for p in pts], dtype=float)
            color = _D_COLORS[i % len(_D_COLORS)]
            ax.plot(Ns, VRs, marker="o", color=color, label=dname,
                    linewidth=1.5, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (params)", fontsize=8)
        ax.set_ylabel("Violation rate", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6)

    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_T_training_trajectories(
    task_results: dict[str, dict],
    arch: str,
    size: str | None = None,
) -> plt.Figure:
    """D_seen vs VR across temperatures for one arch (optionally one size).

    Each T gets a different color. If size is None, uses the middle size.
    """
    T_trajs: dict[float, list[dict]] = {}
    for task_id, results in task_results.items():
        T = _parse_T_from_task(task_id)
        if T is None:
            continue
        for traj in results.get("all_trajectories", []):
            if traj["arch"] != arch:
                continue
            if size and traj["size"] != size:
                continue
            T_trajs.setdefault(T, []).append(traj)

    T_values = sorted(T_trajs.keys())
    if not T_values:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    # If no specific size, pick the one that appears most
    if not size:
        from collections import Counter
        all_sizes = []
        for trajs in T_trajs.values():
            all_sizes.extend(t["size"] for t in trajs)
        if all_sizes:
            size = Counter(all_sizes).most_common(1)[0][0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    title = f"{_arch_display(arch)}"
    if size:
        title += f" ({size})"
    ax.set_title(f"{title} — Training by Temperature", fontsize=11)

    for T in T_values:
        color = _T_color(T, T_values)
        for traj in T_trajs[T]:
            if size and traj["size"] != size:
                continue
            points = traj.get("points", [])
            if not points:
                continue
            ds = [p.get("D_seen", p.get("step", 0) * 256) for p in points]
            vrs = [max(p.get("violation_rate", 1.0), _VR_FLOOR) for p in points]
            d_name = traj.get("d_name", "")
            ax.plot(ds, vrs, color=color, alpha=0.6, linewidth=1.0)

        # Legend proxy
        ax.plot([], [], color=color, label=f"T={T}", linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (samples seen)")
    ax.set_ylabel("Violation rate")
    ax.legend(frameon=False, fontsize=8, title="Temperature")
    fig.tight_layout()
    return fig


def plot_T(args: argparse.Namespace) -> None:
    """Generate T-axis comparison figures for n-body tasks.

    Expects --tasks to contain multiple nbody tasks with different T values,
    e.g. nbody_n15_b2_T0.5,nbody_n15_b2_T1.0,nbody_n15_b2_T2.0

    Output:
        {plots_dir}/T_sweep/
        +-- T_compute_frontier_{arch}.png   — envelope per arch, colored by T
        +-- T_isoflop_{arch}.png            — N vs VR subplots per T
        +-- T_training_{arch}.png           — D_seen vs VR colored by T
        +-- T_arch_comparison_{task}.png    — arch comparison per T value
    """
    from viz.style import save_figure, synthbench_style

    chinchilla_dir = args.chinchilla_dir
    plots_base = args.plots_dir
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Filter to nbody tasks with T
    nbody_tasks = [t for t in tasks if _parse_T_from_task(t) is not None]
    if not nbody_tasks:
        print("[WARN] No n-body T-sweep tasks found. "
              "Use --nbody_temps or pass nbody_n*_T* tasks.", file=sys.stderr)
        return

    # Load results for all T tasks
    task_results: dict[str, dict] = {}
    for task_id in nbody_tasks:
        res_path = _results_path(chinchilla_dir, task_id)
        if not os.path.exists(res_path):
            print(f"[SKIP] No results.json for '{task_id}'", file=sys.stderr)
            continue
        with open(res_path) as f:
            task_results[task_id] = json.load(f)

    if not task_results:
        print("[WARN] No results loaded for T-sweep plotting.", file=sys.stderr)
        return

    # Determine archs from data
    archs_seen = set()
    for results in task_results.values():
        for traj in results.get("all_trajectories", []):
            archs_seen.add(traj["arch"])
    archs_ordered = [a for a in _ARCHS_ORDERED if a in archs_seen]

    plots_dir = os.path.join(plots_base, "T_sweep")
    os.makedirs(plots_dir, exist_ok=True)

    with synthbench_style():
        # Per-arch: compute frontier colored by T
        for arch in archs_ordered:
            fig = plot_T_compute_frontier(task_results, arch)
            save_figure(fig, os.path.join(plots_dir, f"T_compute_frontier_{arch}"))
            print(f"  T_compute_frontier_{arch}")

        # Per-arch: isoflop subplots by T
        for arch in archs_ordered:
            fig = plot_T_isoflop_by_temp(task_results, arch)
            save_figure(fig, os.path.join(plots_dir, f"T_isoflop_{arch}"))
            print(f"  T_isoflop_{arch}")

        # Per-arch: training trajectories colored by T
        for arch in archs_ordered:
            fig = plot_T_training_trajectories(task_results, arch)
            save_figure(fig, os.path.join(plots_dir, f"T_training_{arch}"))
            print(f"  T_training_{arch}")

        # Per-task: arch comparison (same as standard but in T_sweep dir)
        for task_id in sorted(task_results.keys()):
            fig = plot_T_arch_comparison(task_results, task_id)
            save_figure(fig, os.path.join(plots_dir, f"T_arch_comparison_{task_id}"))
            print(f"  T_arch_comparison_{task_id}")
