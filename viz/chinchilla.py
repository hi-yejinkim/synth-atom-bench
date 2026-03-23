"""Visualization for Chinchilla-style scaling law study.

Approach 1 (isoFLOP profiles):
  plot_isoflop_curves        -- N vs violation_rate per D budget
  plot_training_trajectories -- D_seen vs violation_rate per model size
  plot_arch_comparison       -- all 3 archs on same axes per D budget
  plot_smooth_envelope       -- smooth compute-perf frontier (Chinchilla paper style)
  plot_optimal_ND_from_envelope -- N*(C) and D*(C) from envelope minima
  plot_vr_vs_flops_by_data   -- VR vs FLOPs per arch, colored by data budget

Approach 3 (parametric L(N,D) fit):
  plot_loss_surface          -- contour of L(N,D) with scatter overlay
  plot_scaling_exponent_heatmap -- α / β heatmap: archs × tasks
  plot_optimal_allocation    -- N*(C) and D*(C) curves per arch
  plot_cross_task_summary    -- α vs β scatter for all (task, arch)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from viz.style import ARCH_COLORS, ARCH_MARKERS, SINGLE_COL, DOUBLE_COL

# ── Constants ─────────────────────────────────────────────────────────────

_ARCH_DISPLAY = {
    "painn": "PaiNN",
    "transformer": "Transformer",
    "pairformer": "Pairformer",
}

# D-level names (D1–D4 are the 4 data budgets, assigned at generate time)
_D_NAMES = ["D1", "D2", "D3", "D4"]

_D_COLORS = ["#bdd7e7", "#6baed6", "#2171b5", "#084594"]   # light → dark blue
_SIZE_CMAP = "viridis"


def _d_label(d_name: str, d_seen: float | None = None) -> str:
    """Human-readable label for a D-budget level.

    If d_seen (molecule count) is provided, include it in the label.
    """
    if d_seen is not None:
        if d_seen >= 1e6:
            return f"{d_name} ({d_seen/1e6:.1f}M mol)"
        elif d_seen >= 1e3:
            return f"{d_name} ({d_seen/1e3:.0f}K mol)"
        return f"{d_name} ({int(d_seen)} mol)"
    return d_name

_TASK_LABELS = {
    # Large-N primary tasks (Chinchilla targets)
    "sphere_N50":      "Sphere N=50 η=0.3",
    "chain_N50":       "Chain N=50",
    # Small-N diagnostic tasks
    "sphere_easy":     "Sphere η=0.1",
    "sphere_medium":   "Sphere η=0.3",
    "sphere_hard":     "Sphere η=0.5",
    "chain_N10":       "Chain N=10",
    "chain_N20":       "Chain N=20",
    "vsepr_sp3":       "VSEPR sp3",
    "sequence_linear": "Seq. linear",
}

_TASK_MARKERS = {
    "sphere_N50":      "*",
    "chain_N50":       "h",
    "sphere_easy":     "o",
    "sphere_medium":   "s",
    "sphere_hard":     "D",
    "chain_N10":       "^",
    "chain_N20":       "v",
    "vsepr_sp3":       "P",
    "sequence_linear": "X",
}

_TASK_COLORS = [
    "#006d2c", "#31a354",              # large-N: dark green shades (primary)
    "#74c476", "#bae4b3", "#edf8e9",   # sphere N=10: light green shades
    "#54278f", "#756bb1", "#bcbddc",   # chain N≤20: purple shades
    "#e08214",                         # vsepr: orange
    "#d73027",                         # sequence: red
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



# ── Approach 1 plots ──────────────────────────────────────────────────────

def plot_isoflop_curves(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> dict[str, plt.Figure]:
    """N vs violation_rate at each D-budget level (one curve per D level).

    Uses terminal points from best_by_size_d (best LR already selected in
    collect). Each curve shows the tradeoff between model size N and VR at a
    fixed data budget — the curve minimum N* shifts right as D grows, tracing
    the compute-optimal allocation.

    Args:
        best_by_size_d: dict keyed by "arch/size/d_name" from results.json.
        task_id: task identifier for title.

    Returns:
        dict mapping arch name → Figure.
    """
    task_label = _TASK_LABELS.get(task_id, task_id)

    # Organize: arch → d_name → [(N_params, violation_rate, D_seen)]
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
        data[arch].setdefault(d_name, []).append((N, vr, D_seen or 0))

    figs: dict[str, plt.Figure] = {}
    for arch, d_data in data.items():
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.set_title(f"{_arch_display(arch)} — {task_label}", fontsize=11)

        has_data = False
        d_names_sorted = sorted(d_data.keys())  # D1 < D2 < D3 < D4
        colors = _D_COLORS[:len(d_names_sorted)]
        # Pad colors if more than 4 D levels
        while len(colors) < len(d_names_sorted):
            colors.append("#000000")

        for dname, color in zip(d_names_sorted, colors):
            pts = sorted(d_data[dname])  # sort by N
            if len(pts) < 2:
                continue
            Ns_d = np.array([p[0] for p in pts], dtype=float)
            VRs_d = np.array([p[1] for p in pts], dtype=float)
            # Infer D_seen label from median of available values
            d_seen_vals = [p[2] for p in pts if p[2] > 0]
            d_seen_med = float(np.median(d_seen_vals)) if d_seen_vals else None
            label = _d_label(dname, d_seen_med)

            ax.plot(Ns_d, VRs_d, "o-", color=color, label=label, linewidth=1.5,
                    markersize=5, zorder=3)
            # Mark optimal N* at this D level
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
    """Violation rate vs. D_seen during training, colored by model size.

    One figure per architecture. Color gradient: small models = light, large = dark.

    Args:
        trajectories: list of trajectory dicts.
        task_id: task identifier for title.

    Returns:
        dict mapping arch → Figure.
    """
    task_label = _TASK_LABELS.get(task_id, task_id)

    # Best LR per (arch, size)
    best: dict[tuple, dict] = {}
    for traj in trajectories:
        arch, size = traj["arch"], traj["size"]
        if not traj["points"]:
            continue
        final_vr = traj["points"][-1]["violation_rate"]
        key = (arch, size)
        if key not in best or final_vr < best[key]["final_vr"]:
            best[key] = {**traj, "final_vr": final_vr}

    # Group by arch
    by_arch: dict[str, list[dict]] = {}
    for (arch, size), traj in best.items():
        by_arch.setdefault(arch, []).append(traj)

    figs: dict[str, plt.Figure] = {}
    for arch, trajs in by_arch.items():
        # Sort by n_params for color gradient
        trajs = sorted(trajs, key=lambda t: t["n_params"])
        n = len(trajs)
        cmap = cm.get_cmap(_SIZE_CMAP, n)

        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.set_title(f"{_arch_display(arch)} — {task_label}", fontsize=11)

        for i, traj in enumerate(trajs):
            if not traj["points"]:
                continue
            D_seen = [pt["D_seen"] for pt in traj["points"]]
            VRs = [pt["violation_rate"] for pt in traj["points"]]
            color = cmap(i / max(n - 1, 1))
            ax.plot(D_seen, VRs, "-", color=color, linewidth=1.2, alpha=0.85,
                    label=f"N={traj['n_params']:,}" if i % 3 == 0 else None)

        # Draw vertical lines at the last eval step of each D-budget run
        terminal_Ds = sorted({t["points"][-1]["D_seen"] for t in trajs
                               if t["points"] and t["points"][-1].get("D_seen", 0) > 0})
        for D_val in terminal_Ds:
            ax.axvline(D_val, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

        # Colorbar for model size
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
    """All 3 architectures on same axes, one subplot per D-budget level.

    Uses terminal points from best_by_size_d (best LR already selected).

    Args:
        best_by_size_d: dict keyed "arch/size/d_name" from results.json.
        task_id: task identifier.

    Returns:
        Figure with up to 2×2 subplots (one per D budget level found in data).
    """
    task_label = _TASK_LABELS.get(task_id, task_id)

    # Organize: d_name → arch → [(N, VR)]
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
        data.setdefault(d_name, {}).setdefault(arch, []).append((N, vr))
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

    # Hide unused subplots
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# ── Approach 3 plots ──────────────────────────────────────────────────────

def plot_loss_surface(
    best_by_size: dict[str, dict],
    fit_params: dict,
    task_id: str,
    arch: str,
) -> plt.Figure:
    """2D contour of fitted L(N,D) with scatter overlay of actual measurements.

    Args:
        best_by_size: from results.json['best_by_size'].
        fit_params: dict with E, A, alpha, B, beta.
        task_id: for title.
        arch: architecture name.

    Returns:
        Figure.
    """
    task_label = _TASK_LABELS.get(task_id, task_id)
    E = fit_params["E"]
    A = fit_params["A"]
    alpha = fit_params["alpha"]
    B = fit_params["B"]
    beta = fit_params["beta"]
    r2 = fit_params.get("r_squared", float("nan"))

    # Gather scatter data
    scatter_N, scatter_D, scatter_L = [], [], []
    for key, traj in best_by_size.items():
        if traj["arch"] != arch:
            continue
        N = traj["n_params"]
        for pt in traj["points"]:
            scatter_N.append(N)
            scatter_D.append(pt["D_seen"])
            scatter_L.append(pt["violation_rate"])

    if not scatter_N:
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    scatter_N = np.array(scatter_N, dtype=float)
    scatter_D = np.array(scatter_D, dtype=float)
    scatter_L = np.array(scatter_L, dtype=float)

    # Grid for contour
    N_grid = np.geomspace(scatter_N.min() * 0.8, scatter_N.max() * 1.2, 80)
    D_grid = np.geomspace(scatter_D.min() * 0.8, scatter_D.max() * 1.2, 80)
    NN, DD = np.meshgrid(N_grid, D_grid)
    LL = E + A * np.power(NN, -alpha) + B * np.power(DD, -beta)
    LL = np.clip(LL, 1e-4, 1.0)

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    cf = ax.contourf(NN, DD, LL, levels=20, cmap="RdYlGn_r", alpha=0.85)
    fig.colorbar(cf, ax=ax, label="L(N,D) violation rate", shrink=0.85)

    # Scatter overlay
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
    """Heatmap of scaling exponents: rows=architectures, cols=tasks.

    Args:
        all_fits: {task_id: {arch: fit_params}} from collect across all tasks.
        exponent_key: which parameter to display ("alpha", "beta", "N_exponent", etc.)

    Returns:
        Figure.
    """
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

    # Cell annotations
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
    """N*(C) and D*(C) curves per architecture.

    Uses closed-form: N*(C) = (A*α/(B*β))^(β/(α+β)) × (C/K)^(β/(α+β))
    where K = flops_per_step/n_params ≈ constant (estimated from fit metadata).

    Args:
        fits: {arch: fit_params} for this task (from fits.json['fits']).
        task_id: for title.
        compute_budgets: array of total FLOPs to evaluate at.

    Returns:
        Figure with two subplots: N*(C) and D*(C).
    """
    task_label = _TASK_LABELS.get(task_id, task_id)
    fig, (ax_N, ax_D) = plt.subplots(1, 2, figsize=DOUBLE_COL)
    fig.suptitle(f"{task_label} — Compute-optimal allocation (Approach 3)", fontsize=11)

    for arch, fparams in fits.items():
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")
        marker = ARCH_MARKERS.get(disp, "o")

        alpha = fparams["alpha"]
        beta  = fparams["beta"]
        A = fparams["A"]
        B = fparams["B"]
        r2 = fparams.get("r_squared", fparams.get("r_squared_raw", float("nan")))
        N_exp = fparams.get("N_exponent", beta / (alpha + beta))
        D_exp = fparams.get("D_exponent", alpha / (alpha + beta))

        # Coefficient: at optimum dL/dN = dL/dD → A*α/N^(α+1) = B*β/D^(β+1)
        # → N*/D* = (A*α)/(B*β) × (D/N)^? → solve with C ≈ K*N*D
        # Use K ≈ 6 (rough FLOPs/param estimate)
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

    Shows scatter of observed (C, N*) and (C, D*) envelope points alongside
    the fitted power-law lines:
        N*(C) = a_N × C^n_exp
        D*(C) = a_D × C^d_exp

    Args:
        fits: {arch: fit_params} from fits_approach1.json['fits'].
        task_id: for title.
        compute_budgets: array of total FLOPs for the fitted lines.

    Returns:
        Figure with two subplots: N*(C) and D*(C).
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
        env_VR = np.array(fparams["envelope_VR"])

        # Scatter: empirical optimal points, coloured by VR magnitude
        ax_N.scatter(env_C, env_N, color=color, s=50, zorder=4, marker=marker,
                     edgecolors="white", linewidths=0.5)
        ax_D.scatter(env_C, env_D, color=color, s=50, zorder=4, marker=marker,
                     edgecolors="white", linewidths=0.5)

        # Fitted power-law lines
        N_fit = a_N * np.power(compute_budgets, n_exp)
        D_fit = a_D * np.power(compute_budgets, d_exp)
        ax_N.plot(compute_budgets, N_fit, color=color, linewidth=1.8, linestyle="--",
                  label=f"{disp}  a={n_exp:.2f}  R²={r2_N:.2f}")
        ax_D.plot(compute_budgets, D_fit, color=color, linewidth=1.8, linestyle="--",
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


def plot_isoflop_envelope(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """Compute-performance frontier: total FLOPs C vs best violation_rate per arch.

    Uses only terminal (C, VR) points from best_by_size_d — one clean point
    per (arch, size, D) cell with best LR already selected. The lower envelope
    of these points traces the best achievable VR at each compute budget.

    Crossover points show where architecture rankings change with scale.

    Args:
        best_by_size_d: dict keyed "arch/size/d_name" from results.json.
        task_id: task identifier for title.

    Returns:
        Figure with all 3 architectures on the same axes.
    """
    task_label = _TASK_LABELS.get(task_id, task_id)

    # Collect terminal (C, N, D, VR) points per arch
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
        arch_points.setdefault(arch, []).append((C, n_p, D_seen, vr))

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

        pts_sorted = sorted(pts, key=lambda x: x[0])  # sort by C
        Cs  = np.array([p[0] for p in pts_sorted], dtype=float)
        VRs = np.array([p[3] for p in pts_sorted], dtype=float)

        # Scatter all terminal (C, VR) points
        ax.scatter(Cs, VRs, color=color, alpha=0.35, s=20, zorder=2)

        # Lower envelope: progressive min over terminal points only
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


def plot_N_D_regime_map(
    best_by_size_d: dict[str, dict],
    task_id: str,
    fits: dict[str, dict] | None = None,
) -> dict[str, plt.Figure]:
    """N-D scatter colored by VR with regime boundary and optional L(N,D) contours.

    This is the primary diagnostic figure for Approach 1+3:
    - Each point = one (arch, size, D-budget) terminal measurement
    - Color = violation rate (RdYlGn_r: red=bad, green=good)
    - Dashed line: N = D/6 (Chinchilla unique-token boundary)
    - Solid curves: iso-VR contours from Approach 3 L(N,D) fit (if provided)

    Shows simultaneously: which cells are in valid regime, how VR distributes
    across the (N, D) grid, and how well the fitted surface matches observations.

    Args:
        best_by_size_d: dict keyed "arch/size/d_name" from results.json.
        task_id: task identifier.
        fits: optional {arch: fit_params} from fits.json for contour overlay.

    Returns:
        dict mapping arch → Figure.
    """
    task_label = _TASK_LABELS.get(task_id, task_id)

    # Collect (N, D, VR) per arch
    arch_data: dict[str, list[tuple[float, float, float, str]]] = {}
    for cell_key, traj in best_by_size_d.items():
        arch = traj["arch"]
        size = traj.get("size", "")
        pt = traj.get("terminal", {})
        N = float(traj.get("n_params", 0))
        D = float(pt.get("D_seen", 0))
        vr = pt.get("violation_rate")
        if N <= 0 or D <= 0 or vr is None:
            continue
        arch_data.setdefault(arch, []).append((N, D, float(vr), size))

    figs: dict[str, plt.Figure] = {}
    for arch, pts in arch_data.items():
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.set_title(f"{_arch_display(arch)} — {task_label}\nN-D Regime Map", fontsize=10)

        Ns = np.array([p[0] for p in pts])
        Ds = np.array([p[1] for p in pts])
        VRs = np.array([p[2] for p in pts])

        # Scatter colored by VR
        sc = ax.scatter(Ns, Ds, c=VRs, cmap="RdYlGn_r", vmin=0, vmax=1,
                        s=60, zorder=3, edgecolors="white", linewidths=0.4)
        fig.colorbar(sc, ax=ax, label="Violation rate", shrink=0.85)

        # Annotate size labels
        for N, D, vr, size in pts:
            ax.annotate(size.replace("chinchilla_", "c"),
                        (N, D), fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points", color="#444444")

        # Regime boundary: N = D/6 (Chinchilla unique-token condition)
        N_range = np.geomspace(Ns.min() * 0.5, Ns.max() * 2, 100)
        ax.plot(N_range, 6 * N_range, "--", color="gray", linewidth=1.2,
                label="D = 6N (regime boundary)", zorder=2)
        ax.text(N_range[-1] * 0.6, 6 * N_range[-1] * 1.15,
                "D=6N", fontsize=7, color="gray", style="italic")

        # Approach 3 contours (if fit provided)
        if fits and arch in fits:
            fp = fits[arch]
            # Use logit fit if available, else direct
            fp_use = fp.get("fit_logit") or fp.get("fit_direct") or fp
            E = fp_use.get("E", 0)
            A = fp_use.get("A", 1)
            alpha = fp_use.get("alpha", 0.5)
            B = fp_use.get("B", 1)
            beta = fp_use.get("beta", 0.5)
            transform = fp_use.get("transform", "direct")

            N_grid = np.geomspace(Ns.min() * 0.3, Ns.max() * 3, 80)
            D_grid = np.geomspace(Ds.min() * 0.3, Ds.max() * 3, 80)
            NN, DD = np.meshgrid(N_grid, D_grid)
            LL = E + A * np.power(NN, -alpha) + B * np.power(DD, -beta)
            if transform == "logit":
                LL = 1.0 / (1.0 + np.exp(-LL))  # inverse-logit → VR space

            contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
            cs = ax.contour(NN, DD, LL, levels=contour_levels, colors="steelblue",
                            linewidths=0.8, alpha=0.7, zorder=1)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model parameters N")
        ax.set_ylabel("Training molecules D")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        try:
            fig.tight_layout()
        except RuntimeError:
            pass  # colorbar conflicts with tight_layout
        figs[arch] = fig

    return figs


def plot_cross_task_summary(
    all_fits: dict[str, dict],
) -> plt.Figure:
    """α vs β scatter plot for all (task, arch) combinations.

    Each marker = one (task, arch) pair.
    Color = architecture, marker shape = task.
    Shows how scaling strategy varies with task complexity.

    Args:
        all_fits: {task_id: {arch: fit_params}}.

    Returns:
        Figure.
    """
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    plotted_tasks: set[str] = set()
    plotted_archs: set[str] = set()

    for task_id, arch_fits in all_fits.items():
        task_label = _TASK_LABELS.get(task_id, task_id)
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

    # Reference line α = β (equal model and data scaling)
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.5)
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.4,
            label="α = β")

    # Legend: architecture colors
    for arch in sorted(plotted_archs):
        disp = _arch_display(arch)
        ax.scatter([], [], color=ARCH_COLORS.get(disp, "gray"),
                   label=disp, s=60, marker="o")
    # Legend: task markers
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


# ── New analysis plots ───────────────────────────────────────────────────

_ARCHS_ORDERED = ["painn", "transformer", "pairformer"]


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
                           VR=vr, d_name=d_name, size=size))
    return points


def plot_smooth_envelope(
    best_by_size_d: dict[str, dict],
    task_id: str,
    d_unique_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Chinchilla Approach 1: sigmoid envelope fit in log-compute space.

    Fits VR(C) = E + (1−E) / (1 + exp(b·(log₁₀C − a))) per architecture.
    This naturally captures:
      - VR → 1  at low compute (model hasn't learned)
      - VR → E  at high compute (irreducible floor)
      - Smooth S-shaped transition in log-C space

    The scaling exponent α is reported as the power-law slope in the
    converged regime (VR < 0.95): α = −d(log VR)/d(log C).
    """
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

        # Scatter all points (faded)
        ax.scatter(Cs, VRs, color=color, alpha=0.2, s=15, zorder=2)

        # Build lower envelope: progressive min
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

        # Scatter envelope points
        ax.scatter(env_C, env_VR, color=color, s=45, zorder=5, marker=marker,
                   edgecolors="white", linewidths=0.5)

        # Sigmoid fit in log₁₀(C) space:
        #   VR(logC) = E + (1 - E) / (1 + exp(b * (logC - a)))
        # where a = transition midpoint, b = steepness, E = floor
        label = disp
        if len(env_C) >= 3:
            log_C_env = np.log10(env_C)
            try:
                def _sigmoid_vr(logC, a, b, E):
                    return E + (1.0 - E) / (1.0 + np.exp(b * (logC - a)))

                # Initial guesses from data
                vr_min = float(env_VR.min())
                logC_mid = float(np.median(log_C_env))
                p0 = [logC_mid, 2.0, max(vr_min * 0.5, 1e-4)]
                bounds = (
                    [log_C_env.min() - 2, 0.1, 0],
                    [log_C_env.max() + 2, 20.0, min(vr_min + 0.1, 0.99)],
                )
                popt, _ = curve_fit(
                    _sigmoid_vr, log_C_env, env_VR,
                    p0=p0, bounds=bounds, maxfev=50000,
                )
                a_fit, b_fit, E_fit = popt

                # Draw smooth fitted curve
                C_smooth = np.geomspace(env_C.min() * 0.5, env_C.max() * 3, 400)
                VR_smooth = _sigmoid_vr(np.log10(C_smooth), a_fit, b_fit, E_fit)
                ax.plot(C_smooth, VR_smooth, "-", color=color, linewidth=2.5,
                        zorder=4, alpha=0.9)

                # Report power-law exponent from converged envelope subset
                converged = env_VR < 0.95
                if converged.sum() >= 2:
                    coeffs = np.polyfit(
                        np.log10(env_C[converged]),
                        np.log10(np.clip(env_VR[converged], 1e-6, None)),
                        1,
                    )
                    alpha_slope = -coeffs[0]
                    label = f"{disp}  α={alpha_slope:.3f}"
                else:
                    label = f"{disp}  (floor={E_fit:.2f})"
            except Exception:
                # Fallback: connect envelope points
                ax.plot(env_C, env_VR, "-", color=color, linewidth=2.0,
                        zorder=4, alpha=0.9)
        elif len(env_C) >= 2:
            ax.plot(env_C, env_VR, "-", color=color, linewidth=2.0,
                    zorder=4, alpha=0.9)

        # Dummy artist for legend
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
    """N*(C) and D*(C) from IsoFLOP profile minima (Chinchilla Approach 1).

    For each D-budget level (D1–D4), the IsoFLOP profile is VR vs N at
    fixed D.  The minimum of each profile gives:
        N*(D_k) = argmin_N VR(N, D_k)
        C*(D_k) = FLOPs_per_step(N*) × steps(D_k)
        D*(D_k) = D_seen at that cell

    These (C*, N*, D*) triplets — one per D level — are the proper
    Chinchilla Approach 1 optimal allocation points.  Power-law fits
    N*(C) ∝ C^a  and  D*(C) ∝ C^b  are fitted on these points.

    Additionally, for each D level, a quadratic fit in log-log space
    on the IsoFLOP curve gives a continuous N*(C) estimate via the
    fitted curve's minimum, which is overlaid as an interpolated line.
    """
    from scipy.interpolate import PchipInterpolator

    task_label = _TASK_LABELS.get(task_id, task_id)
    points = _extract_terminal_points(best_by_size_d)

    fig, (ax_N, ax_D) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Compute-Optimal Allocation (IsoFLOP minima) — {task_label}", fontsize=13)

    for arch in _ARCHS_ORDERED:
        disp = _arch_display(arch)
        color = ARCH_COLORS.get(disp, "gray")
        marker = ARCH_MARKERS.get(disp, "o")

        pts = [p for p in points if p["arch"] == arch]
        if not pts:
            continue

        # Group by d_name → IsoFLOP profile per D level
        by_d: dict[str, list] = {}
        for p in pts:
            by_d.setdefault(p["d_name"], []).append(p)

        # For each D level, find the minimum of VR vs N (IsoFLOP minimum)
        opt_C, opt_N, opt_D, opt_VR = [], [], [], []
        for d_name in sorted(by_d.keys()):
            d_pts = sorted(by_d[d_name], key=lambda p: p["N"])
            if len(d_pts) < 2:
                continue

            Ns = np.array([p["N"] for p in d_pts])
            VRs = np.array([p["VR"] for p in d_pts])
            Cs = np.array([p["C"] for p in d_pts])
            Ds = np.array([p["D_seen"] for p in d_pts])

            # Fit quadratic in log(N) vs log(VR) to find smooth minimum
            log_N = np.log10(Ns)
            log_VR = np.log10(np.clip(VRs, 1e-6, None))

            if len(d_pts) >= 3:
                # Quadratic fit: log(VR) = c2*log(N)^2 + c1*log(N) + c0
                coeffs = np.polyfit(log_N, log_VR, min(2, len(d_pts) - 1))
                if len(coeffs) == 3 and coeffs[0] > 0:
                    # Parabola opens up → minimum exists
                    log_N_star = -coeffs[1] / (2 * coeffs[0])
                    # Clamp to observed range
                    log_N_star = np.clip(log_N_star, log_N.min(), log_N.max())
                    N_star = 10 ** log_N_star
                else:
                    # Monotone: take raw argmin
                    best_idx = int(np.argmin(VRs))
                    N_star = Ns[best_idx]
            else:
                best_idx = int(np.argmin(VRs))
                N_star = Ns[best_idx]

            # Interpolate C and D at N_star
            # C ∝ FLOPs_per_step(N) × steps — interpolate in log space
            if len(Cs) >= 2:
                try:
                    interp_C = PchipInterpolator(log_N, np.log10(Cs))
                    C_star = 10 ** float(interp_C(np.log10(N_star)))
                except Exception:
                    best_idx = int(np.argmin(np.abs(Ns - N_star)))
                    C_star = Cs[best_idx]
            else:
                best_idx = int(np.argmin(np.abs(Ns - N_star)))
                C_star = Cs[best_idx]

            D_star = float(np.median(Ds))  # D is fixed within a D-level
            VR_star = float(np.min(VRs))

            opt_C.append(C_star)
            opt_N.append(N_star)
            opt_D.append(D_star)
            opt_VR.append(VR_star)

        if not opt_C:
            continue

        opt_C = np.array(opt_C)
        opt_N = np.array(opt_N)
        opt_D = np.array(opt_D)
        opt_VR = np.array(opt_VR)

        # Scatter: IsoFLOP-optimal points (one per D level)
        converged = opt_VR < 0.99
        unconverged = ~converged
        if converged.any():
            ax_N.scatter(opt_C[converged], opt_N[converged], color=color, s=80,
                         zorder=5, marker=marker, edgecolors="white", linewidths=0.5)
            ax_D.scatter(opt_C[converged], opt_D[converged], color=color, s=80,
                         zorder=5, marker=marker, edgecolors="white", linewidths=0.5)
        if unconverged.any():
            ax_N.scatter(opt_C[unconverged], opt_N[unconverged], facecolors="none",
                         edgecolors=color, s=50, zorder=3, marker=marker,
                         linewidths=1.0, alpha=0.4)
            ax_D.scatter(opt_C[unconverged], opt_D[unconverged], facecolors="none",
                         edgecolors=color, s=50, zorder=3, marker=marker,
                         linewidths=1.0, alpha=0.4)

        # Connect optimal points with line (shows trend)
        order = np.argsort(opt_C)
        ax_N.plot(opt_C[order], opt_N[order], "-", color=color, linewidth=1.0,
                  alpha=0.5, zorder=4)
        ax_D.plot(opt_C[order], opt_D[order], "-", color=color, linewidth=1.0,
                  alpha=0.5, zorder=4)

        # Fit power law on converged points
        if converged.sum() >= 2:
            log_C = np.log10(opt_C[converged])
            C_line = np.geomspace(opt_C[converged].min() * 0.5,
                                  opt_C[converged].max() * 2, 100)

            coeffs_N = np.polyfit(log_C, np.log10(opt_N[converged]), 1)
            n_exp = coeffs_N[0]
            N_pred = 10 ** np.polyval(coeffs_N, log_C)
            ss_res = np.sum((np.log10(opt_N[converged]) - np.log10(N_pred)) ** 2)
            ss_tot = np.sum((np.log10(opt_N[converged]) - np.log10(opt_N[converged]).mean()) ** 2)
            r2_N = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax_N.plot(C_line, 10 ** np.polyval(coeffs_N, np.log10(C_line)),
                      "--", color=color, linewidth=1.8,
                      label=f"{disp}  a={n_exp:.2f}  R²={r2_N:.2f}")

            coeffs_D = np.polyfit(log_C, np.log10(opt_D[converged]), 1)
            d_exp = coeffs_D[0]
            D_pred = 10 ** np.polyval(coeffs_D, log_C)
            ss_res = np.sum((np.log10(opt_D[converged]) - np.log10(D_pred)) ** 2)
            ss_tot = np.sum((np.log10(opt_D[converged]) - np.log10(opt_D[converged]).mean()) ** 2)
            r2_D = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax_D.plot(C_line, 10 ** np.polyval(coeffs_D, np.log10(C_line)),
                      "--", color=color, linewidth=1.8,
                      label=f"{disp}  b={d_exp:.2f}  R²={r2_D:.2f}")
        else:
            ax_N.plot([], [], "--", color=color, linewidth=1.8,
                      label=f"{disp}  (insufficient converged pts)")

    ax_N.set_title("N*(C) = a_N · C^a", fontsize=10)
    ax_D.set_title("D*(C) = a_D · C^b", fontsize=10)
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
    """Per-arch subplots: VR vs total FLOPs, one line per data budget.

    Each line = one data budget level (D1–D4), connecting model sizes by
    ascending FLOPs. Color intensity = data budget (light→dark blue).
    Model size annotations on each point.

    Args:
        best_by_size_d: from results.json.
        task_id: for title.
        d_unique_labels: optional {d_name: label} for unique data sizes,
            e.g. {"D1": "10K", "D2": "20K", ...}.
    """
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

        by_d: dict[str, list] = {}
        for p in pts:
            by_d.setdefault(p["d_name"], []).append(p)

        for d_name in sorted(by_d.keys()):
            d_pts = sorted(by_d[d_name], key=lambda p: p["C"])
            Cs = np.array([p["C"] for p in d_pts])
            VRs = np.array([p["VR"] for p in d_pts])
            Ns = np.array([p["N"] for p in d_pts])

            color = _D_COLORS[_D_NAMES.index(d_name)] if d_name in _D_NAMES else "#333"
            d_seen_val = d_pts[0]["D_seen"] if d_pts else 0

            # Build label: prefer unique count if available
            if d_name in d_unique_labels:
                lbl = f"{d_name}: {d_unique_labels[d_name]} uniq ({d_seen_val/1e6:.1f}M seen)"
            else:
                lbl = _d_label(d_name, d_seen_val)

            ax.plot(Cs, VRs, "o-", color=color, linewidth=1.8, markersize=5,
                    label=lbl, zorder=3)

            # Annotate model sizes
            for c, vr, n in zip(Cs, VRs, Ns):
                if n >= 1e6:
                    s = f"{n/1e6:.1f}M"
                elif n >= 1e3:
                    s = f"{n/1e3:.0f}K"
                else:
                    s = str(n)
                ax.annotate(s, (c, vr), fontsize=5.5, color=color,
                            xytext=(2, 4), textcoords="offset points", alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total FLOPs C", fontsize=10)
        ax.set_ylabel("Violation rate", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        ax.grid(True, which="minor", alpha=0.1)

    return fig


def plot_vr_vs_flops_by_params(
    best_by_size_d: dict[str, dict],
    task_id: str,
) -> plt.Figure:
    """Per-arch subplots: VR vs total FLOPs, one line per model size.

    Each line = one model size (chinchilla_1–13), connecting data budgets
    by ascending FLOPs. Color gradient: small model = light, large = dark.
    Data budget annotations on each point.
    """
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

        # Group by model size
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

            # Label with param count
            if N >= 1e6:
                n_str = f"{N/1e6:.1f}M"
            elif N >= 1e3:
                n_str = f"{N/1e3:.0f}K"
            else:
                n_str = str(N)
            lbl = f"{size.replace('chinchilla_', 'c')}: {n_str}"

            ax.plot(Cs, VRs, "o-", color=color, linewidth=1.8, markersize=5,
                    label=lbl, zorder=3)

            # Annotate data budgets
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
