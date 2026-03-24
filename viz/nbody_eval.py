"""Compare generated samples vs MCMC reference for n-body tasks.

Reads a generated .npz (from evaluate.py) and reference train.npz, then plots:
  1. Energy distribution overlay (generated vs reference)
  2. Per-body energy decomposition overlay
  3. Pairwise distance distribution g(r) overlay
  4. Min pairwise distance overlay
  5. Energy QQ-plot (quantile-quantile)
  6. W2 convergence (subsample sweep)

Usage
-----
    # Single temperature
    uv run viz/nbody_eval.py \\
        --gen outputs/eval/transformer_nbody_T1.0/generated.npz \\
        --ref outputs/data/nbody_n15_b2_T1.0/train.npz

    # Multiple temperatures side by side
    uv run viz/nbody_eval.py \\
        --gen outputs/eval/transformer_nbody_T0.6/generated.npz \\
             outputs/eval/transformer_nbody_T1.0/generated.npz \\
             outputs/eval/transformer_nbody_T2.0/generated.npz \\
             outputs/eval/transformer_nbody_T3.0/generated.npz \\
        --ref outputs/data/nbody_n15_b2_T0.6/train.npz \\
              outputs/data/nbody_n15_b2_T1.0/train.npz \\
              outputs/data/nbody_n15_b2_T2.0/train.npz \\
              outputs/data/nbody_n15_b2_T3.0/train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz.style import save_figure, synthbench_style
from metrics.wasserstein_distance import _w2_1d

_COLORS_GEN = ["#C44E52", "#E07B54", "#CCB974", "#8172B3"]
_COLORS_REF = ["#4C72B0", "#55A868", "#64B5CD", "#8C8C8C"]


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    d = {k: data[k] for k in data.files}
    for k in list(d.keys()):
        if isinstance(d[k], np.ndarray) and d[k].ndim == 0:
            d[k] = d[k].item()
    return d


def _label(d: dict, path: str) -> str:
    parts = []
    if "T" in d:
        parts.append(f"T={d['T']}")
    if "body" in d:
        parts.append(f"b={d['body']}")
    return ", ".join(parts) if parts else Path(path).stem


def _compute_energies(positions: np.ndarray, ref_data: dict,
                      bc_override: str | None = None) -> np.ndarray:
    """Compute total energies from positions using n-body potential.

    Args:
        bc_override: if set, overrides the boundary condition from ref_data.
                     Use 'hard_wall' for open-boundary evaluation.
    """
    from data.generate_nbody import PotentialParams, total_energy
    bc = bc_override if bc_override else str(ref_data.get("boundary", "pbc"))
    params = PotentialParams(
        body=int(ref_data["body"]),
        sigma=float(ref_data["sigma"]),
        epsilon=float(ref_data["epsilon"]),
        nu=float(ref_data.get("nu", 1.0)),
        mu=float(ref_data.get("mu", 0.2)),
        box_size=float(ref_data["box_size"]),
        bc=bc,
    )
    energies = np.empty(len(positions))
    for i in range(len(positions)):
        et, _, _, _ = total_energy(positions[i], params)
        energies[i] = et
    return energies


def _hist_density(ax, data, bins, color, label, alpha=0.3):
    """Histogram normalized by TOTAL sample count (area = fraction in range)."""
    bw = bins[1] - bins[0]
    weights = np.ones(len(data)) / (len(data) * bw)
    ax.hist(data, bins=bins, weights=weights, alpha=alpha, color=color, label=label)
    ax.hist(data, bins=bins, weights=weights, histtype="step",
            linewidth=1.5, color=color)


def plot_energy_overlay(gen_list, ref_list, labels, outdir: Path) -> None:
    """1. Energy distribution: generated vs reference per temperature.

    Single row showing the full data range (union of ref and gen).
    """
    n = len(gen_list)
    with synthbench_style():
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            E_ref = ref["energies"]
            E_gen = gen["_energies"]

            ax = axes[0, i]
            # Full range covering both ref and gen
            all_E = np.concatenate([E_ref, E_gen])
            lo = np.percentile(all_E, 0.5)
            hi = np.percentile(all_E, 99.5)
            margin = 0.1 * (hi - lo)
            lo -= margin
            hi += margin
            bins = np.linspace(lo, hi, 80)

            w2_full = _w2_1d(E_gen, E_ref)

            _hist_density(ax, E_ref, bins, _COLORS_REF[i % len(_COLORS_REF)], "Reference")
            _hist_density(ax, E_gen, bins, _COLORS_GEN[i % len(_COLORS_GEN)], "Generated")

            ax.set_xlabel("Total Energy")
            ax.set_ylabel("Density")
            ax.set_title(f"{label}")
            info = (f"W2={w2_full:.2f}\n"
                    f"μ_ref={E_ref.mean():.1f}, σ_ref={E_ref.std():.1f}\n"
                    f"μ_gen={E_gen.mean():.1f}, σ_gen={E_gen.std():.1f}")
            ax.text(0.97, 0.97, info, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, outdir / "energy_overlay")


def plot_energy_all_overlay(gen_list, ref_list, labels, outdir: Path) -> None:
    """All temperatures overlaid on a single plot (ref + gen)."""
    with synthbench_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Global x range
        all_E = []
        for gen, ref in zip(gen_list, ref_list):
            all_E.append(ref["energies"])
            all_E.append(gen["_energies"])
        all_E_cat = np.concatenate(all_E)
        lo = np.percentile(all_E_cat, 0.5)
        hi = np.percentile(all_E_cat, 99.5)
        margin = 0.1 * (hi - lo)
        bins = np.linspace(lo - margin, hi + margin, 100)

        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            E_ref = ref["energies"]
            E_gen = gen["_energies"]
            c_ref = _COLORS_REF[i % len(_COLORS_REF)]
            c_gen = _COLORS_GEN[i % len(_COLORS_GEN)]

            ax1.hist(E_ref, bins=bins, density=True, histtype="step",
                     linewidth=1.5, color=c_ref, label=f"{label}")
            ax2.hist(E_gen, bins=bins, density=True, histtype="step",
                     linewidth=1.5, color=c_gen, label=f"{label}")

        ax1.set_xlabel("Total Energy")
        ax1.set_ylabel("Density")
        ax1.set_title("Reference (all T)")
        ax1.legend(fontsize=8)

        ax2.set_xlabel("Total Energy")
        ax2.set_ylabel("Density")
        ax2.set_title("Generated (all T)")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        save_figure(fig, outdir / "energy_all_overlay")


def plot_pairwise_overlay(gen_list, ref_list, labels, outdir: Path) -> None:
    """2. Pairwise distance g(r) overlay."""
    n = len(gen_list)
    with synthbench_style():
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            ax = axes[0, i]
            sigma = float(ref.get("sigma", 1.0))
            box_size = float(ref["box_size"])
            rng = np.random.default_rng(0)
            rmax = box_size / 2

            for source, clr, lbl in [
                (ref["positions"], _COLORS_REF[i % len(_COLORS_REF)], "Reference"),
                (gen["positions"], _COLORS_GEN[i % len(_COLORS_GEN)], "Generated"),
            ]:
                idx = rng.choice(len(source), min(500, len(source)), replace=False)
                all_d = []
                for j in idx:
                    diff = source[j][:, None, :] - source[j][None, :, :]
                    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                    triu = dist[np.triu_indices(len(source[j]), k=1)]
                    all_d.append(triu)
                all_d = np.concatenate(all_d)
                ax.hist(all_d, bins=100, density=True, alpha=0.3, color=clr,
                        label=lbl, range=(0, rmax))
                ax.hist(all_d, bins=100, density=True, histtype="step",
                        linewidth=1.5, color=clr, range=(0, rmax))
            ax.axvline(sigma, color="gray", ls="--", lw=1, label=f"σ={sigma}")
            ax.set_xlabel("Pairwise Distance r")
            ax.set_ylabel("Density")
            ax.set_title(label)
            ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, outdir / "pairwise_overlay")


def plot_min_distance_overlay(gen_list, ref_list, labels, outdir: Path) -> None:
    """3. Min pairwise distance overlay."""
    n = len(gen_list)
    with synthbench_style():
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            ax = axes[0, i]
            sigma = float(ref.get("sigma", 1.0))

            for source, clr, lbl in [
                (ref["positions"], _COLORS_REF[i % len(_COLORS_REF)], "Reference"),
                (gen["positions"], _COLORS_GEN[i % len(_COLORS_GEN)], "Generated"),
            ]:
                n_check = min(len(source), 1000)
                min_dists = np.empty(n_check)
                for j in range(n_check):
                    diff = source[j][:, None, :] - source[j][None, :, :]
                    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                    np.fill_diagonal(dist, 999.0)
                    min_dists[j] = dist.min()
                ax.hist(min_dists, bins=50, density=True, alpha=0.3, color=clr,
                        label=lbl)
                ax.hist(min_dists, bins=50, density=True, histtype="step",
                        linewidth=1.5, color=clr)
            ax.axvline(sigma, color="gray", ls="--", lw=1, label=f"σ={sigma}")
            ax.set_xlabel("Min Pairwise Distance")
            ax.set_ylabel("Density")
            ax.set_title(label)
            ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, outdir / "min_distance_overlay")


def plot_qq(gen_list, ref_list, labels, outdir: Path) -> None:
    """4. Energy QQ-plot (quantile-quantile)."""
    n = len(gen_list)
    with synthbench_style():
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            ax = axes[0, i]
            E_ref = np.sort(ref["energies"])
            E_gen = np.sort(gen["_energies"])
            # Interpolate to same size
            m = min(len(E_ref), len(E_gen))
            q = np.linspace(0, 1, m)
            qr = np.interp(q, np.linspace(0, 1, len(E_ref)), E_ref)
            qg = np.interp(q, np.linspace(0, 1, len(E_gen)), E_gen)

            ax.scatter(qr, qg, s=2, alpha=0.5, color=_COLORS_GEN[i % len(_COLORS_GEN)])
            lims = [min(qr.min(), qg.min()), max(qr.max(), qg.max())]
            ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y=x")
            ax.set_xlabel("Reference Quantiles")
            ax.set_ylabel("Generated Quantiles")
            ax.set_title(f"{label}")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
        fig.tight_layout()
        save_figure(fig, outdir / "energy_qq")


def plot_w2_convergence(gen_list, ref_list, labels, outdir: Path) -> None:
    """5. W2 convergence: how W2 changes with subsample size."""
    with synthbench_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            E_ref = ref["energies"]
            E_gen = gen["_energies"]
            sizes = np.unique(np.geomspace(50, len(E_gen), 20).astype(int))
            w2s = []
            rng = np.random.default_rng(42)
            for s in sizes:
                idx = rng.choice(len(E_gen), s, replace=False)
                w2s.append(_w2_1d(E_gen[idx], E_ref))
            c = _COLORS_GEN[i % len(_COLORS_GEN)]
            ax.plot(sizes, w2s, "o-", color=c, ms=3, lw=1.5, label=label)
        ax.set_xlabel("Number of Generated Samples")
        ax.set_ylabel("W2 Distance")
        ax.set_title("W2 Convergence")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, outdir / "w2_convergence")


def plot_high_energy_structures(gen_list, ref_list, labels, outdir: Path,
                                n_samples: int = 4) -> None:
    """6. Visualize generated samples with highest total energy.

    For each temperature, pick the n_samples with largest |energy| and render
    their 3D structures so we can see what's going wrong (clashes, out-of-box, etc.).
    """
    from viz.structure import plot_structure
    n = len(gen_list)
    with synthbench_style():
        fig = plt.figure(figsize=(4 * n_samples, 4 * n))
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            E_gen = gen["_energies"]
            positions = gen["positions"]
            sigma = float(ref.get("sigma", 1.0))
            box_size = float(ref["box_size"])
            radius = sigma / 2.0

            # Pick samples with highest energy
            worst_idx = np.argsort(np.abs(E_gen))[-n_samples:][::-1]

            for j, idx in enumerate(worst_idx):
                ax = fig.add_subplot(n, n_samples, i * n_samples + j + 1,
                                     projection="3d")
                plot_structure(positions[idx], radius, box_size, ax=ax,
                               title=f"{label}\nE={E_gen[idx]:.1e}",
                               draw_box=True)
        fig.suptitle("Highest-Energy Generated Samples", fontsize=14, y=1.02)
        fig.tight_layout()
        save_figure(fig, outdir / "high_energy_structures")


def plot_energy_filtered(gen_list, ref_list, labels, outdir: Path) -> None:
    """7. Energy-filtered comparison: keep only in-range samples, show W2 and fraction.

    Top row: filtered energy overlay (area=1 for both, shape comparison).
    Bottom row: bar chart summary across temperatures.
    """
    n = len(gen_list)
    filter_stats = []

    # Pre-compute global x range across all temperatures
    global_lo = float("inf")
    global_hi = float("-inf")
    for gen, ref in zip(gen_list, ref_list):
        E_ref = ref["energies"]
        ref_std = E_ref.std()
        lo = E_ref.min() - 3 * ref_std
        hi = E_ref.max() + 3 * ref_std
        global_lo = min(global_lo, lo)
        global_hi = max(global_hi, hi)
    global_bins = np.linspace(global_lo, global_hi, 80)

    with synthbench_style():
        # --- Per-temperature filtered histograms (shared x-axis) ---
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, (gen, ref, label) in enumerate(zip(gen_list, ref_list, labels)):
            ax = axes[0, i]
            E_ref = ref["energies"]
            E_gen = gen["_energies"]
            ref_std = E_ref.std()

            lo = E_ref.min() - 3 * ref_std
            hi = E_ref.max() + 3 * ref_std
            mask = (E_gen >= lo) & (E_gen <= hi)
            E_clean = E_gen[mask]

            n_clean = len(E_clean)
            frac = n_clean / len(E_gen)
            w2 = _w2_1d(E_clean, E_ref) if n_clean > 10 else float("inf")
            filter_stats.append({"label": label, "frac": frac, "w2": w2,
                                 "n_clean": n_clean, "n_total": len(E_gen)})

            ax.hist(E_ref, bins=global_bins, density=True, alpha=0.3,
                    color=_COLORS_REF[i % len(_COLORS_REF)], label="Reference")
            ax.hist(E_ref, bins=global_bins, density=True, histtype="step",
                    linewidth=1.5, color=_COLORS_REF[i % len(_COLORS_REF)])
            if n_clean > 10:
                ax.hist(E_clean, bins=global_bins, density=True, alpha=0.3,
                        color=_COLORS_GEN[i % len(_COLORS_GEN)], label="Generated (filtered)")
                ax.hist(E_clean, bins=global_bins, density=True, histtype="step",
                        linewidth=1.5, color=_COLORS_GEN[i % len(_COLORS_GEN)])

            ax.set_xlim(global_lo, global_hi)
            ax.set_xlabel("Total Energy")
            ax.set_ylabel("Density")
            ax.set_title(label)
            gen_std = E_gen.std()
            info = (f"W2={w2:.2f}\n"
                    f"σ_ref={ref_std:.1f}, σ_gen={gen_std:.1f}\n"
                    f"{n_clean}/{len(E_gen)} kept ({frac*100:.1f}%)")
            ax.text(0.97, 0.97, info, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, outdir / "energy_filtered")

    # --- Summary bar chart ---
    with synthbench_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(n)
        labels_t = [s["label"] for s in filter_stats]
        w2s = [s["w2"] for s in filter_stats]
        fracs = [s["frac"] * 100 for s in filter_stats]

        ax1.bar(x, w2s, color=[_COLORS_GEN[i % len(_COLORS_GEN)] for i in range(n)])
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_t)
        ax1.set_ylabel("W2 Distance (filtered)")
        ax1.set_title("Energy W2 by Temperature")
        for xi, w in zip(x, w2s):
            ax1.text(xi, w + 0.3, f"{w:.1f}", ha="center", fontsize=9)

        ax2.bar(x, fracs, color=[_COLORS_REF[i % len(_COLORS_REF)] for i in range(n)])
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_t)
        ax2.set_ylabel("In-range Fraction (%)")
        ax2.set_title("Fraction of Valid Samples")
        for xi, f in zip(x, fracs):
            ax2.text(xi, f + 0.5, f"{f:.1f}%", ha="center", fontsize=9)

        fig.tight_layout()
        save_figure(fig, outdir / "energy_filtered_summary")


def plot_summary_table(gen_list, ref_list, labels) -> None:
    """Print summary metrics to console."""
    header = (f"{'Label':>15} {'N_gen':>6} {'N_ref':>6} "
              f"{'W2':>8} {'CR':>6} {'g(r)_d':>7}")
    print(f"\n{'='*60}")
    print(header)
    print(f"{'-'*60}")
    for gen, ref, label in zip(gen_list, ref_list, labels):
        E_ref = ref["energies"]
        E_gen = gen["_energies"]
        w2 = _w2_1d(E_gen, E_ref)
        cr = float(gen.get("clash_rate", -1))
        grd = float(gen.get("gr_distance", -1))
        print(f"{label:>15} {len(E_gen):>6} {len(E_ref):>6} "
              f"{w2:>8.4f} {cr:>6.4f} {grd:>7.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated vs reference n-body distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gen", nargs="+", required=True,
                        help="Generated .npz files (from evaluate.py)")
    parser.add_argument("--ref", nargs="+", required=True,
                        help="Reference .npz files (train.npz from MCMC)")
    parser.add_argument("--outdir", type=str, default="outputs/plots/nbody_eval",
                        help="Output directory for plots")
    parser.add_argument("--skip-energy-calc", action="store_true",
                        help="Skip energy recalculation if energy_w2 is in generated.npz")
    parser.add_argument("--bc", type=str, default=None, choices=["pbc", "hard_wall"],
                        help="Override boundary condition for energy calc (default: from data)")
    args = parser.parse_args()

    if len(args.gen) != len(args.ref):
        print(f"Error: --gen ({len(args.gen)} files) and --ref ({len(args.ref)} files) "
              "must have the same count.")
        sys.exit(1)

    outdir = Path(args.outdir)
    if args.bc:
        outdir = outdir / f"bc_{args.bc}"
    gen_list = []
    ref_list = []
    labels = []

    for gf, rf in zip(args.gen, args.ref):
        gen = load_npz(gf)
        ref = load_npz(rf)
        label = _label(ref, rf)
        labels.append(label)

        # Compute energies for generated positions if not cached
        if "energy_w2" in gen and args.skip_energy_calc and args.bc is None:
            # Use positions but skip energy calc — put dummy energies
            gen["_energies"] = np.zeros(len(gen["positions"]))
            print(f"  {label}: skipping energy calc (W2={gen['energy_w2']:.4f} from file)")
        else:
            bc_label = f" (bc={args.bc})" if args.bc else ""
            print(f"  {label}: computing energies for {len(gen['positions'])} generated samples{bc_label}...")
            gen["_energies"] = _compute_energies(gen["positions"], ref, bc_override=args.bc)

        # Recompute reference energies with same BC for fair comparison
        if args.bc and args.bc != str(ref.get("boundary", "pbc")):
            print(f"  {label}: recomputing reference energies with bc={args.bc}...")
            ref["energies"] = _compute_energies(ref["positions"], ref, bc_override=args.bc)

        gen_list.append(gen)
        ref_list.append(ref)

    print(f"\nLoaded {len(gen_list)} pair(s)")
    plot_summary_table(gen_list, ref_list, labels)

    print("Generating plots...")
    plot_energy_overlay(gen_list, ref_list, labels, outdir)
    print("  [1/8] energy_overlay")
    plot_energy_all_overlay(gen_list, ref_list, labels, outdir)
    print("  [2/8] energy_all_overlay")
    plot_pairwise_overlay(gen_list, ref_list, labels, outdir)
    print("  [3/8] pairwise_overlay")
    plot_min_distance_overlay(gen_list, ref_list, labels, outdir)
    print("  [4/8] min_distance_overlay")
    plot_qq(gen_list, ref_list, labels, outdir)
    print("  [5/8] energy_qq")
    plot_w2_convergence(gen_list, ref_list, labels, outdir)
    print("  [6/8] w2_convergence")
    plot_high_energy_structures(gen_list, ref_list, labels, outdir)
    print("  [7/8] high_energy_structures")
    plot_energy_filtered(gen_list, ref_list, labels, outdir)
    print("  [8/8] energy_filtered + summary")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
