"""Chain dataset metric lower bound estimation.

Splits train data into two random halves A/B and computes metric(A, B).
This gives the finite-sample noise floor — the best any generative model
can achieve for a given evaluation sample count.

Metrics:
  - W2 on energy (total, bond, LJ, angle, dihedral)
  - W2 on structural distributions (bond length, bond angle, dihedral angle)
    computed directly from positions

Usage
-----
    uv run experiments/chain_metric_lower_bound.py \
        --datasets outputs/data/nbody_chain_N15_b4_T1.0/train.npz \
        --n_samples 1000 5000 10000 50000 \
        --n_seeds 5 \
        --outdir outputs/plots/chain_lower_bound
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz.style import save_figure, synthbench_style

# ---------------------------------------------------------------------------
# Geometry helpers: extract structural features from (B, N, 3) positions
# ---------------------------------------------------------------------------

def _bond_lengths(pos: np.ndarray) -> np.ndarray:
    """(B, N, 3) -> (B*(N-1),) bond lengths."""
    diffs = pos[:, 1:] - pos[:, :-1]          # (B, N-1, 3)
    return np.linalg.norm(diffs, axis=-1).ravel()


def _bond_angles(pos: np.ndarray) -> np.ndarray:
    """(B, N, 3) -> (B*(N-2),) bond angles in degrees."""
    v1 = pos[:, :-2] - pos[:, 1:-1]           # (B, N-2, 3)
    v2 = pos[:, 2:]  - pos[:, 1:-1]           # (B, N-2, 3)
    cos = (v1 * v2).sum(-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12
    )
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos)).ravel()


def _dihedral_angles(pos: np.ndarray) -> np.ndarray:
    """(B, N, 3) -> (B*(N-3),) dihedral angles in degrees."""
    p0, p1, p2, p3 = pos[:, :-3], pos[:, 1:-2], pos[:, 2:-1], pos[:, 3:]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-12
    n1 /= n1_norm
    n2 /= n2_norm
    cos = np.clip((n1 * n2).sum(-1), -1.0, 1.0)
    b1_unit = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
    sin = (np.cross(n1, n2) * b1_unit).sum(-1)
    return np.degrees(np.arctan2(sin, cos)).ravel()


# ---------------------------------------------------------------------------
# W2 (1-D Wasserstein-2) between two arrays
# ---------------------------------------------------------------------------

def w2_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.ravel().astype(np.float64))
    b = np.sort(b.ravel().astype(np.float64))
    if len(a) != len(b):
        n = max(len(a), len(b))
        q = np.linspace(0, 1, n)
        a = np.interp(q, np.linspace(0, 1, len(a)), a)
        b = np.interp(q, np.linspace(0, 1, len(b)), b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Per-split metric computation
# ---------------------------------------------------------------------------

def compute_split_metrics(data: dict, idx_a: np.ndarray, idx_b: np.ndarray) -> dict[str, float]:
    """Compute all W2 metrics between split A and split B."""
    result = {}

    # Energy-based W2
    for key in ("energies", "energies_bond", "energies_lj", "energies_angle", "energies_dihedral"):
        arr = data.get(key)
        if arr is None:
            continue
        a_vals = arr[idx_a]
        b_vals = arr[idx_b]
        if np.all(a_vals == 0) and np.all(b_vals == 0):
            continue
        label = {
            "energies": "W2_energy",
            "energies_bond": "W2_bond_E",
            "energies_lj": "W2_lj_E",
            "energies_angle": "W2_angle_E",
            "energies_dihedral": "W2_dihedral_E",
        }[key]
        result[label] = w2_1d(a_vals, b_vals)

    # Structural W2 (from positions)
    pos_a = data["positions"][idx_a]
    pos_b = data["positions"][idx_b]

    result["W2_bond_len"] = w2_1d(_bond_lengths(pos_a), _bond_lengths(pos_b))

    body = int(data.get("body", 2))
    if body >= 3:
        result["W2_bond_angle"] = w2_1d(_bond_angles(pos_a), _bond_angles(pos_b))
    if body >= 4:
        result["W2_dihedral"] = w2_1d(_dihedral_angles(pos_a), _dihedral_angles(pos_b))

    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _precompute_features(data: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-extract all scalar features for a set of indices (energy + structural)."""
    feats: dict[str, np.ndarray] = {}
    for key in ("energies", "energies_bond", "energies_lj", "energies_angle", "energies_dihedral"):
        arr = data.get(key)
        if arr is None:
            continue
        vals = arr[idx]
        if not (np.all(vals == 0)):
            feats[key] = vals
    pos = data["positions"][idx]
    feats["bond_len"] = _bond_lengths(pos)
    body = int(data.get("body", 2))
    if body >= 3:
        feats["bond_angle"] = _bond_angles(pos)
    if body >= 4:
        feats["dihedral"] = _dihedral_angles(pos)
    return feats


def _metrics_from_features(fa: dict, fb: dict) -> dict[str, float]:
    """Compute W2 for each matching feature key between fa and fb."""
    label_map = {
        "energies": "W2_energy",
        "energies_bond": "W2_bond_E",
        "energies_lj": "W2_lj_E",
        "energies_angle": "W2_angle_E",
        "energies_dihedral": "W2_dihedral_E",
        "bond_len": "W2_bond_len",
        "bond_angle": "W2_bond_angle",
        "dihedral": "W2_dihedral",
    }
    result = {}
    for key, label in label_map.items():
        if key in fa and key in fb:
            result[label] = w2_1d(fa[key], fb[key])
    return result


def run_lower_bound(
    npz_path: str,
    n_samples_list: list[int],
    n_seeds: int,
    rng: np.random.Generator,
    ref_mode: str = "split",
) -> dict[int, dict[str, list[float]]]:
    """For each sample count, run n_seeds trials and collect metrics.

    ref_mode:
      "split"    — W2(A_n, B_n): both halves are size n (symmetric split)
      "full_ref" — W2(A_n, B_full): A is n samples, B is the full dataset
                   Matches real evaluation where ref = entire training set.
                   Reference features are precomputed once for efficiency.
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    total = len(data["positions"])

    # full_ref: precompute reference features once (expensive but done only once)
    ref_feats: dict | None = None
    if ref_mode == "full_ref":
        print(f"  Precomputing reference features on {total} samples...", flush=True)
        ref_feats = _precompute_features(data, np.arange(total))
        print(f"  Done.", flush=True)

    results: dict[int, dict[str, list[float]]] = {}
    for n in n_samples_list:
        if n >= total:
            print(f"  [skip] n_samples={n} >= total {total}")
            continue
        if ref_mode == "split" and n * 2 > total:
            print(f"  [skip] n_samples={n} requires {n*2} samples but only {total} available")
            continue
        results[n] = {}
        for seed_i in range(n_seeds):
            idx_a = rng.choice(total, size=n, replace=False)
            fa = _precompute_features(data, idx_a)
            if ref_mode == "full_ref":
                fb = ref_feats
            else:
                # draw non-overlapping B of same size
                mask = np.ones(total, dtype=bool)
                mask[idx_a] = False
                idx_b = rng.choice(np.where(mask)[0], size=n, replace=False)
                fb = _precompute_features(data, idx_b)
            metrics = _metrics_from_features(fa, fb)
            for k, v in metrics.items():
                results[n].setdefault(k, []).append(v)
        # Print summary
        print(f"  n={n:>6d}: ", end="")
        for k, vals in results[n].items():
            print(f"{k}={np.mean(vals):.4f}±{np.std(vals):.4f}  ", end="")
        print()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lower_bounds(
    all_results: dict[str, dict[int, dict[str, list[float]]]],
    outdir: Path,
) -> None:
    """Plot W2 lower bounds vs sample size for all datasets."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Collect all metric keys across datasets
    all_keys: set[str] = set()
    for ds_results in all_results.values():
        for n_results in ds_results.values():
            all_keys.update(n_results.keys())
    all_keys = sorted(all_keys)

    # Color per dataset
    labels = list(all_results.keys())
    colors = cm.tab10(np.linspace(0, 1, len(labels)))

    with synthbench_style():
        # --- Figure 1: one subplot per metric ---
        n_metrics = len(all_keys)
        ncols = 3
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).ravel()

        for ax_i, metric_key in enumerate(all_keys):
            ax = axes[ax_i]
            for label, color in zip(labels, colors):
                ds_results = all_results[label]
                ns = sorted(ds_results.keys())
                means, stds = [], []
                for n in ns:
                    vals = ds_results[n].get(metric_key, [])
                    if vals:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                means = np.array(means)
                stds = np.array(stds)
                valid = ~np.isnan(means)
                if not valid.any():
                    continue
                ax.plot(np.array(ns)[valid], means[valid], "o-", color=color,
                        label=label, linewidth=1.5, markersize=4)
                ax.fill_between(np.array(ns)[valid],
                                (means - stds)[valid],
                                (means + stds)[valid],
                                alpha=0.15, color=color)
            ax.set_xscale("log")
            ax.set_xlabel("# samples per split")
            ax.set_ylabel("W2 lower bound")
            ax.set_title(metric_key.replace("_", " "))
            if ax_i == 0:
                ax.legend(fontsize=7, ncol=1)

        for ax in axes[n_metrics:]:
            ax.set_visible(False)

        fig.suptitle("Chain metric lower bounds (train-split W2)", fontsize=14, y=1.01)
        save_figure(fig, outdir / "lower_bound_all_metrics")
        print(f"  Saved: {outdir}/lower_bound_all_metrics.png")

        # --- Figure 2: W2_energy across all datasets, grouped by body/T ---
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        body_vals = sorted(set(int(lbl.split("b")[1].split("_")[0]) for lbl in labels if "_b" in lbl))
        T_vals = sorted(set(float(lbl.split("T")[1]) for lbl in labels if "_T" in lbl))
        T_colors = {t: c for t, c in zip(T_vals, ["#4C72B0", "#C44E52", "#55A868"])}

        for ax, body in zip(axes2, body_vals):
            for T in T_vals:
                match = [l for l in labels if f"_b{body}_" in l and f"_T{T}" in l]
                if not match:
                    continue
                label = match[0]
                ds_results = all_results[label]
                ns = sorted(ds_results.keys())
                means, stds = [], []
                for n in ns:
                    vals = ds_results[n].get("W2_energy", [])
                    if vals:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                means = np.array(means)
                stds = np.array(stds)
                valid = ~np.isnan(means)
                if not valid.any():
                    continue
                ax.plot(np.array(ns)[valid], means[valid], "o-",
                        color=T_colors[T], label=f"T={T}", linewidth=2, markersize=5)
                ax.fill_between(np.array(ns)[valid],
                                (means - stds)[valid],
                                (means + stds)[valid],
                                alpha=0.15, color=T_colors[T])
            ax.set_xscale("log")
            ax.set_xlabel("# samples per split")
            ax.set_ylabel("W2 lower bound (energy)")
            ax.set_title(f"body = {body}")
            ax.legend()

        fig2.suptitle("W2(energy) lower bound vs sample size — by body order & temperature", fontsize=13)
        save_figure(fig2, outdir / "lower_bound_energy_by_body_T")
        print(f"  Saved: {outdir}/lower_bound_energy_by_body_T.png")

        # --- Figure 3: structural lower bounds (bond_len / angle / dihedral) for b=4 ---
        b4_labels = [l for l in labels if "_b4_" in l]
        if b4_labels:
            struct_keys = [k for k in all_keys if k in ("W2_bond_len", "W2_bond_angle", "W2_dihedral")]
            fig3, axes3 = plt.subplots(1, len(struct_keys), figsize=(5 * len(struct_keys), 4))
            if len(struct_keys) == 1:
                axes3 = [axes3]
            T_all = sorted(set(float(l.split("_T")[1]) for l in b4_labels))
            for ax3, sk in zip(axes3, struct_keys):
                for T in T_all:
                    match = [l for l in b4_labels if f"_T{T}" in l]
                    if not match:
                        continue
                    label = match[0]
                    ds_results = all_results[label]
                    ns = sorted(ds_results.keys())
                    means, stds = [], []
                    for n in ns:
                        vals = ds_results[n].get(sk, [])
                        if vals:
                            means.append(np.mean(vals))
                            stds.append(np.std(vals))
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    means = np.array(means)
                    stds = np.array(stds)
                    valid = ~np.isnan(means)
                    if not valid.any():
                        continue
                    ax3.plot(np.array(ns)[valid], means[valid], "o-",
                             color=T_colors[T], label=f"T={T}", linewidth=2, markersize=5)
                    ax3.fill_between(np.array(ns)[valid],
                                     (means - stds)[valid],
                                     (means + stds)[valid],
                                     alpha=0.15, color=T_colors[T])
                ax3.set_xscale("log")
                ax3.set_xlabel("# samples per split")
                ax3.set_ylabel(f"W2 lower bound")
                ax3.set_title(sk.replace("_", " ") + " (b=4)")
                ax3.legend()
            fig3.suptitle("Structural W2 lower bounds — b=4 by temperature", fontsize=13)
            save_figure(fig3, outdir / "lower_bound_structural_b4")
            print(f"  Saved: {outdir}/lower_bound_structural_b4.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chain metric lower bound via train-split W2")
    parser.add_argument(
        "--datasets", nargs="+",
        default=[
            "outputs/data/nbody_chain_N15_b2_T0.5/train.npz",
            "outputs/data/nbody_chain_N15_b2_T1.0/train.npz",
            "outputs/data/nbody_chain_N15_b2_T2.0/train.npz",
            "outputs/data/nbody_chain_N15_b3_T0.5/train.npz",
            "outputs/data/nbody_chain_N15_b3_T1.0/train.npz",
            "outputs/data/nbody_chain_N15_b3_T2.0/train.npz",
            "outputs/data/nbody_chain_N15_b4_T0.5/train.npz",
            "outputs/data/nbody_chain_N15_b4_T1.0/train.npz",
            "outputs/data/nbody_chain_N15_b4_T2.0/train.npz",
        ],
        help="Path(s) to train.npz files",
    )
    parser.add_argument(
        "--n_samples", nargs="+", type=int,
        default=[500, 1000, 2000, 5000, 10000, 50000],
        help="Sample counts per split half to evaluate",
    )
    parser.add_argument("--n_seeds", type=int, default=5, help="Random seeds per (dataset, n_samples)")
    parser.add_argument("--outdir", default="outputs/plots/chain_lower_bound")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ref_mode", choices=["split", "full_ref"], default="split",
        help="split: W2(A_n, B_n) symmetric halves | full_ref: W2(A_n, B_full) matches real eval",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_results: dict[str, dict] = {}
    npz_by_label: dict[str, str] = {}
    for npz_path in args.datasets:
        p = Path(npz_path)
        label = p.parent.name  # e.g. "nbody_chain_N15_b4_T1.0"
        parts = label.replace("nbody_chain_", "")  # "N15_b4_T1.0"
        print(f"\n=== {parts} ({npz_path}) ===")
        all_results[parts] = run_lower_bound(npz_path, args.n_samples, args.n_seeds, rng, ref_mode=args.ref_mode)
        npz_by_label[parts] = npz_path

    # Save lower_bound.json next to each train.npz
    if args.ref_mode == "full_ref":
        import json
        for parts, ds_results in all_results.items():
            npz_path = npz_by_label[parts]
            json_path = Path(npz_path).parent / "lower_bound.json"
            payload = {
                "ref_mode": args.ref_mode,
                "n_seeds": args.n_seeds,
                "by_n": {
                    str(n): {
                        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for k, v in n_results.items()
                    }
                    for n, n_results in ds_results.items()
                },
            }
            json_path.write_text(json.dumps(payload, indent=2))
            print(f"  Saved lower_bound.json → {json_path}")

    print("\nGenerating plots...")
    plot_lower_bounds(all_results, outdir)
    print(f"\nDone. Results saved to {outdir}/")

    # Print final summary table
    print("\n" + "=" * 100)
    print(f"{'Dataset':<25} {'n_samples':>10}", end="")
    # Get all metric keys
    sample_keys: list[str] = []
    for ds in all_results.values():
        for n_res in ds.values():
            sample_keys = sorted(n_res.keys())
            break
        break
    for k in sample_keys:
        print(f"  {k[:15]:>15}", end="")
    print()
    print("-" * 100)
    for ds_label, ds_results in all_results.items():
        for n, n_results in sorted(ds_results.items()):
            print(f"{ds_label:<25} {n:>10}", end="")
            for k in sample_keys:
                vals = n_results.get(k, [])
                if vals:
                    print(f"  {np.mean(vals):>9.4f}±{np.std(vals):.4f}"[:17], end="")
                else:
                    print(f"  {'—':>15}", end="")
            print()


if __name__ == "__main__":
    main()
