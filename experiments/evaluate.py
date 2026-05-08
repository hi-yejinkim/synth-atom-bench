"""Standalone evaluation: load checkpoint, generate samples, compute metrics, save plots."""

import argparse
import os
import sys

import numpy as np
import torch

from data.chain_dataset import ChainDataset
from data.dataset import HardSphereDataset
from experiments.checkpointing import load_checkpoint
from flow_matching.sampling import sample_batched
from experiments.chinchilla_lib.helpers import load_lower_bound
from metrics.bond_violation import bond_violation_rate_batched, nonbonded_clash_rate_batched
from metrics.clash_rate import clash_rate_batched
from metrics.gr_distance import gr_distance
from metrics.wasserstein_distance import energy_w2, energy_w2_from_positions
from models.painn import PaiNNVelocityNetwork
from models.pairformer import PairformerVelocityNetwork
from models.transformer import TransformerVelocityNetwork

MODEL_REGISTRY = {
    "painn": PaiNNVelocityNetwork,
    "transformer": TransformerVelocityNetwork,
    "pairformer": PairformerVelocityNetwork,
}


def build_model_from_config(config: dict, box_size: float,
                            state_dict: dict | None = None) -> torch.nn.Module:
    """Reconstruct model from saved config dict.

    If state_dict is provided, auto-detects flags like use_chain_pe that may
    have been added after the config was originally saved.
    """
    arch = config["model"]["arch"]
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = dict(config["model"]["model_kwargs"])
    if "cutoff" in kwargs:
        kwargs["cutoff"] = box_size * 1.5
    # Auto-detect chain positional encoding from state_dict keys
    if state_dict is not None and "use_chain_pe" not in kwargs:
        if any("chain_pos_bias" in k for k in state_dict):
            kwargs["use_chain_pe"] = True
    return MODEL_REGISTRY[arch](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained velocity network")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt or latest.pt)")
    parser.add_argument("--data", required=True, help="Path to data directory (containing train.npz)")
    parser.add_argument("--output", default=None, help="Output directory (default: outputs/eval/{arch}/)")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of ODE steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for sampling")
    parser.add_argument("--wrap", action="store_true", help="Wrap positions to [0, box_size] via PBC before metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    state = load_checkpoint(args.checkpoint, device=str(device))
    config = state.config
    arch = config["model"]["arch"]
    print(f"Architecture: {arch} | Step: {state.step} | Best clash rate: {state.best_clash_rate:.4f}")

    # Load dataset for metadata (auto-detect: nbody vs chain vs hard-sphere)
    train_path = os.path.join(args.data, "train.npz")
    npz = np.load(train_path, allow_pickle=True)
    #is_nbody = "energies_2body" in npz.files and "body" in npz.files
    is_nbody = "energies" in npz.files and "body" in npz.files
    is_chain = "bond_length" in npz.files

    is_nbody_chain = is_nbody and "k2" in npz.files

    if is_nbody:
        ref_positions = npz["positions"]
        ref_energies = npz["energies"]
        n_atoms = ref_positions.shape[1]
        box_size = float(npz["box_size"])
        radius = float(npz["sigma"]) / 2.0  # clash radius = σ/2
        if is_nbody_chain:
            chain_params_dict = {
                "body": int(npz["body"]),
                "N": int(npz["N"]),
                "k2": float(npz["k2"]),
                "r0": float(npz["r0"]),
                "sigma": float(npz["sigma"]),
                "epsilon_lj": float(npz["epsilon_lj"]),
                "k3": float(npz.get("k3", 20.0)),
                "theta0": float(npz.get("theta0", 1.911)),
                "c1": float(npz.get("c1", 1.0)),
                "c2": float(npz.get("c2", 0.5)),
                "box_size": box_size,
            }
            print(f"Data: n-body chain (body={chain_params_dict['body']}), N={n_atoms}, "
                  f"T={float(npz['T'])}, box_size={box_size:.3f}")
        else:
            nbody_params = {
                "body": int(npz["body"]),
                "sigma": float(npz["sigma"]),
                "epsilon": float(npz["epsilon"]),
                "nu": float(npz.get("nu", 1.0)),
                "mu": float(npz.get("mu", 0.2)),
                "box_size": box_size,
                "bc": str(npz.get("boundary", "pbc")),
            }
            print(f"Data: n-body (body={nbody_params['body']}), N={n_atoms}, "
                  f"T={float(npz['T'])}, box_size={box_size:.3f}, bc={nbody_params['bc']}")
        ref_energy_std = float(np.std(ref_energies))
        npz.close()
    elif is_chain:
        npz.close()
        dataset = ChainDataset(train_path)
        print(f"Data: chain, N={dataset.positions.shape[1]}, bond_length={dataset.bond_length}, radius={dataset.radius}, box_size={dataset.box_size:.4f}")
        box_size = dataset.box_size
        radius = dataset.radius
        n_atoms = dataset.positions.shape[1]
    else:
        npz.close()
        dataset = HardSphereDataset(train_path)
        print(f"Data: hard-sphere, N={dataset.positions.shape[1]}, radius={dataset.radius}, box_size={dataset.box_size:.4f}")
        box_size = dataset.box_size
        radius = dataset.radius
        n_atoms = dataset.positions.shape[1]

    # Build and load model
    model = build_model_from_config(config, box_size, state.model_state_dict).to(device)
    model.load_state_dict(state.model_state_dict)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Generate samples
    print(f"Generating {args.n_samples} samples ({args.n_steps} ODE steps)...")
    model.eval()
    samples = sample_batched(
        model,
        n_atoms=n_atoms,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=str(device),
    )
    # Shift back to [0, box_size]
    samples = samples + box_size / 2

    # Compute metrics
    cr = clash_rate_batched(samples, radius)

    # n-body energy Wasserstein metrics
    w1 = None
    w2 = None
    if is_nbody:
        print("Computing energy Wasserstein distance...")
        if is_nbody_chain:
            from data.generate_nbody_chain import ChainParams, total_energy as chain_total_energy
            from metrics.wasserstein_distance import _w2_1d, _w1_1d
            chain_params = ChainParams(**{k: v for k, v in chain_params_dict.items() if k != "N"},
                                       N=chain_params_dict["N"])
            ref_e_f64 = ref_energies.astype(np.float64)
            e_clip_hi = float(ref_e_f64.mean()) + 20.0 * float(ref_e_f64.std())
            pos_np = samples.numpy()
            gen_e_f64 = np.array(
                [chain_total_energy(pos_np[i].astype(np.float64), chain_params)[0]
                 for i in range(len(pos_np))],
                dtype=np.float64,
            )
            gen_e_clipped = np.clip(gen_e_f64, None, e_clip_hi)
            ref_e_clipped = np.clip(ref_e_f64, None, e_clip_hi)
            w2 = float(_w2_1d(gen_e_clipped, ref_e_clipped))
            w1 = float(_w1_1d(gen_e_clipped, ref_e_clipped))
        else:
            wd = energy_w2_from_positions(
                samples.numpy(), ref_energies,
                **nbody_params,
            )
            w1 = wd["w1_total"]
            w2 = wd["w2_total"]

    # Ground truth g(r) for distance metric
    from data.validate import pair_correlation
    gt_pos = ref_positions if is_nbody else dataset.positions.numpy()
    gt_r, gt_g_r = pair_correlation(gt_pos, box_size)
    grd = gr_distance(samples.numpy(), gt_r, gt_g_r, box_size)

    # Chain-specific metrics
    bvr = None
    ncr = None
    if is_chain:
        bvr = bond_violation_rate_batched(samples, dataset.bond_length)
        ncr = nonbonded_clash_rate_batched(samples, dataset.radius)

    # Load lower bound for comparison (if available)
    lb = load_lower_bound(args.data, metric="W2_energy", n_eval=args.n_samples) if w2 is not None else None

    print(f"\nResults:")
    print(f"  Samples generated: {args.n_samples}")
    print(f"  Clash rate:        {cr:.4f}")
    print(f"  g(r) distance:     {grd:.4f}")
    if w1 is not None:
        print(f"  Energy W1:         {w1:.4f}")
        if lb is not None:
            ratio = w2 / lb["mean"] if lb["mean"] > 0 else float("inf")
            print(f"  Energy W2:         {w2:.4f}  (lower bound [{lb['ref_mode']}, n={lb['n_actual']}]: {lb['mean']:.4f}±{lb['std']:.4f}, ratio: {ratio:.2f}x)")
        else:
            print(f"  Energy W2:         {w2:.4f}")
    if is_chain:
        print(f"  Bond violation:    {bvr:.4f}")
        print(f"  Non-bonded clash:  {ncr:.4f}")

    # Output directory
    output_dir = args.output or os.path.join("outputs", "eval", arch)
    os.makedirs(output_dir, exist_ok=True)

    # Save generated positions
    save_kwargs = dict(
        positions=samples.numpy(),
        radius=radius,
        box_size=box_size,
        clash_rate=cr,
        gr_distance=grd,
        step=state.step,
    )
    if w1 is not None and w2 is not None:
        save_kwargs["energy_w1"] = w1
        save_kwargs["energy_w2"] = w2
    if is_chain:
        save_kwargs["bond_violation_rate"] = bvr
        save_kwargs["nonbonded_clash_rate"] = ncr
        save_kwargs["bond_length"] = dataset.bond_length
    out_path = os.path.join(output_dir, "generated.npz")
    np.savez(out_path, **save_kwargs)
    print(f"  Saved: {out_path}")

    # Plot structures grid
    from viz import save_figure, synthbench_style
    from viz.structure import plot_structures_grid

    with synthbench_style():
        n_show = min(8, args.n_samples)
        fig = plot_structures_grid(
            [samples[i].numpy() for i in range(n_show)],
            radius=radius,
            box_size=box_size,
        )
        save_figure(fig, os.path.join(output_dir, "structures"))
        print(f"  Saved: {output_dir}/structures.png")

    # Plot pair correlation g(r)
    from viz.metrics import plot_gr

    pos_np = samples.numpy()
    r, g_r = pair_correlation(pos_np, box_size)

    with synthbench_style():
        fig = plot_gr(r, g_r, radius, gt_r=gt_r, gt_g_r=gt_g_r, label=arch)
        save_figure(fig, os.path.join(output_dir, "pair_correlation"))
        print(f"  Saved: {output_dir}/pair_correlation.png")

    # Plot min distance histogram
    from viz.metrics import plot_min_distance_hist

    with synthbench_style():
        fig = plot_min_distance_hist(pos_np, radius, label=arch)
        save_figure(fig, os.path.join(output_dir, "min_distance_hist"))
        print(f"  Saved: {output_dir}/min_distance_hist.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
