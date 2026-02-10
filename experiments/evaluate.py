"""Standalone evaluation: load checkpoint, generate samples, compute metrics, save plots."""

import argparse
import os
import sys

import numpy as np
import torch

from data.dataset import HardSphereDataset
from experiments.checkpointing import load_checkpoint
from flow_matching.sampling import sample_batched
from metrics.clash_rate import clash_rate_batched
from models.painn import PaiNNVelocityNetwork
from models.pairformer import PairformerVelocityNetwork
from models.transformer import TransformerVelocityNetwork

MODEL_REGISTRY = {
    "painn": PaiNNVelocityNetwork,
    "transformer": TransformerVelocityNetwork,
    "pairformer": PairformerVelocityNetwork,
}


def build_model_from_config(config: dict, box_size: float) -> torch.nn.Module:
    """Reconstruct model from saved config dict."""
    arch = config["model"]["arch"]
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = dict(config["model"]["model_kwargs"])
    if "cutoff" in kwargs:
        kwargs["cutoff"] = box_size * 1.5
    return MODEL_REGISTRY[arch](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained velocity network")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt or latest.pt)")
    parser.add_argument("--data", required=True, help="Path to data directory (containing train.npz)")
    parser.add_argument("--output", default=None, help="Output directory (default: outputs/eval/{arch}/)")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of ODE steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    state = load_checkpoint(args.checkpoint, device=str(device))
    config = state.config
    arch = config["model"]["arch"]
    print(f"Architecture: {arch} | Step: {state.step} | Best clash rate: {state.best_clash_rate:.4f}")

    # Load dataset for metadata
    train_path = os.path.join(args.data, "train.npz")
    dataset = HardSphereDataset(train_path)
    box_size = dataset.box_size
    radius = dataset.radius
    n_atoms = dataset.positions.shape[1]
    print(f"Data: N={n_atoms}, radius={radius}, box_size={box_size:.4f}")

    # Build and load model
    model = build_model_from_config(config, box_size).to(device)
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
    print(f"\nResults:")
    print(f"  Samples generated: {args.n_samples}")
    print(f"  Clash rate:        {cr:.4f}")

    # Output directory
    output_dir = args.output or os.path.join("outputs", "eval", arch)
    os.makedirs(output_dir, exist_ok=True)

    # Save generated positions
    out_path = os.path.join(output_dir, "generated.npz")
    np.savez(
        out_path,
        positions=samples.numpy(),
        radius=radius,
        box_size=box_size,
        clash_rate=cr,
        step=state.step,
    )
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
    from data.validate import pair_correlation
    from viz.metrics import plot_gr

    pos_np = samples.numpy()
    r, g_r = pair_correlation(pos_np, box_size)

    # Ground truth g(r) from training data
    gt_pos = dataset.positions.numpy()
    gt_r, gt_g_r = pair_correlation(gt_pos, box_size)

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
