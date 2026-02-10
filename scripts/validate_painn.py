"""End-to-end validation: train PaiNN on easy hard-sphere data, check it beats noise baseline."""

import sys
import torch

from data.generate import mcmc_sample
from models.painn import PaiNNVelocityNetwork
from flow_matching.training import flow_matching_loss
from flow_matching.sampling import sample
from metrics.clash_rate import clash_rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate training data
    print("Generating 1000 training samples (N=10, eta=0.1)...")
    N, radius, eta = 10, 0.5, 0.1
    positions, box_size = mcmc_sample(N=N, radius=radius, eta=eta, num_samples=1000, burn_in=5000, thin_interval=500, seed=42)
    print(f"  Box size: {box_size:.4f}")

    # Center data for flow matching (noise is N(0,I), data should be centered)
    x_train = torch.from_numpy(positions).float().to(device)
    x_train = x_train - box_size / 2

    # Create model
    model = PaiNNVelocityNetwork(hidden_dim=64, n_layers=3, cutoff=box_size * 1.5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 64
    n_steps = 1000

    print(f"Training for {n_steps} steps...")
    model.train()
    for step in range(n_steps):
        idx = torch.randint(len(x_train), (batch_size,))
        x_0 = x_train[idx]
        loss = flow_matching_loss(model, x_0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1:4d} | Loss: {loss.item():.4f}")

    # Sample
    print("Generating 100 samples...")
    model.eval()
    generated = sample(model, n_atoms=N, n_samples=100, n_steps=100, device=device)
    generated = generated + box_size / 2  # shift back to [0, box_size)

    # Baseline: random noise (not centered, just random in box)
    baseline = torch.rand(100, N, 3) * box_size

    # Compute clash rates
    cr_model = clash_rate(generated.cpu(), radius)
    cr_baseline = clash_rate(baseline, radius)

    print(f"\nResults:")
    print(f"  Model clash rate:    {cr_model:.3f}")
    print(f"  Baseline clash rate: {cr_baseline:.3f}")
    print(f"  Improvement:         {cr_baseline - cr_model:.3f}")

    if cr_model < cr_baseline:
        print("\nPASSED: Model beats random baseline!")
        return 0
    else:
        print("\nFAILED: Model does not beat random baseline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
