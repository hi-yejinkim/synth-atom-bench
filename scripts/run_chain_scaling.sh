#!/bin/bash
# Run the full scaling law experiment pipeline for chain datasets.
# Usage: bash scripts/run_chain_scaling.sh [DATA_CONFIG] [SCALING_DIR]
# Example: bash scripts/run_chain_scaling.sh chain_N10 outputs/scaling_chain
set -euo pipefail

DATA_CONFIG="${1:-chain_N10}"
SCALING_DIR="${2:-outputs/scaling_chain}"

# Infer N from config name (chain_N10 -> 10, chain_N20 -> 20, etc.)
N=$(echo "$DATA_CONFIG" | grep -oP 'N\K[0-9]+')
DATA_DIR="outputs/data/$DATA_CONFIG"

# Generate chain data if not present
if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo "Generating chain data (N=$N)..."
    mkdir -p "$DATA_DIR"
    uv run python data/generate_chains.py --N "$N" --num_samples 50000 --output "$DATA_DIR/train.npz"
    uv run python data/generate_chains.py --N "$N" --num_samples 5000 --seed 123 --output "$DATA_DIR/val.npz"
    uv run python data/generate_chains.py --N "$N" --num_samples 10000 --seed 456 --output "$DATA_DIR/test.npz"
    echo "Data generation complete."
fi

# Step 1: Generate the experiment grid
echo "Generating scaling grid..."
mkdir -p "$SCALING_DIR"
uv run python experiments/scaling.py generate \
    --scaling_dir "$SCALING_DIR" \
    --data "$DATA_CONFIG" \
    --wandb \
    > "$SCALING_DIR/grid.txt"
echo "Grid saved to $SCALING_DIR/grid.txt"
echo "$(wc -l < "$SCALING_DIR/grid.txt") runs to execute."

# Step 2: Run the grid
echo ""
echo "Running scaling experiments..."
uv run python experiments/scaling.py run \
    --scaling_dir "$SCALING_DIR" \
    --data "$DATA_CONFIG" \
    --wandb

# Step 3: Collect results
echo ""
echo "Collecting results..."
uv run python experiments/scaling.py collect \
    --scaling_dir "$SCALING_DIR"

# Step 4: Fit scaling laws and generate plots
echo ""
echo "Fitting scaling laws..."
uv run python experiments/scaling.py fit \
    --scaling_dir "$SCALING_DIR"

echo ""
echo "Scaling experiment complete."
echo "Results: $SCALING_DIR/results.json"
echo "Fits:    $SCALING_DIR/scaling_fits.json"
echo "Plots:   outputs/plots/scaling_curves.png"
echo "         outputs/plots/scaling_curves_bond_violation.png"
echo "         outputs/plots/scaling_curves_nonbonded_clash.png"
echo "         outputs/plots/isoflop_profiles.png"
