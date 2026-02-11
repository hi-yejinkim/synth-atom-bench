#!/bin/bash
# Run the full scaling law experiment pipeline.
set -euo pipefail

DATA_DIR="outputs/data/N10_eta0.3"
SCALING_DIR="${1:-outputs/scaling}"

# Generate training data if not present
if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo "Generating training data..."
    mkdir -p "$DATA_DIR"
    uv run python data/generate.py --N 10 --eta 0.3 --num_samples 50000 --output "$DATA_DIR/train.npz"
    uv run python data/generate.py --N 10 --eta 0.3 --num_samples 5000 --seed 123 --output "$DATA_DIR/val.npz"
    echo "Data generation complete."
fi

# Step 1: Generate the experiment grid (measures FLOPs, prints commands)
echo "Generating scaling grid..."
uv run python experiments/scaling.py generate \
    --scaling_dir "$SCALING_DIR" \
    --wandb \
    > "$SCALING_DIR/grid.txt"
echo "Grid saved to $SCALING_DIR/grid.txt"
echo "$(wc -l < "$SCALING_DIR/grid.txt") runs to execute."

# Step 2: Run the grid
echo ""
echo "Running scaling experiments..."
uv run python experiments/scaling.py run \
    --scaling_dir "$SCALING_DIR" \
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
echo "         outputs/plots/isoflop_profiles.png"
