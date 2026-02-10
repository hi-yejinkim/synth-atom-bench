#!/bin/bash
# Run hyperparameter sweep: generate data if needed, sweep, summarize.
set -euo pipefail

DATA_DIR="outputs/data/N10_eta0.3"
SWEEP_DIR="${1:-outputs/sweep}"
MAX_STEPS="${2:-10000}"

# Generate training data if not present
if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo "Generating training data..."
    mkdir -p "$DATA_DIR"
    uv run python data/generate.py --N 10 --eta 0.3 --num_samples 50000 --output "$DATA_DIR/train.npz"
    uv run python data/generate.py --N 10 --eta 0.3 --num_samples 5000 --seed 123 --output "$DATA_DIR/val.npz"
    echo "Data generation complete."
fi

# Run sweep
echo "Running hyperparameter sweep (max_steps=$MAX_STEPS)..."
uv run python experiments/sweep_hparams.py run \
    --sweep_dir "$SWEEP_DIR" \
    --max_steps "$MAX_STEPS"

# Summarize results
echo ""
echo "Summarizing results..."
uv run python experiments/sweep_hparams.py summarize \
    --sweep_dir "$SWEEP_DIR"

echo ""
echo "Sweep complete. Results in $SWEEP_DIR/summary.json"
