#!/bin/bash
# Run hyperparameter sweep using Hydra multirun with joblib launcher (8 GPUs).
set -euo pipefail

DATA_DIR="outputs/data/N10_eta0.3"
SWEEP_DIR="${1:-outputs/sweep}"
MAX_STEPS="${2:-100000}"

# Generate training data if not present
if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo "Generating training data..."
    mkdir -p "$DATA_DIR"
    uv run python data/generate.py --N 10 --eta 0.3 --num_samples 50000 --output "$DATA_DIR/train.npz"
    uv run python data/generate.py --N 10 --eta 0.3 --num_samples 5000 --seed 123 --output "$DATA_DIR/val.npz"
    echo "Data generation complete."
fi

echo "Running sweep (max_steps=$MAX_STEPS, 8 GPUs)..."
uv run python experiments/train.py --multirun \
    model=painn,transformer,pairformer \
    model.size=small,medium,large \
    train.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
    train.max_steps=$MAX_STEPS \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=8 \
    hydra.sweep.dir=$SWEEP_DIR \
    'hydra.sweep.subdir=${model.arch}_${model.size}_lr${train.lr}' \
    'checkpoint.dir='$SWEEP_DIR'/${model.arch}_${model.size}_lr${train.lr}' \
    logging.enabled=true

# Summarize results
echo ""
echo "Summarizing results..."
uv run python experiments/sweep_hparams.py summarize \
    --sweep_dir "$SWEEP_DIR"

echo ""
echo "Sweep complete. Results in $SWEEP_DIR/summary.json"
