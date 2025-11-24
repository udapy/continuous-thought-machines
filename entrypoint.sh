#!/bin/bash
set -e

# Collect arguments passed to the container (CMD)
args=("$@")

# Sanitize OMP_NUM_THREADS if it's not an integer (e.g. "3500m" from HF Spaces)
if ! [[ "$OMP_NUM_THREADS" =~ ^[0-9]+$ ]]; then
    echo "WARNING: OMP_NUM_THREADS is '$OMP_NUM_THREADS', which is not an integer. Resetting to 1."
    export OMP_NUM_THREADS=1
fi

# If HF_TOKEN is set, append it to the arguments
if [ -n "$HF_TOKEN" ]; then
    args+=("--hub_token" "$HF_TOKEN")
fi

# Run accelerate launch with the training script and arguments
exec accelerate launch tasks/image_classification/train_energy.py "${args[@]}"
