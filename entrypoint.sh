#!/bin/bash
set -e

# Collect arguments passed to the container (CMD)
args=("$@")

# If HF_TOKEN is set, append it to the arguments
if [ -n "$HF_TOKEN" ]; then
    args+=("--hub_token" "$HF_TOKEN")
fi

# Run accelerate launch with the training script and arguments
exec accelerate launch tasks/image_classification/train_energy.py "${args[@]}"
