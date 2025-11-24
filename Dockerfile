FROM ghcr.io/prefix-dev/pixi:0.39.0 AS builder

# Copy source code
COPY . /app
WORKDIR /app

# Install dependencies
RUN pixi install

# Create a shell script to run the training
# We need to activate the environment
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'pixi run python tasks/image_classification/train_energy.py "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Runtime image (optional, but good for size)
# For simplicity, we'll just use the builder image for now as it has everything.
# But HF Spaces might need specific permissions.

# Set up user for HF Spaces (optional but recommended)
# RUN useradd -m -u 1000 user
# USER user
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# ENTRYPOINT ["/app/entrypoint.sh"]
# CMD ["--help"]

# Let's try a simpler approach compatible with standard HF Spaces
# They often just run the CMD.

ENTRYPOINT ["pixi", "run", "python", "tasks/image_classification/train_energy.py"]
CMD ["--energy_head_enabled", "--loss_type", "energy_contrastive", "--push_to_hub", "--hub_model_id", "Uday/ctm-energy-based-halting", "--hub_token", "$HF_TOKEN"]
