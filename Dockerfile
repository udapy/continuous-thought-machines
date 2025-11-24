# Use a PyTorch image with CUDA support for faster training and better compatibility
# PyTorch 2.1.0 with CUDA 12.1 is fully compatible with NVIDIA A10G (Ampere)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set architecture list for A10G (Ampere, Compute Capability 8.6)
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install system dependencies (ffmpeg for imageio/visualization, git for pip)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to match HF Spaces default (user 1000)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .

# 1. Remove torch and torchvision from requirements.txt to prevent pip from upgrading them
# 2. Install the rest of the requirements.
# 3. Explicitly ensure compatible torchvision is installed (0.16.0 matches torch 2.1.0).
RUN sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchvision==0.16.0

# Copy all project files into the container
COPY . .

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set up environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# Create cache directories with correct permissions
RUN mkdir -p /tmp/matplotlib /tmp/numba_cache && \
    chmod 777 /tmp/matplotlib /tmp/numba_cache && \
    chown -R user:user /app

# Switch to the non-root user
USER user

# Configure Accelerate for the user (default to fp16 for speed)
# This writes to ~/.cache/huggingface/accelerate/default_config.yaml
RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--energy_head_enabled", "--loss_type", "energy_contrastive", "--push_to_hub", "--hub_model_id", "Uday/ctm-energy-based-halting"]
