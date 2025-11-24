# Use a PyTorch image with CUDA support for faster training and better compatibility
# PyTorch 2.1.0 with CUDA 12.1 is fully compatible with NVIDIA A10G (Ampere)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set architecture list for A10G (Ampere, Compute Capability 8.6)
# This ensures that if any CUDA extensions are built, they target the correct architecture
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg for imageio/visualization, git for pip)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .

# 1. Remove torch and torchvision from requirements.txt to prevent pip from upgrading them
#    and replacing the optimized base image version with a generic wheel.
# 2. Install the rest of the requirements.
# 3. Explicitly ensure compatible torchvision is installed (0.16.0 matches torch 2.1.0).
RUN sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchvision==0.16.0

# Configure Accelerate (default to fp16 for speed)
RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

# Copy all project files into the container
COPY . .

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--energy_head_enabled", "--loss_type", "energy_contrastive", "--push_to_hub", "--hub_model_id", "Uday/ctm-energy-based-halting"]
