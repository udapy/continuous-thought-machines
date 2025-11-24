# Training on Hugging Face with GPUs

This guide explains how to train the Energy Halting experiment on Hugging Face infrastructure, including local GPU training with `accelerate` and deployment to Hugging Face Spaces.

## Prerequisites

1.  **Hugging Face Account**: Create one at [huggingface.co](https://huggingface.co).
2.  **Access Token**: Get a write token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3.  **Pixi**: Installed locally.

## 1. Local Training with Accelerate

We use Hugging Face `accelerate` for robust multi-GPU and mixed-precision training.

### Setup

Ensure dependencies are installed:

```bash
pixi install
```

### Configure Accelerate

Run the configuration wizard to set up your GPU environment (e.g., number of GPUs, mixed precision):

```bash
pixi run accelerate config
```

### Run Training

Use `accelerate launch` to start training. This handles device placement automatically.

```bash
pixi run accelerate launch tasks/image_classification/train_energy.py \
    --energy_head_enabled \
    --loss_type energy_contrastive \
    --dataset cifar10 \
    --batch_size 32 \
    --use_amp \
    --push_to_hub \
    --hub_model_id <your-username>/ctm-energy-cifar10 \
    --hub_token <your-token>
```

## 2. Deploying to Hugging Face Spaces (GPU)

You can run this training job on a Hugging Face Space with a GPU.

### Create a Space

1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  Name: `ctm-energy-training` (or similar).
3.  SDK: **Docker**.
4.  Hardware: Choose a **GPU** instance (e.g., Nvidia T4, A10G).

### Deploy Code

You can deploy by pushing your code to the Space's repository.

1.  **Clone the Space**:

    ```bash
    git clone https://huggingface.co/spaces/<your-username>/ctm-energy-training
    cd ctm-energy-training
    ```

2.  **Copy Files**:
    Copy your project files into this directory (excluding `.git`, `.pixi`, `data`, `logs`).
    _Crucially, ensure `Dockerfile`, `pixi.toml`, `pixi.lock`, `tasks/`, `models/`, `utils/`, and `configs/` are present._

3.  **Push**:
    ```bash
    git add .
    git commit -m "Deploy training job"
    git push
    ```

### Environment Variables

To allow the Space to push the trained model back to the Hub, you need to set your HF token as a secret.

1.  Go to your Space's **Settings**.
2.  Scroll to **Variables and secrets**.
3.  Add a New Secret:
    - Name: `HF_TOKEN`
    - Value: Your write token.

### Update Dockerfile CMD (Optional)

The default `Dockerfile` CMD prints help. To run training immediately upon deployment, modify the `CMD` in the `Dockerfile` before pushing:

```dockerfile
CMD ["--energy_head_enabled", "--loss_type", "energy_contrastive", "--push_to_hub", "--hub_model_id", "<your-username>/ctm-energy-cifar10", "--hub_token", "$HF_TOKEN"]
```

_Note: You'll need to pass the token via env var or arg._

## 3. Monitoring

- **Local**: Check the `logs/` directory or WandB if enabled (`--wandb`).
- **Spaces**: Check the **Logs** tab in your Space.
