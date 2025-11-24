---
title: Continuous Thought Machine
emoji: 🕰️
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "20.10.21"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🕰️ The Continuous Thought Machine

📚 [PAPER: Technical Report](https://arxiv.org/abs/2505.05522) | 📝 [Blog](https://sakana.ai/ctm/) | 🕹️ [Interactive Website](https://pub.sakana.ai/ctm) | ✏️ [Tutorial](examples/01_mnist.ipynb)

## Overview

The **Continuous Thought Machine (CTM)** is a novel neural architecture designed to unfold and leverage neural activity as the underlying mechanism for observation and action. By introducing an internal temporal axis decoupled from input data, CTM enables neurons to process information over time with fine-grained temporal dynamics.

### Key Contributions

1. **Internal Temporal Axis**: Decoupled from input data, allowing neuron activity to unfold independently
2. **Neuron-Level Temporal Processing**: Each neuron uses unique weight parameters to process a history of incoming signals
3. **Neural Synchronisation**: Direct latent representation for modulating data and producing outputs, encoding information in the timing of neural activity

The CTM demonstrates strong performance across diverse tasks including ImageNet classification, 2D maze solving, sorting, parity computation, question-answering, and reinforcement learning.

---

## 🔬 Energy-Based Halting Experiment

This repository includes an implementation of **Energy-Based Halting**, a mechanism that frames "thinking" as an optimization process where the model dynamically adjusts its internal thought process duration based on sample difficulty.

### Concept

Instead of using heuristic certainty thresholds, we train a learned energy scalar that:

- **Minimizes energy** for correct predictions (pushing the system to low-energy equilibrium)
- **Maximizes energy** for incorrect predictions (pushing away from stable states)
- **Enables adaptive halting** based on energy thresholds or convergence

### Implementation

**Modified Components:**

- `models/ctm.py`: Added energy projection head that maps synchronization states to scalar energy values
- `utils/losses.py`: Implemented `EnergyContrastiveLoss` for training the energy function
- `tasks/image_classification/train_energy.py`: Training script with energy halting
- `inference_energy.py`: Adaptive inference that halts when energy drops below threshold or stabilizes
- `configs/energy_experiment.yaml`: Configuration for energy experiments

**Training:**

```bash
# Local training
pixi run accelerate launch tasks/image_classification/train_energy.py \
    --energy_head_enabled \
    --loss_type energy_contrastive \
    --dataset cifar10

# Or with traditional python
pixi run python tasks/image_classification/train_energy.py \
    --energy_head_enabled \
    --loss_type energy_contrastive
```

**Deployment to Hugging Face:**
See [GUIDE_HF.md](GUIDE_HF.md) for instructions on deploying the training job to Hugging Face Spaces with GPU support.

---

## 🚀 Quick Start

### Setup with Pixi (Recommended)

We use [Pixi](https://pixi.sh) for dependency management, which handles both Python packages and system dependencies like `ffmpeg`.

```bash
# Install dependencies
pixi install

# Run training
pixi run python tasks/image_classification/train.py
```

### Alternative: Conda Setup

```bash
conda create --name=ctm python=3.12
conda activate ctm
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

If there are PyTorch version issues:

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📁 Repository Structure

```
├── tasks/
│   ├── image_classification/
│   │   ├── train.py                    # Standard training
│   │   ├── train_energy.py             # Energy halting training
│   │   ├── analysis/run_imagenet_analysis.py
│   │   └── plotting.py
│   ├── mazes/
│   │   ├── train.py
│   │   └── analysis/
│   ├── sort/
│   ├── parity/
│   ├── qamnist/
│   └── rl/
├── models/
│   ├── ctm.py                          # Main CTM model (with energy head support)
│   ├── modules.py                      # Neuron-level models, Synapse UNET
│   ├── ff.py                           # Feed-forward baseline
│   └── lstm.py                         # LSTM baseline
├── utils/
│   ├── losses.py                       # Loss functions (includes EnergyContrastiveLoss)
│   ├── schedulers.py
│   └── housekeeping.py
├── data/
│   └── custom_datasets.py
├── configs/
│   └── energy_experiment.yaml          # Energy halting hyperparameters
├── inference_energy.py                 # Adaptive energy-based inference
├── Dockerfile                          # For HF Spaces deployment
├── GUIDE_HF.md                         # Hugging Face deployment guide
└── checkpoints/                        # Model checkpoints
```

---

## 🎯 Model Training

Each task has dedicated training code designed for ease-of-use and collaboration. Training scripts include reasonable defaults, with paper-replicating configurations in accompanying script folders.

### Image Classification Example

```bash
# Standard CTM training
python -m tasks.image_classification.train

# Energy halting training
python -m tasks.image_classification.train_energy \
    --energy_head_enabled \
    --loss_type energy_contrastive
```

### VSCode Debug Configuration

```json
{
  "name": "Debug: train image classifier",
  "type": "debugpy",
  "request": "launch",
  "module": "tasks.image_classification.train",
  "console": "integratedTerminal",
  "justMyCode": false
}
```

---

## 🔍 Analysis & Visualization

Analysis and plotting code to replicate paper figures is provided in `tasks/.../analysis/*`.

**Note:** `ffmpeg` is required for generating videos:

```bash
conda install -c conda-forge ffmpeg
# or with pixi (already included)
pixi install
```

---

## 📦 Checkpoints and Data

Download pre-trained checkpoints and datasets:

- **Checkpoints**: [Google Drive](https://drive.google.com/drive/folders/1vSg8T7FqP-guMDk1LU7_jZaQtXFP9sZg)
- **Maze Data**: [Google Drive](https://drive.google.com/file/d/1cBgqhaUUtsrll8-o2VY42hPpyBcfFv86/view?usp=drivesdk)

Place checkpoints in the `checkpoints/` folder following the structure `checkpoints/{task}/...`

---

## 🤗 Hugging Face Integration

This repository includes full support for training on Hugging Face infrastructure:

- **Accelerate**: Multi-GPU and mixed precision training
- **Hub Integration**: Automatic checkpoint uploading
- **Spaces Deployment**: Run training jobs on GPU Spaces

See [GUIDE_HF.md](GUIDE_HF.md) for detailed instructions.

---

## 📖 Interactive Resources

- **[Interactive Website](https://pub.sakana.ai/ctm)**: Maze-solving demo, videos, and visualizations
- **[Paper](https://arxiv.org/abs/2505.05522)**: Technical details and experiments
- **[Blog](https://sakana.ai/ctm/)**: High-level overview and insights
- **[Tutorial Notebook](examples/01_mnist.ipynb)**: Hands-on introduction

---

## 🙏 Citation

If you use this code or build upon CTM in your work, please cite:

```bibtex
@article{ctm2025,
  title={The Continuous Thought Machine},
  author={...},
  journal={arXiv preprint arXiv:2505.05522},
  year={2025}
}
```

---

## 📝 License

This project is released under the MIT License. See LICENSE file for details.
