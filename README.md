# LucidViT 🧠

> A clean, professional, and improved implementation of **Vision Transformers (ViT)** built from scratch in PyTorch. Designed for transparency, research experimentation, and high-performance training.

## 📌 Overview
**LucidViT** (formerly *ViT Scratch*) is a minimalist yet powerful implementation of the Vision Transformer architecture. Unlike standard implementations, this project integrates modern architectural enhancements—such as **SwiGLU** activations and **Rotary Positional Embeddings (RoPE)**—to improve training stability and performance.

This repository serves as a clear reference for understanding how state-of-the-art attention mechanisms and transformer components are built and trained on vision datasets like **Tiny ImageNet**.

## ✨ Key Features
*   **From Principles**: Built entirely from scratch using PyTorch `nn.Module`.
*   **Modern Architecture**:
    *   **SwiGLU Activations**: Replaces standard GELU for better convergence (`SwiGLU` class).
    *   **RoPE (Rotary Positional Embeddings)**: Implements 2D rotary embeddings for better spatial awareness (`RoPEmbedding` class).
    *   **Grouped Query Attention (GQA)**: Configurable heads for efficient attention computation.
    *   **Pre-Norm & ResNets**: Stable training dynamics with LayerNorm and Residual connections.
*   **Modular Design**: Clean separation of `Encoder`, `SelfAttention`, and `FeedForwardLayer` components.
*   **Training Pipeline**: Complete training loop with `AdamW` optimizer, `CrossEntropyLoss` (w/ label smoothing), and real-time loss tracking.

## 📊 Model Statistics
| Metric | Value |
| :--- | :--- |
| **Total Parameters** | **~26.7 Million** |
| **Embedding Dimension** | 768 |
| **Patch Size** | 16x16 |
| **Sequence Length** | 65 (64 patches + 1 CLS token) |
| **Attention Heads** | 6 (Query), 3 (Key/Value) |
| **Layers** | 4 |
| **Activation** | SwiGLU |

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Torchvision
*   Datasets (Hugging Face)
*   Matplotlib (for plotting)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VidyasagarDudekula/nanoViT
    cd nanoViT
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision datasets matplotlib
    ```

## 📂 Project Structure
```
LucidViT/
├── config.py              # Configuration dataclass (ModelArgs)
├── model.py               # Core Vision Transformer implementation (RoPE, SwiGLU, Attention)
├── load_image_dataset.py  # Data loading logic (Tiny ImageNet via Hugging Face)
├── train.py               # Training loop, evaluation, and plotting
└── bpe_tokenizer.py       # (Placeholder/Extra) Tokenizer utilities
```

## 🛠️ Usage

### 1. Configure the Model
Adjust hyperparameters in `config.py` to suit your hardware or research needs:
```python
@dataclass
class ModelArgs:
    dim = 768
    n_q_head = 6    # Number of Query Heads
    n_encoder = 4   # Number of Layers
    num_classes = 200 # Tiny ImageNet classes
```

### 2. Train the Model
Run the training script to start training on the **Tiny ImageNet** dataset. The script automatically handles downloading and preprocessing the data.
```bash
python train.py
```
*   **Checkpoints**: Saved to `checkpoints/` (or current dir based on config).
*   **Visualization**: A training loss plot (`training_loss_plot.png`) is generated at the end of training.

## 📚 Dataset
The project is configured to use **Tiny ImageNet** (`Maysee/tiny-imagenet`).
*   **Classes**: 200
*   **Input Resolution**: Resized to 128x128
*   **Preprocessing**: Normalization using ImageNet mean/std conventions.


