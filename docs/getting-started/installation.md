# Installation

## Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Install from Source

```bash
# Clone the repository
git clone https://github.com/pollo/fat.git
cd fat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Manual Installation

If you prefer to install dependencies manually:

=== "Core Dependencies"

    ```bash
    pip install torch torchvision pyyaml tqdm
    ```

=== "With TensorBoard"

    ```bash
    pip install torch torchvision pyyaml tqdm tensorboard
    ```

=== "With Quantization (Brevitas)"

    ```bash
    pip install torch torchvision pyyaml tqdm brevitas
    ```

=== "Full Installation"

    ```bash
    pip install torch torchvision pyyaml tqdm tensorboard brevitas
    ```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision {torchvision.__version__}')"
```

Check CUDA availability:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Optional Dependencies

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `tensorboard` | Training visualization | `pip install tensorboard` |
| `brevitas` | Quantization-aware training | `pip install brevitas` |
| `numpy` | Additional utilities | `pip install numpy` |

## Documentation Dependencies

To build the documentation locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to train your first model.
