# Datasets

The framework supports several standard image classification datasets with automatic configuration.

## Supported Datasets

| Dataset | Name | Classes | Image Size | Channels | Description |
|---------|------|---------|------------|----------|-------------|
| CIFAR-10 | `cifar10` | 10 | 32×32 | 3 (RGB) | General objects |
| CIFAR-100 | `cifar100` | 100 | 32×32 | 3 (RGB) | Fine-grained classification |
| MNIST | `mnist` | 10 | 28×28 | 1 (Gray) | Handwritten digits |
| FashionMNIST | `fashion_mnist` | 10 | 28×28 | 1 (Gray) | Fashion articles |

## Configuration

### Basic Usage

```yaml
dataset:
  name: "cifar10"
  root: "./data"
  download: true
```

### Full Configuration

```yaml
dataset:
  name: "cifar10"        # Dataset name
  root: "./data"         # Download/storage directory
  download: true         # Download if not present
  num_workers: 4         # DataLoader worker processes
  val_split: 0.1         # Validation split ratio (optional)
  seed: 42               # Seed for reproducible splits
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | **required** | Dataset identifier |
| `root` | string | `"./data"` | Directory for dataset storage |
| `download` | bool | `true` | Automatically download if missing |
| `num_workers` | int | `4` | Parallel data loading workers |
| `val_split` | float | `null` | Fraction for validation (0.0-1.0) |
| `seed` | int | `42` | Random seed for validation split |

## CIFAR-10

10 classes of common objects with 50,000 training and 10,000 test images.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

```yaml
dataset:
  name: "cifar10"
  root: "./data"
```

**Data Augmentation (Training):**

- Random horizontal flip
- Random crop (32×32 with 4px padding)
- Normalization (mean/std per channel)

## CIFAR-100

100 classes grouped into 20 superclasses.

```yaml
dataset:
  name: "cifar100"
  root: "./data"
```

**Note:** More challenging than CIFAR-10 due to finer-grained classes.

## MNIST

Handwritten digit recognition (0-9).

```yaml
dataset:
  name: "mnist"
  root: "./data"
```

**Characteristics:**

- Grayscale images (1 channel)
- 60,000 training / 10,000 test images
- Simple task, good for testing

## FashionMNIST

Fashion article classification, drop-in replacement for MNIST.

**Classes:** T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

```yaml
dataset:
  name: "fashion_mnist"
  root: "./data"
```

## Validation Split

Split training data into train/validation sets:

```yaml
dataset:
  name: "cifar10"
  val_split: 0.1    # 10% for validation
  seed: 42          # Reproducible split
```

**Behavior:**

- Training data split into train (90%) and validation (10%)
- Validation uses test transforms (no augmentation)
- Model evaluated on validation during training
- Final evaluation on test set after training

## Auto-Detection

Model parameters are automatically detected from the dataset:

```yaml
dataset:
  name: "mnist"  # 10 classes, 1 channel

model:
  name: "resnet18"
  # num_classes: 10    (auto-detected)
  # in_channels: 1     (auto-detected)
```

## Data Augmentation

Default augmentation strategies per dataset:

### RGB Datasets (CIFAR-10/100)

**Training:**
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

**Test:**
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

### Grayscale Datasets (MNIST, FashionMNIST)

**Training:**
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

## Next Steps

- [Configuration](../configuration/dataset.md) - Full configuration reference
