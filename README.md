# Image Classifier Project (PyTorch)

A production-style image classification pipeline built with **PyTorch** that trains a deep convolutional neural network on the **102-category Oxford Flowers dataset**, saves a reusable checkpoint, and performs top‑K inference from the command line. This project follows a modular, script-driven design suitable for experimentation, reproducibility, and extension.

---

## Project Overview

This repository implements an end-to-end **transfer learning** workflow:

1. **Data ingestion & augmentation** using `torchvision.datasets.ImageFolder`
2. **Model selection** (VGG16 or DenseNet121 backbones)
3. **Custom classifier head** with configurable hidden layers and dropout
4. **Training & validation loop** with GPU/CPU support
5. **Checkpoint serialization** (architecture + weights + class mapping)
6. **Inference pipeline** for single-image prediction with top‑K probabilities

The codebase is intentionally split into **train-time** and **predict-time** entry points, mirroring real-world ML workflows.

---

## Repository Structure

```text
.
├── train.py                 # CLI entry point for model training
├── predict.py               # CLI entry point for inference
├── helper.py                # Core ML utilities (data, model, training, inference)
├── workspace_utils.py       # Long-running session keep-alive (Udacity workspace)
├── cat_to_name.json         # Class index → human-readable flower names
├── Image Classifier Project.ipynb / .html
│                            # Exploratory & development notebook
├── checkpoint.pth           # (Generated) trained model checkpoint
└── LICENSE
```

---

## Full Tech Stack

### Language & Runtime

* **Python 3.x**
* Designed for Linux-based ML environments (Udacity / cloud VMs)

### Core ML Frameworks

* **PyTorch** (`torch`)

  * `torch.nn` – neural network layers & loss functions
  * `torch.optim` – optimizers (Adam)
  * `torch.utils.data` – DataLoader abstraction
* **Torchvision** (`torchvision`)

  * `models` – pretrained CNN backbones (VGG16, DenseNet121)
  * `datasets.ImageFolder` – directory-based dataset loading
  * `transforms` – image preprocessing & augmentation

### Model Architectures

* **VGG16** (default)

  * Feature extractor output: `25088` features
* **DenseNet121**

  * Feature extractor output: `1024` features

Backbone parameters are **frozen**; only the classifier head is trained.

### Scientific & Utility Libraries

* **NumPy** – tensor/array manipulation
* **Matplotlib** – visualization (training & debugging)
* **Pillow (PIL)** – image loading during inference
* **JSON** – class label mapping
* **argparse** – production-grade CLI interface

### Infrastructure / Ops

* **workspace_utils.active_session**

  * Sends periodic keep-alive requests to prevent idle shutdowns during long training runs

---

## Dataset Layout (Required)

The training script assumes the following directory structure:

```text
flowers/
├── train/
│   ├── 1/
│   ├── 2/
│   └── ...
├── valid/
│   ├── 1/
│   ├── 2/
│   └── ...
└── test/
    ├── 1/
    ├── 2/
    └── ...
```

Each subdirectory name corresponds to a **class index** that is later mapped to a human-readable label via `cat_to_name.json`.

---

## Data Pipeline Details

### Training Transforms

Applied **on-the-fly** for regularization:

* Random rotation (±30°)
* Random resized crop → `224×224`
* Random horizontal flip
* Normalization using ImageNet mean/std

### Validation / Test Transforms

Deterministic preprocessing:

* Resize shortest side to `256`
* Center crop → `224×224`
* ImageNet normalization

---

## Model Architecture (Classifier Head)

The pretrained backbone is replaced with a custom fully connected classifier:

```text
Input Features (arch dependent)
 → Linear(hidden_units_1)
 → ReLU
 → Dropout(p)
 → Linear(hidden_units_2)
 → ReLU
 → Dropout(p)
 → Linear(102)
 → LogSoftmax(dim=1)
```

### Defaults

* Hidden layer 1: `1024`
* Hidden layer 2: `512`
* Dropout: `0.5`
* Output classes: `102`

Loss Function:

* **Negative Log Likelihood Loss** (`nn.NLLLoss`)

Optimizer:

* **Adam** (classifier parameters only)

---

## Training Workflow (`train.py`)

### CLI Usage

```bash
python train.py \
  --data_dir flowers \
  --save_dir checkpoint.pth \
  --arch vgg16 \
  --learning_rate 0.001 \
  --hid_units1 1024 \
  --hid_units2 512 \
  --epochs 5 \
  --dropout 0.5 \
  --device cuda
```

### What Happens Under the Hood

1. Parse CLI arguments
2. Build dataloaders + class index mapping
3. Load pretrained backbone
4. Attach custom classifier
5. Train over `N` epochs with validation checks
6. Serialize checkpoint:

   * Model architecture
   * Classifier state dict
   * Optimizer parameters
   * Class-to-index mapping

---

## Checkpoint Format

The saved checkpoint is **architecture-aware** and reusable:

```python
{
  'arch': 'vgg16',
  'state_dict': model.state_dict(),
  'classifier': model.classifier,
  'class_to_idx': class_to_idx,
  'hyperparameters': {...}
}
```

This enables **exact reconstruction** of the trained model during inference.

---

## Inference Workflow (`predict.py`)

### CLI Usage

```bash
python predict.py \
  --image_file flowers/test/9/image_06413.jpg \
  --load_chkpt checkpoint.pth \
  --topk 5 \
  --class_name cat_to_name.json \
  --device cpu
```

### Inference Steps

1. Load checkpoint & rebuild model
2. Preprocess input image (resize, crop, normalize)
3. Forward pass through network
4. Extract **Top‑K probabilities & class indices**
5. Map indices → class labels → flower names

### Outputs

* Top‑K probabilities
* Corresponding class indices
* Human-readable flower names

---

## Device Support

* **CPU**: fully supported
* **GPU (CUDA)**: supported when available

The device is explicitly controlled via CLI to avoid silent mismatches.

---

## Development Notes

* The Jupyter notebook (`.ipynb`) was used for iterative development and visualization.
* Final logic was refactored into scripts for reproducibility and grading.
* Code follows a **research-to-production transition pattern** common in ML teams.

---

## License

This project is released under the terms specified in the `LICENSE` file.

---

## Future Improvements

* Add learning rate schedulers
* Support additional backbones (ResNet, EfficientNet)
* Add mixed-precision training
* Package as a Python module
* Add unit tests for helper utilities

---

If you want, I can also:

* Harden this into a production-ready repo
* Convert it into a FastAPI inference service
* Add experiment tracking (Weights & Biases / MLflow)
* Clean up the checkpoint schema
