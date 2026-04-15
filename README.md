# Amyloid Acoustic AI — Code Repository

This repository accompanies the manuscript describing a machine learning pipeline for analysing acoustic microscopy images of amyloid samples. The pipeline contains three modular components:

| Module | Script | Purpose |
|---|---|---|
| **Classification** | `Classification/main_unified_experiments.py` | Classify needle images by amyloid concentration (2–8 wt%) at 35 kHz |
| **Generation** | `Generation/main_diffusion_generator_...py` | Conditional diffusion model to synthesise microscopy images at unseen concentrations |
| **Prediction** | `Prediction/main_unified_prediction.py` | Regress amyloid concentration from needle images |
| **UMAP** | `UMAP/umap.py` | Frozen DINOv2 ViT-B/14 embedding, UMAP visualisation, LOCO ridge regression, and in-distribution classification |

---

## 1. System Requirements

### Operating System
- **Tested on**: Ubuntu 22.04 LTS (Linux x86-64)
- macOS 13+ and Windows 10/11 are expected to work with the same package versions but have not been formally tested.

### Software Dependencies

| Package | Tested version | Purpose |
|---|---|---|
| Python | 3.13.5 | Runtime (≥ 3.9 supported) |
| PyTorch | 2.9.1 | Deep learning framework |
| torchvision | 0.24.1 | Image transforms and model zoo |
| diffusers | 0.35.2 | Conditional diffusion model |
| accelerate | 1.11.0 | Distributed / mixed-precision training |
| numpy | 2.3.5 | Numerical operations |
| Pillow | 12.0.0 | Image I/O |
| tqdm | 4.67.1 | Progress bars |
| scikit-learn | 1.8.0 | Confusion matrix, classification, ridge regression |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical visualisation |
| pandas | ≥ 2.0 | CSV metadata handling (UMAP module) |
| umap-learn | ≥ 0.5 | UMAP dimensionality reduction (UMAP module) |

### Hardware
- **CPU-only**: All four modules run on CPU; inference scripts complete within minutes.
- **GPU (recommended)**: NVIDIA GPU with CUDA 12.x for training the classification, prediction, and generation models. Training times cited below assume an NVIDIA GPU.
- **Internet access (first UMAP run only)**: The UMAP module downloads the frozen DINOv2 ViT-B/14 weights (~330 MB) from `torch.hub` (`facebookresearch/dinov2`) on the first run. Use `--torch-hub-dir` to cache the weights to a custom directory.
- No other non-standard hardware is required.

---

## 2. Installation Guide

### Step 1 — Install Miniconda (if not already available)
```bash
# Linux
bash Installers\&proxy/Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts, then restart your shell or run: source ~/.bashrc
```

### Step 2 — Create and activate a conda environment
```bash
conda create -n amyloid_ai python=3.13 -y
conda activate amyloid_ai
```

### Step 3 — Install PyTorch (with CUDA 12.x)
Visit https://pytorch.org/get-started/locally/ to select the command matching your system. For CUDA 12.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
For CPU-only:
```bash
pip install torch torchvision
```

### Step 4 — Install remaining dependencies
```bash
pip install diffusers accelerate numpy "Pillow>=10" tqdm scikit-learn matplotlib seaborn pandas "umap-learn>=0.5"
```

### Typical installation time
| Environment | Approximate time |
|---|---|
| Fresh conda env, fast internet | 5–10 minutes |
| CPU-only PyTorch build | 3–5 minutes |

---

## 3. Demo

The demo uses **pre-trained regression models** and a **real measured test set** of needle acoustic microscopy images captured at 35 kHz across seven amyloid concentration levels (2–8 wt%). No training is required; the script runs inference only and prints quantitative metrics.

### Demo dataset

```
Data/01 needle/prediction/concentration_at_frequency_35kHz/
└── test/
    ├── 2/   (samples at 2 wt%)
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    ├── 7/
    └── 8/   (samples at 8 wt%)
```

Total: **826 images** spread across 7 concentration classes.

### Pre-trained models required

The following model files must be present in `save_models/`:

```
save_models/
├── best_cnn_regression_prediction_concentration_2.0.pth
├── best_cnn_regression_prediction_concentration_3.0.pth
├── best_cnn_regression_prediction_concentration_4.0.pth
├── best_cnn_regression_prediction_concentration_5.0.pth
├── best_cnn_regression_prediction_concentration_6.0.pth
├── best_cnn_regression_prediction_concentration_7.0.pth
└── best_cnn_regression_prediction_concentration_8.0.pth
```

These `.pth` files are included in the repository alongside this README.

### Step-by-step instructions

**1. Activate the conda environment**
```bash
conda activate amyloid_ai
```

**2. Navigate to the Prediction folder**
```bash
cd "Code for C4RUDE/Prediction"
```

**3. Run the demo (train + evaluate)**
```bash
python main_unified_prediction.py --experiment needle_concentration
```

> **Note:** No path configuration is needed. All data paths are resolved automatically relative to the repository root. The trained model is saved to `results/Prediction/needle_concentration/`.

### Expected output

The script prints training progress (epoch loss and validation MSE), then evaluates on the test set:

```
=================================================================
Experiment : Needle Sensor - Concentration Regression (Train + Evaluate)
Device     : cuda  (or "cpu" if no GPU is detected)
=================================================================
Checking for duplicate filenames across splits...
Known values   : [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Epoch 1/50:  loss=<value>   val_mse=<value>
...
Epoch 50/50: loss=<value>   val_mse=<value>
Best model loaded for final evaluation.

--- Test Evaluation: Known Values ---
  MSE: <value>   RMSE: <value>   MAE: <value>

Experiment 'needle_concentration' completed.
```

The trained model is saved to `results/Prediction/needle_concentration/best_cnn_regression_concentration.pth`.

### Expected run time (demo)

| Hardware | Approximate time |
|---|---|
| NVIDIA GPU (any modern card) | 10–20 minutes |
| CPU only (8-core desktop) | 30–60 minutes |

---

## 4. Instructions for Use

### Running all experiments

All scripts resolve their data paths automatically from their own location — no manual path editing is required. Simply run the scripts from within their respective folders as shown below.

---

### 4.1 Classification (`Classification/main_unified_experiments.py`)

**Purpose:** Train a CNN classifier to distinguish needle images across 7 amyloid concentration classes (2–8 wt%) measured at 35 kHz.

**Experiment:** `needle_concentration`

**Run:**
```bash
cd "Code for C4RUDE/Classification"
python main_unified_experiments.py --experiment needle_concentration --model simplecnn
```

Select the model with `--model`. Supported choices: `simplecnn`, `deepercnn`, `resnet18`, `resnet34`, `resnet50`, `densenet121`, `mobilenet_v2`, `efficientnet_b0`, `vgg16`.

**Typical training time (SimpleCNN, GPU):** 10–20 minutes for 50 epochs.

---

### 4.2 Image Generation (`Generation/main_diffusion_generator_...py`)

**Purpose:** Train a conditional denoising diffusion probabilistic model (DDPM) conditioned on amyloid concentration to generate synthetic acoustic microscopy images for a held-out (unseen) concentration level.

**Key configuration** (at the top of the script, inside `TrainingConfig`):

| Parameter | Description |
|---|---|
| `test_concentration` | The concentration to hold out and generate (e.g., `3.0`) |
| `num_epochs` | Number of training epochs (default: 200) |
| `guidance_scale` | Classifier-free guidance strength (default: 7.5) |

Data paths (`train_root`, `val_root`) and `output_dir` are resolved automatically from the repository root.

**Run:**
```bash
cd "Code for C4RUDE/Generation"
python main_diffusion_generator_generateallconcentrations_with_comparation_noiseseed_withCFG.py
```

**Typical training time:** 2–6 hours on a single NVIDIA GPU for 200 epochs on the full concentration dataset.

**Output:** Generated images saved to `output_dir/holdout_<concentration>/`, one subdirectory per epoch checkpoint. Quantitative comparison plots between real and generated images are also produced.

---

### 4.3 Prediction / Regression (`Prediction/main_unified_prediction.py`)

**Purpose:** Train a CNN to regress a continuous scalar value (amyloid concentration) directly from an image, then evaluate it on a held-out test set.

**Experiment:** `needle_concentration`

**Run:**
```bash
cd "Code for C4RUDE/Prediction"
python main_unified_prediction.py --experiment needle_concentration
```

The script trains for 50 epochs with early stopping (patience 10), saves the best checkpoint to `results/Prediction/needle_concentration/`, then evaluates on the test set reporting MSE, RMSE, and MAE.

**Typical run time:** 10–20 minutes on GPU, 30–60 minutes on CPU.

**Extending to frequency or chip experiments:**  
The `CNN_regression` model and training loop support any continuous scalar target. To add a frequency or chip experiment, add an entry to the `EXPERIMENTS` dictionary with `"mode": "train_eval"`, provide `root_train`/`root_val`/`root_test` paths, and set `known_values`/`unknown_values` to the label values you wish to train and test on. The script handles all preprocessing, training, early stopping, and evaluation automatically.

---

### 4.4 UMAP / Embedding Analysis (`UMAP/umap.py`)

**Purpose:** Extract frozen DINOv2 ViT-B/14 CLS embeddings (768-dim, L2-normalised) from all acoustic microscopy images, then perform:
- **UMAP visualisation** coloured by concentration class and by data split
- **Cosine distance matrix** between concentration tiers
- **In-distribution multinomial logistic regression** on the embeddings
- **LOCO (Leave-One-Concentration-Out) ridge regression** to predict unseen concentration tiers from the embedding space

The script is driven by a **CSV metadata file** with one row per image. The required columns are:

| Column | Description |
|---|---|
| `path` | Absolute path to the image file |
| `split` | Data split label (`train`, `val`, or `test`) |
| `target` | Concentration label (e.g. `2`, `3.0`, `"6 wt%"`) |
| `image_id` *(optional)* | Image identifier |
| `trial` *(optional)* | Trial / repeat identifier |

Column names can be customised via CLI flags (see below).

**Prepare the metadata CSV** (example structure):
```
path,split,target
/your/repo/Data/01 needle/prediction/concentration_at_frequency_35kHz/train/2/img001.png,train,2
/your/repo/Data/01 needle/prediction/concentration_at_frequency_35kHz/val/2/img002.png,val,2
/your/repo/Data/01 needle/prediction/concentration_at_frequency_35kHz/test/2/img003.png,test,2
...
```

**Run:**
```bash
conda activate amyloid_ai
cd "Code for C4RUDE/UMAP"

python umap.py \
    --csv /path/to/metadata.csv \
    --path-col path \
    --split-col split \
    --target-col target \
    --mask-type circle \
    --mask-value 0.25 \
    --outdir results_dinov2
```

**Key CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to the metadata CSV |
| `--outdir` | `results_dinov2` | Output directory for all results |
| `--path-col` | `path` | CSV column containing image paths |
| `--split-col` | `split` | CSV column containing split labels |
| `--target-col` | `target` | CSV column containing concentration labels |
| `--mask-type` | `none` | Preprocessing mask: `none`, `circle`, or `square` |
| `--mask-value` | `0.25` | Radius/side ratio for the mask |
| `--batch-size` | `16` | Batch size for embedding extraction |
| `--ridge-alpha` | `1.0` | Regularisation strength for LOCO ridge regression |
| `--torch-hub-dir` | *(auto)* | Directory to cache DINOv2 weights |

**Output files produced in `--outdir`:**

| File | Description |
|---|---|
| `umap_by_target.png` | UMAP scatter plot coloured by concentration |
| `umap_by_split.png` | UMAP scatter plot coloured by split |
| `median_cosine_distance_matrix.csv` | Pairwise median cosine distances between tiers |
| `embeddings.npy` | Raw 768-dim embeddings array |
| `classification/classification_report.csv` | Per-class precision, recall, F1 from logistic regression |
| `classification/classification_confusion_matrix.png` | Confusion matrix plot |
| `loco_ridge/loco_ridge_metrics.csv` | MAE, RMSE, R² per held-out concentration |
| `loco_ridge/loco_ridge_predictions.csv` | Individual predictions from LOCO regression |
| `loco_ridge/loco_ridge_summary.png` | Predicted mean ± std vs true tier plot |
| `run_summary.json` | Top-level summary metrics (accuracy, mean MAE, mean RMSE) |

**Expected output (run_summary.json):**
```json
{
  "n_total_samples": 3956,
  "embedding_dim": 768,
  "classification_test_accuracy": 0.95,
  "loco_n_tiers_evaluated": 7,
  "mean_loco_mae": 0.42,
  "mean_loco_rmse": 0.55
}
```
*(Actual numbers will vary with your dataset size and concentration spread.)*

**Typical run time:**

| Hardware | Approximate time |
|---|---|
| NVIDIA GPU | 5–10 minutes (embedding extraction + analyses) |
| CPU only (8-core desktop) | 20–40 minutes |

---

### 4.5 Reproducing Paper Results

The complete reproduction workflow follows these three stages in order:

```
Stage 1: Classification
  → Run main_unified_experiments.py --experiment needle_concentration
    to reproduce concentration classification accuracy.

Stage 2: Generation
  → Run the diffusion generator script with test_concentration set to each
    held-out concentration in turn (2.0 through 8.0) to reproduce the
    synthetic image generation results.

Stage 3: Prediction
  → Run main_unified_prediction.py --experiment needle_concentration
    to train and evaluate the concentration regression model.
    For frequency or chip data, add a train_eval entry to EXPERIMENTS
    and run with the corresponding dataset and label values.

Stage 4: UMAP / Embedding Analysis
  → Prepare a metadata CSV pointing to all images across train/val/test splits.
  → Run umap.py to reproduce UMAP visualisations, the cosine distance matrix,
    in-distribution classification accuracy, and LOCO ridge regression metrics.
```

**Random seed:** All scripts use a fixed random seed (2025 for prediction; 42 for generation) to ensure reproducibility. GPU-to-GPU variability may cause minor numerical differences; results should be within ±1% of reported values.

---

## File Structure

```
Code for C4RUDE/
├── README.md                          ← this file
├── Classification/
│   └── main_unified_experiments.py    ← classification pipeline
├── Generation/
│   └── main_diffusion_generator_...py ← conditional diffusion model
├── Prediction/
│   └── main_unified_prediction.py     ← regression pipeline
└── UMAP/
    └── umap.py                        ← DINOv2 embedding + UMAP + LOCO regression

Data/
└── 01 needle/
    └── concentration_at_frequency_35kHz/
        ├── train/                 ← 2 334 real training images (7 classes)
        ├── val/                   ← 796 real validation images
        └── test/                  ← 826 real test images  ← DEMO DATA


```

---

## Troubleshooting

**`FileNotFoundError` for data paths**  
Update `root_train`, `root_val`, and `root_test` in the `EXPERIMENTS` dictionary inside the script to match your local file system.

**`CUDA out of memory`**  
Reduce `batch_size` in the `EXPERIMENTS` dict (e.g. from 512 to 128).

**`torch.cuda.is_available()` returns `False`**  
The scripts fall back to CPU automatically. Verify your PyTorch installation matches your CUDA version by running:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**Slow training on CPU**  
Reduce `num_epochs` for a quick functionality check, or use a machine with a CUDA-capable GPU.

**`ImportError: Missing package 'umap-learn'`**  
Install it with:
```bash
pip install umap-learn
```

**DINOv2 download fails (UMAP module)**  
The first run of `umap.py` downloads ~330 MB of DINOv2 weights from GitHub. If you are behind a firewall or have a slow connection, download them manually and pass `--torch-hub-dir /path/to/cache`.

**`ValueError: Missing required columns in CSV`**  
Check that your metadata CSV has the columns specified by `--path-col`, `--split-col`, and `--target-col`, or pass the correct column names via those flags.
