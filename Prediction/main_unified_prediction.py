"""
Unified Prediction (Regression) Pipeline for Amyloid Acoustic AI

Experiments:
  needle_concentration -- Train a CNN to regress amyloid concentration from
                          needle images captured at 35 kHz, then evaluate on
                          the held-out test set.

Usage:
  # Train and evaluate the concentration experiment
  python main_unified_prediction.py --experiment needle_concentration
"""

import argparse
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


# ===========================================================================
# Global Constants
# ===========================================================================

# Repository root is resolved automatically from this script's location.
# Structure: <repo_root>/Code for C4RUDE/Prediction/main_unified_prediction.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_RANDOM_SEED = 2025

EXPERIMENTS: Dict[str, dict] = {
    "needle_concentration": {
        "description": "Needle Sensor - Concentration Regression (Train + Evaluate)",
        "mode": "train_eval",
        "root_train": os.path.join(REPO_ROOT, "Data", "01 needle", "prediction", "concentration_at_frequency_35kHz", "train"),
        "root_val":   os.path.join(REPO_ROOT, "Data", "01 needle", "prediction", "concentration_at_frequency_35kHz", "val"),
        "root_test":  os.path.join(REPO_ROOT, "Data", "01 needle", "prediction", "concentration_at_frequency_35kHz", "test"),
        "known_values":   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "unknown_values": [],
        "mask_type":  "circle",
        "mask_ratio": 0.25,
        "batch_size": 64,
        "num_epochs": 50,
        "lr": 1e-3,
        "save_name": "best_cnn_regression_concentration.pth",
    },
}

OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "Prediction")


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ===========================================================================
# Custom Image Transforms
# ===========================================================================

class CenterCircleMask:
    """Apply a circular mask centred on the image, zeroing out all pixels outside."""

    def __init__(self, radius_ratio: float = 0.35) -> None:
        if not (0 < radius_ratio < 1):
            raise ValueError("radius_ratio must be between 0 and 1")
        self.radius_ratio = radius_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        r = int(min(w, h) * self.radius_ratio)
        cx, cy = w // 2, h // 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        img = img.copy()
        img.putalpha(mask)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
        return img.convert("RGB")


class CenterSquareMask:
    """Apply a square mask centred on the image, zeroing out all pixels outside."""

    def __init__(self, side_ratio: float = 0.4) -> None:
        if not (0 < side_ratio < 1):
            raise ValueError("side_ratio must be between 0 and 1")
        self.side_ratio = side_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size
        side = int(min(w, h) * self.side_ratio)
        cx, cy = w // 2, h // 2
        left, right = cx - side // 2, cx + side // 2
        top, bottom = cy - side // 2, cy + side // 2
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, right, bottom], fill=255)
        img = img.copy()
        img.putalpha(mask)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
        return img.convert("RGB")


def get_transforms(mask_type: str, mask_ratio: float) -> transforms.Compose:
    """Build image preprocessing pipeline. Normalization uses (0.5, 0.5, 0.5) to
    match the custom CNN_regression training convention (non-pretrained network)."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    mask = CenterCircleMask(radius_ratio=mask_ratio) if mask_type == "circle" \
        else CenterSquareMask(side_ratio=mask_ratio)
    return transforms.Compose([
        mask,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])


# ===========================================================================
# Dataset
# ===========================================================================

def _extract_scalar_label(folder_name: str) -> float:
    """Parse a floating-point label from a folder name using regex.

    Handles patterns such as:
      - "frequency_35kHz"  -> 35.0
      - "conc_5.5mg_ml"    -> 5.5
      - "2.0"              -> 2.0
    Raises ValueError if no number is found.
    """
    cleaned = folder_name.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"Cannot parse numeric label from folder name: '{folder_name}'")
    return float(match.group(0))


class ScalarDataset(Dataset):
    """Generic scalar-regression image dataset.

    Convention: each sub-folder of ``root_dir`` represents one label value.
    The label is extracted from the folder name via regex (see _extract_scalar_label).
    All PNG/JPG/JPEG/BMP files inside a folder are treated as samples for that label.
    """

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None) -> None:
        self.image_paths: List[str] = []
        self.labels: List[float] = []
        self.transform = transform

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            try:
                label_value = _extract_scalar_label(folder_name)
            except ValueError:
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.image_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(label_value)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ===========================================================================
# Model
# ===========================================================================

class CNN_regression(nn.Module):
    """Custom 6-layer CNN for continuous-value regression.

    Architecture:
      6 convolutional blocks (Conv2d -> ReLU -> MaxPool2d)
      followed by two fully-connected layers that output a single scalar.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 224 -> 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56  -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28  -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 7  -> 3
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(512 * 3 * 3, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.regressor(x)


# ===========================================================================
# Data Utilities
# ===========================================================================

def get_all_files(root: str) -> Iterable[str]:
    for rootdir, _, files in os.walk(root):
        for fname in files:
            yield os.path.join(rootdir, fname)


def check_overlap(train_root: str, val_root: str, test_root: str) -> None:
    """Warn if any filename appears in more than one data split."""
    print("Checking for duplicate filenames across splits...")
    train_files = set(get_all_files(train_root))
    val_files   = set(get_all_files(val_root))
    test_files  = set(get_all_files(test_root))
    for label, s1, s2 in [("Train-Val", train_files, val_files),
                           ("Train-Test", train_files, test_files),
                           ("Val-Test",   val_files, test_files)]:
        overlap = len(s1 & s2)
        status = "WARNING: data leakage detected!" if overlap > 0 else "OK"
        print(f"  {label} Overlap: {overlap} ({status})")


def subset_by_value(dataset: ScalarDataset, allowed_values: Sequence[float]) -> Subset:
    allowed = set(allowed_values)
    indices = [i for i, v in enumerate(dataset.labels) if v in allowed]
    return Subset(dataset, indices)


def build_loader(
    subset: Subset,
    batch_size: int,
    shuffle: bool,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Optional[DataLoader]:
    if len(subset) == 0:
        return None

    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,           # set to 0 for cross-platform stability
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


# ===========================================================================
# Training
# ===========================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_path: str,
    patience: int = 10,
) -> None:
    """Training loop with early stopping and mixed-precision (AMP) support."""
    best_val_mse = float("inf")
    epochs_no_improve = 0

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_samples = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_samples += inputs.size(0)

        train_mse = running_loss / max(1, running_samples)
        train_rmse = math.sqrt(train_mse)
        print(f"Epoch {epoch + 1}/{num_epochs}  Train MSE: {train_mse:.4f}  RMSE: {train_rmse:.4f}")

        if val_loader is not None:
            result = evaluate_model(model, val_loader, criterion, device, name="Val")
            if result is not None:
                val_mse = result[0]
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    epochs_no_improve = 0
                    _save_state(model, save_path)
                    print(f"  --> Best model saved  (Val MSE: {best_val_mse:.4f})")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
        else:
            _save_state(model, save_path)

    # Ensure a model file always exists after training
    if not os.path.exists(save_path):
        _save_state(model, save_path)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_model(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    name: str = "Test",
    verbose: bool = True,
) -> Optional[Tuple[float, float, float, List[float], List[float]]]:
    """Return (MSE, RMSE, MAE, predictions, targets) or None if loader is empty."""
    if loader is None or len(loader.dataset) == 0:
        if verbose:
            print(f"[{name}] No samples available.")
        return None

    model.eval()
    total_loss, total_abs_err, total_samples = 0.0, 0.0, 0
    predictions: List[float] = []
    targets: List[float] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_abs_err += torch.abs(outputs - labels).sum().item()
            predictions.extend(outputs.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    mse  = total_loss    / total_samples
    rmse = math.sqrt(mse)
    mae  = total_abs_err / total_samples

    if verbose:
        print(f"[{name}]  MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    return mse, rmse, mae, predictions, targets


def print_per_value_stats(
    preds: List[float],
    targets: List[float],
    label: str = "Unseen",
    max_samples_to_print: int = 5,
) -> None:
    """Aggregate predictions by ground-truth value and print per-group statistics."""
    groups: defaultdict[float, List[float]] = defaultdict(list)
    for truth, pred in zip(targets, preds):
        groups[truth].append(pred)

    print(f"\n{label} Value Statistics:")
    for value in sorted(groups):
        group_preds = groups[value]
        n = len(group_preds)
        mae  = sum(abs(p - value) for p in group_preds) / n
        rmse = math.sqrt(sum((p - value) ** 2 for p in group_preds) / n)
        print(
            f"  Ground truth {value:.2f} | n={n:4d} | "
            f"Mean pred: {statistics.mean(group_preds):.2f} | "
            f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
        )

    print(f"\n{label} Sample Predictions (up to {max_samples_to_print} per value):")
    for value in sorted(groups):
        for pred in groups[value][:max_samples_to_print]:
            print(f"  GT {value:.2f} kHz  -->  Pred {pred:.2f} kHz")


# ===========================================================================
# Helpers
# ===========================================================================

def _save_state(model: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, save_path)


def _load_state(model: nn.Module, save_path: str, device: torch.device) -> None:
    state_dict = torch.load(save_path, map_location=device, weights_only=True)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


# ===========================================================================
# Experiment Runners
# ===========================================================================

def run_train_eval_experiment(exp_name: str) -> None:
    """Train and evaluate a regression experiment defined in EXPERIMENTS."""
    cfg = EXPERIMENTS[exp_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 65}")
    print(f"Experiment : {cfg['description']}")
    print(f"Device     : {device}")
    print(f"{'=' * 65}")

    # Output directory
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, cfg["save_name"])

    # Data integrity check
    check_overlap(cfg["root_train"], cfg["root_val"], cfg["root_test"])

    # Preprocessing pipeline
    tfm = get_transforms(cfg["mask_type"], cfg["mask_ratio"])

    # Build datasets
    print("Loading datasets...")
    ds_train = ScalarDataset(cfg["root_train"], transform=tfm)
    ds_val   = ScalarDataset(cfg["root_val"],   transform=tfm)
    ds_test  = ScalarDataset(cfg["root_test"],  transform=tfm)

    known_values   = list(cfg["known_values"])
    unknown_values = list(cfg["unknown_values"])

    print(f"Known values   : {known_values}")
    print(f"Unknown values : {unknown_values}")

    # Split subsets
    sub_train        = subset_by_value(ds_train, known_values)
    sub_val          = subset_by_value(ds_val,   known_values)
    sub_test_known   = subset_by_value(ds_test,  known_values)
    sub_test_unknown = subset_by_value(ds_test,  unknown_values)

    if len(sub_train) == 0:
        raise RuntimeError("Training subset is empty. Check data paths and known_values.")

    print(f"Samples  Train: {len(sub_train)}  Val: {len(sub_val)}  "
          f"Test-Known: {len(sub_test_known)}  Test-Unknown: {len(sub_test_unknown)}")

    # DataLoaders
    bs = cfg["batch_size"]
    train_loader   = build_loader(sub_train,        bs, shuffle=True)
    val_loader     = build_loader(sub_val,           bs, shuffle=False)
    known_loader   = build_loader(sub_test_known,    bs, shuffle=False)
    unknown_loader = build_loader(sub_test_unknown,  bs, shuffle=False)

    # Model
    model = CNN_regression()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer,
                device, cfg["num_epochs"], save_path)

    # Load best checkpoint
    if os.path.exists(save_path):
        _load_state(model, save_path, device)
        print("Best model loaded for final evaluation.")

    # Evaluate
    print("\n--- Test Evaluation: Known Values ---")
    evaluate_model(model, known_loader, criterion, device, name="Test-Known")

    print("\n--- Test Evaluation: Unknown Values ---")
    unknown_result = evaluate_model(model, unknown_loader, criterion, device, name="Test-Unknown")

    if unknown_result is not None:
        _, _, _, preds, targets = unknown_result
        print_per_value_stats(preds, targets, label="Unknown")

    print(f"\nExperiment '{exp_name}' completed. Results saved to: {exp_dir}")


def run_concentration_eval_experiment() -> None:
    """
    Inference-only experiment: for each concentration level present in the test
    set, load the corresponding pre-trained model and report regression metrics.
    """
    cfg = EXPERIMENTS["needle_concentration_eval"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 65}")
    print(f"Experiment : {cfg['description']}")
    print(f"Device     : {device}")
    print(f"{'=' * 65}")

    # Validate paths
    if not os.path.exists(cfg["root_test"]):
        raise FileNotFoundError(f"Test directory not found: {cfg['root_test']}")
    if not os.path.exists(cfg["model_dir"]):
        raise FileNotFoundError(f"Model directory not found: {cfg['model_dir']}")

    # Preprocessing
    tfm = get_transforms(cfg["mask_type"], cfg["mask_ratio"])

    # Load test dataset
    print(f"Loading test dataset: {cfg['root_test']}")
    ds_test = ScalarDataset(cfg["root_test"], transform=tfm)
    concentrations = sorted(set(ds_test.labels))

    if not concentrations:
        raise RuntimeError("No concentration labels could be parsed from the test dataset.")

    print(f"Detected concentrations: {concentrations}")

    criterion = nn.MSELoss()

    # Loop over each concentration level
    for conc in concentrations:
        print(f"\n{'─' * 50}")
        print(f"Concentration: {conc}")

        model_name = cfg["model_name_template"].format(value=conc)
        model_path = os.path.join(cfg["model_dir"], model_name)

        if not os.path.exists(model_path):
            print(f"  [SKIP] Model not found: {model_path}")
            continue

        # Load model
        model = CNN_regression()
        model = nn.DataParallel(model)
        model = model.to(device)

        try:
            _load_state(model, model_path, device)
        except Exception as exc:
            print(f"  [ERROR] Failed to load model: {exc}")
            continue

        # Evaluate
        subset = subset_by_value(ds_test, [conc])
        loader = build_loader(subset, cfg["batch_size"], shuffle=False)

        if loader is None:
            print("  No samples for this concentration.")
            continue

        print(f"  Samples: {len(subset)}")
        result = evaluate_model(model, loader, criterion, device, name=f"Conc {conc}")

        if result is not None:
            _, _, _, preds, targets = result
            print(f"  Sample predictions (first 20):")
            for i, (p, t) in enumerate(zip(preds[:20], targets[:20])):
                print(f"    [{i+1:02d}]  True: {t:.3f}   Pred: {p:.3f}   Error: {p - t:+.3f}")

    print("\nAll concentration evaluations completed.")


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Acoustic AI Prediction (Regression) Experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="needle_concentration",
        choices=["all"] + list(EXPERIMENTS.keys()),
        help="Which experiment to run (default: needle_concentration)",
    )
    args = parser.parse_args()

    set_random_seed(DEFAULT_RANDOM_SEED)

    start_time = time.time()

    if args.experiment == "all":
        for exp_name in EXPERIMENTS.keys():
            try:
                run_train_eval_experiment(exp_name)
            except Exception as exc:
                print(f"\n[ERROR] Experiment '{exp_name}' failed: {exc}")
    else:
        run_train_eval_experiment(args.experiment)

    print(f"\nTotal elapsed time: {time.time() - start_time:.1f} s")
