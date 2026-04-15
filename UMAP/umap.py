#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frozen DINOv2 ViT-B/14 (768-dim) pipeline .

What it does
------------
1) Reads a CSV with at least:
   - absolute image path
   - split (train / val / test)
   - target label (e.g. concentration tier)
   - image id (optional)
   - trial (optional)

2) Extracts frozen DINOv2 CLS embeddings 
3) L2-normalizes embeddings
4) Saves embeddings + metadata
5) Runs:
   - UMAP colored by target
   - UMAP colored by split
   - cosine-distance matrix between target tiers
   - in-distribution multiclass logistic regression
   - LOCO Ridge regression

Example
-------
umap.py \
    --csv metadata.csv \
    --path-col path \
    --split-col split \
    --target-col concentration \
    --image-id-col image_id \
    --trial-col trial \
    --outdir results_dinov2

Dependencies
------------
pip install torch torchvision pandas numpy pillow scikit-learn umap-learn matplotlib tqdm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import umap
except ImportError as e:
    raise ImportError(
        "Missing package 'umap-learn'. Install it with: pip install umap-learn"
    ) from e


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_numeric_targets(series: pd.Series) -> np.ndarray:
    """
    Converts values like:
      6
      6.0
      "6 wt%"
      "6wt"
      "tier_6"
    into floats if possible.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).to_numpy()

    extracted = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    if extracted.isna().any():
        bad_examples = series[extracted.isna()].astype(str).unique().tolist()[:10]
        raise ValueError(
            "Could not parse some target values as numeric for regression. "
            f"Bad examples: {bad_examples}"
        )
    return extracted.astype(float).to_numpy()


def normalize_split_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def centered_circle_mask(img: Image.Image, radius_ratio: float) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    radius = int(radius_ratio * short)
    cx, cy = w // 2, h // 2

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)

    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(img, mask=mask)
    return out


def centered_square_mask(img: Image.Image, side_ratio: float) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    side = int(side_ratio * short)
    cx, cy = w // 2, h // 2
    half = side // 2

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=255)

    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(img, mask=mask)
    return out


def apply_mask(
    img: Image.Image,
    mask_type: str = "none",
    mask_value: float = 0.25,
) -> Image.Image:
    if mask_type == "none":
        return img
    if mask_type == "circle":
        return centered_circle_mask(img, radius_ratio=mask_value)
    if mask_type == "square":
        return centered_square_mask(img, side_ratio=mask_value)
    raise ValueError(f"Unknown mask_type: {mask_type}")


def build_transform(image_size: int = 224) -> transforms.Compose:
    """
    Official-ish DINOv2 eval-style preprocessing:
    Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet normalization
    """
    resize_size = int(round(image_size / 224 * 256))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# -----------------------------
# Dataset
# -----------------------------
class CSVDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        transform: transforms.Compose,
        mask_type: str = "none",
        mask_value: float = 0.25,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.transform = transform
        self.mask_type = mask_type
        self.mask_value = mask_value

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        path = self.df.iloc[idx][self.path_col]
        img = Image.open(path).convert("RGB")
        img = apply_mask(img, mask_type=self.mask_type, mask_value=self.mask_value)
        x = self.transform(img)
        return idx, x


# -----------------------------
# DINOv2 loading
# -----------------------------
def load_dinov2_vitb14(torch_hub_dir: Optional[str] = None) -> torch.nn.Module:
    if torch_hub_dir is not None:
        torch.hub.set_dir(torch_hub_dir)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    model.to("cpu")

    for p in model.parameters():
        p.requires_grad = False

    return model


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    df: pd.DataFrame,
    path_col: str,
    batch_size: int,
    mask_type: str,
    mask_value: float,
    num_workers: int,
    image_size: int,
) -> np.ndarray:
    transform = build_transform(image_size=image_size)
    ds = CSVDataset(
        df=df,
        path_col=path_col,
        transform=transform,
        mask_type=mask_type,
        mask_value=mask_value,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    embeddings = np.zeros((len(ds), 768), dtype=np.float32)

    for indices, batch in tqdm(loader, desc="Extracting DINOv2 embeddings", ncols=100):
        batch = batch.to("cpu", non_blocking=False)

        # Official repo exposes the CLS token here.
        features_dict = model.forward_features(batch)
        cls_tokens = features_dict["x_norm_clstoken"]  # shape: [B, 768]

        # Manuscript says L2-normalized features for downstream comparisons.
        cls_tokens = F.normalize(cls_tokens, p=2, dim=1)

        embeddings[indices.numpy()] = cls_tokens.cpu().numpy().astype(np.float32)

    return embeddings


# -----------------------------
# Analysis helpers
# -----------------------------
def save_embeddings(
    outdir: Path,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    np.save(outdir / "embeddings.npy", embeddings)
    df.to_csv(outdir / "metadata_with_order.csv", index=False)

    meta = {
        "n_samples": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    with open(outdir / "embedding_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def plot_umap(
    embeddings: np.ndarray,
    labels: pd.Series,
    outpath: Path,
    title: str,
    random_state: int = 42,
) -> pd.DataFrame:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    z = reducer.fit_transform(embeddings)

    df_plot = pd.DataFrame(
        {
            "umap_1": z[:, 0],
            "umap_2": z[:, 1],
            "label": labels.astype(str).to_numpy(),
        }
    )
    df_plot.to_csv(outpath.with_suffix(".csv"), index=False)

    unique_labels = sorted(df_plot["label"].unique().tolist())
    label_to_code = {lab: i for i, lab in enumerate(unique_labels)}
    codes = df_plot["label"].map(label_to_code).to_numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_plot["umap_1"], df_plot["umap_2"], c=codes, s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # manual legend
    handles = []
    for lab in unique_labels:
        code = label_to_code[lab]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                label=lab,
                markerfacecolor=scatter.cmap(scatter.norm(code)),
                markeredgecolor="none",
            )
        )
    plt.legend(handles=handles, title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return df_plot


def compute_median_cosine_distance_matrix(
    embeddings: np.ndarray,
    labels: pd.Series,
    out_csv: Path,
) -> pd.DataFrame:
    labels = labels.astype(str).reset_index(drop=True)
    unique_labels = sorted(labels.unique().tolist())

    rows = []
    for lab_i in unique_labels:
        idx_i = np.where(labels.to_numpy() == lab_i)[0]
        emb_i = embeddings[idx_i]

        row = {"label": lab_i}
        for lab_j in unique_labels:
            idx_j = np.where(labels.to_numpy() == lab_j)[0]
            emb_j = embeddings[idx_j]

            d = cosine_distances(emb_i, emb_j)
            if lab_i == lab_j:
                # remove diagonal self-distances
                if len(idx_i) > 1:
                    d = d[~np.eye(d.shape[0], dtype=bool)]
                else:
                    d = np.array([0.0], dtype=float)

            median_d = float(np.median(d))
            row[lab_j] = median_d
        rows.append(row)

    mat = pd.DataFrame(rows).set_index("label")
    mat.to_csv(out_csv)
    return mat


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    outpath: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def run_in_distribution_classification(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    split_col: str,
    target_col: str,
    train_splits: List[str],
    test_split: str,
    outdir: Path,
    seed: int,
) -> Dict:
    df = df.copy()
    df["__row_idx__"] = np.arange(len(df))

    train_mask = df[split_col].isin(train_splits)
    test_mask = df[split_col] == test_split

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("No train/val or test samples found for classification.")

    X_train = embeddings[train_df["__row_idx__"].to_numpy()]
    X_test = embeddings[test_df["__row_idx__"].to_numpy()]

    y_train_raw = train_df[target_col].astype(str).to_numpy()
    y_test_raw = test_df[target_col].astype(str).to_numpy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T
    report_df.to_csv(outdir / "classification_report.csv")

    plot_confusion_matrix(
        cm=cm,
        class_names=list(le.classes_),
        outpath=outdir / "classification_confusion_matrix.png",
        title="In-distribution multinomial logistic regression",
    )

    pred_df = pd.DataFrame(
        {
            "image_path": test_df["image_path"].to_numpy() if "image_path" in test_df.columns else "",
            "true_label": y_test_raw,
            "pred_label": le.inverse_transform(y_pred),
        }
    )
    if "image_id" in test_df.columns:
        pred_df["image_id"] = test_df["image_id"].to_numpy()
    if "trial" in test_df.columns:
        pred_df["trial"] = test_df["trial"].to_numpy()

    pred_df.to_csv(outdir / "classification_predictions.csv", index=False)

    metrics = {
        "test_accuracy": acc,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "classes": list(le.classes_),
    }
    with open(outdir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_loco_ridge_regression(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    split_col: str,
    target_col: str,
    train_splits: List[str],
    test_split: str,
    outdir: Path,
    alpha: float,
) -> pd.DataFrame:
    """
    LOCO over target tiers:
      For each held-out tier t:
        train on train/val and target != t
        test on test and target == t
    """
    df = df.copy()
    df["__row_idx__"] = np.arange(len(df))
    df["__target_str__"] = df[target_col].astype(str)
    df["__target_num__"] = parse_numeric_targets(df[target_col])

    tiers = sorted(df["__target_str__"].unique().tolist())
    rows = []
    pred_rows = []

    for held_out in tiers:
        train_mask = df[split_col].isin(train_splits) & (df["__target_str__"] != held_out)
        test_mask = (df[split_col] == test_split) & (df["__target_str__"] == held_out)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = embeddings[train_df["__row_idx__"].to_numpy()]
        y_train = train_df["__target_num__"].to_numpy()

        X_test = embeddings[test_df["__row_idx__"].to_numpy()]
        y_test = test_df["__target_num__"].to_numpy()

        reg = Ridge(alpha=alpha, random_state=None)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred)) if len(np.unique(y_test)) > 1 else float("nan")
        bias = float(np.mean(y_pred - y_test))

        rows.append(
            {
                "held_out_tier": held_out,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "true_numeric_value": float(np.unique(y_test)[0]) if len(np.unique(y_test)) == 1 else float(np.mean(y_test)),
                "pred_mean": float(np.mean(y_pred)),
                "pred_std": float(np.std(y_pred)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "bias_mean_pred_minus_true": bias,
            }
        )

        tmp = pd.DataFrame(
            {
                "held_out_tier": held_out,
                "true_value": y_test,
                "pred_value": y_pred,
            }
        )
        if "image_id" in test_df.columns:
            tmp["image_id"] = test_df["image_id"].to_numpy()
        if "trial" in test_df.columns:
            tmp["trial"] = test_df["trial"].to_numpy()
        tmp["split"] = test_df[split_col].to_numpy()
        tmp["target_raw"] = test_df[target_col].astype(str).to_numpy()
        pred_rows.append(tmp)

    if not rows:
        raise ValueError("LOCO regression produced no valid held-out tiers. Check your splits and target column.")

    metrics_df = pd.DataFrame(rows).sort_values("true_numeric_value")
    metrics_df.to_csv(outdir / "loco_ridge_metrics.csv", index=False)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_csv(outdir / "loco_ridge_predictions.csv", index=False)

    # Summary plot: predicted mean vs true tier
    plt.figure(figsize=(8, 5))
    x = np.arange(len(metrics_df))
    y_true = metrics_df["true_numeric_value"].to_numpy()
    y_pred = metrics_df["pred_mean"].to_numpy()
    y_std = metrics_df["pred_std"].to_numpy()

    plt.errorbar(x, y_pred, yerr=y_std, fmt="o", label="Predicted mean ± std")
    plt.plot(x, y_true, "x", label="True value")
    plt.xticks(x, metrics_df["held_out_tier"].astype(str).tolist(), rotation=45, ha="right")
    plt.xlabel("Held-out tier")
    plt.ylabel("Target value")
    plt.title("LOCO Ridge regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loco_ridge_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Error by tier plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        metrics_df["held_out_tier"].astype(str).tolist(),
        metrics_df["mae"].to_numpy(),
        marker="o",
    )
    plt.xlabel("Held-out tier")
    plt.ylabel("MAE")
    plt.title("LOCO prediction error by tier")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "loco_ridge_mae_by_tier.png", dpi=300, bbox_inches="tight")
    plt.close()

    return metrics_df


# -----------------------------
# Main
# -----------------------------
@dataclass
class Args:
    csv: str
    outdir: str
    path_col: str
    split_col: str
    target_col: str
    image_id_col: str
    trial_col: str
    batch_size: int
    num_workers: int
    num_threads: int
    image_size: int
    mask_type: str
    mask_value: float
    train_split: str
    val_split: str
    test_split: str
    ridge_alpha: float
    seed: int
    torch_hub_dir: Optional[str]


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Frozen DINOv2 ViT-B/14 pipeline")
    p.add_argument("--csv", type=str, required=True, help="CSV file with metadata")
    p.add_argument("--outdir", type=str, default="results_dinov2")
    p.add_argument("--path-col", type=str, default="path")
    p.add_argument("--split-col", type=str, default="split")
    p.add_argument("--target-col", type=str, default="target")
    p.add_argument("--image-id-col", type=str, default="image_id")
    p.add_argument("--trial-col", type=str, default="trial")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 4))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--mask-type", type=str, default="none", choices=["none", "circle", "square"])
    p.add_argument("--mask-value", type=float, default=0.25)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-hub-dir", type=str, default=None)
    ns = p.parse_args()
    return Args(**vars(ns))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    torch.set_num_threads(args.num_threads)
    outdir = ensure_dir(args.outdir)

    # Read CSV
    df = pd.read_csv(args.csv)

    required = [args.path_col, args.split_col, args.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Keep optional columns if present
    df = df.copy()
    df["image_path"] = df[args.path_col].astype(str)
    df["split"] = normalize_split_col(df[args.split_col])
    df["target"] = df[args.target_col]

    if args.image_id_col in df.columns:
        df["image_id"] = df[args.image_id_col]
    if args.trial_col in df.columns:
        df["trial"] = df[args.trial_col]

    # Drop rows whose files do not exist
    exists_mask = df["image_path"].map(lambda p: Path(p).exists())
    n_missing_files = int((~exists_mask).sum())
    if n_missing_files > 0:
        print(f"[WARN] Dropping {n_missing_files} rows because image files do not exist.")
    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid images left after checking paths.")

    # Save cleaned metadata
    df.to_csv(outdir / "cleaned_metadata.csv", index=False)

    # Load model
    print("[INFO] Loading DINOv2 ViT-B/14 ...")
    model = load_dinov2_vitb14(torch_hub_dir=args.torch_hub_dir)

    # Extract embeddings
    embeddings = extract_embeddings(
        model=model,
        df=df,
        path_col="image_path",
        batch_size=args.batch_size,
        mask_type=args.mask_type,
        mask_value=args.mask_value,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    save_embeddings(outdir, df, embeddings)

    # UMAP by target
    plot_umap(
        embeddings=embeddings,
        labels=df["target"],
        outpath=outdir / "umap_by_target.png",
        title="UMAP of frozen DINOv2 embeddings colored by target",
        random_state=args.seed,
    )

    # UMAP by split
    plot_umap(
        embeddings=embeddings,
        labels=df["split"],
        outpath=outdir / "umap_by_split.png",
        title="UMAP of frozen DINOv2 embeddings colored by split",
        random_state=args.seed,
    )

    # Cosine distance matrix across tiers
    compute_median_cosine_distance_matrix(
        embeddings=embeddings,
        labels=df["target"],
        out_csv=outdir / "median_cosine_distance_matrix.csv",
    )

    # In-distribution classification
    train_splits = [args.train_split.strip().lower()]
    if args.val_split.strip():
        train_splits.append(args.val_split.strip().lower())
    train_splits = list(dict.fromkeys(train_splits))  # unique, keep order
    test_split = args.test_split.strip().lower()

    classif_outdir = ensure_dir(outdir / "classification")
    cls_metrics = run_in_distribution_classification(
        df=df,
        embeddings=embeddings,
        split_col="split",
        target_col="target",
        train_splits=train_splits,
        test_split=test_split,
        outdir=classif_outdir,
        seed=args.seed,
    )

    # LOCO Ridge regression
    loco_outdir = ensure_dir(outdir / "loco_ridge")
    loco_df = run_loco_ridge_regression(
        df=df,
        embeddings=embeddings,
        split_col="split",
        target_col="target",
        train_splits=train_splits,
        test_split=test_split,
        outdir=loco_outdir,
        alpha=args.ridge_alpha,
    )

    summary = {
        "n_total_samples": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "classification_test_accuracy": cls_metrics["test_accuracy"],
        "classification_n_train": cls_metrics["n_train"],
        "classification_n_test": cls_metrics["n_test"],
        "loco_n_tiers_evaluated": int(len(loco_df)),
        "mean_loco_mae": float(loco_df["mae"].mean()),
        "mean_loco_rmse": float(loco_df["rmse"].mean()),
    }
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Finished.")
    print(f"Results saved to: {outdir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()