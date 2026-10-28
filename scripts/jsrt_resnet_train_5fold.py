"""
JSRT lung nodule classification with a pretrained ResNet (PyTorch).

Binary classification using malignancy labels from the metadata CSV
(malignant = 1, benign = 0). Uses stratified 5-fold cross-validation on
the development split and retains a separate held-out test set.
"""

import json
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "image_dir": "./dataset/JSRT_images",
    "label_csv": "./dataset/jsrt_metadata.csv",
    "label_column": "state",
    "positive_label": "malignant",
    "output_root": "./runs",
    "arch": "resnet18",  # "resnet18", "resnet34", or "resnet50"
    "img_size": 512,
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "patience": 8,
    "threshold": 0.5,
    "seed": 42,
    "num_workers": 2,
    "n_splits": 5,
    "test_size": 0.15,
}

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: str) -> logging.Logger:
    """Log to both the console and a training.log file."""
    logger = logging.getLogger("jsrt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_image(path: str) -> Image.Image:
    """Load a JSRT image as 8-bit grayscale. Handles PNG and raw .IMA format."""
    if path.lower().endswith(".ima"):
        raw = np.fromfile(path, dtype=">u2").reshape(2048, 2048).astype(np.float32)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return Image.fromarray((raw * 255).astype(np.uint8))
    return Image.open(path).convert("L")


def load_labels(cfg: dict, log: logging.Logger) -> pd.DataFrame:
    """Read metadata and return filename and binary label columns."""
    meta = pd.read_csv(cfg["label_csv"])
    meta = meta.rename(columns={"study_id": "filename"})

    values = meta[cfg["label_column"]].astype(str).str.strip().str.lower()
    meta["label"] = (values == cfg["positive_label"].lower()).astype(int)
    meta = meta[["filename", "label"]]

    exists = meta["filename"].apply(
        lambda f: os.path.exists(os.path.join(cfg["image_dir"], f))
    )
    if (~exists).any():
        log.warning(
            f"{(~exists).sum()} images listed in the metadata were not found "
            f"in {cfg['image_dir']} and will be skipped"
        )
    meta = meta[exists].reset_index(drop=True)

    n_pos = int(meta["label"].sum())
    log.info(
        f"Labels from '{cfg['label_column']}' (positive = "
        f"'{cfg['positive_label']}'): {len(meta)} images | "
        f"positive: {n_pos} | negative: {len(meta) - n_pos}"
    )
    return meta


# =============================================================================
# Data
# =============================================================================
def build_transforms(img_size: int):
    # No horizontal flips: chest anatomy is asymmetric (heart on the left).
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class JSRTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(os.path.join(self.image_dir, row["filename"]))
        return self.transform(img), torch.tensor(row["label"], dtype=torch.float32)


# =============================================================================
# Model
# =============================================================================
def build_model(arch: str, dropout: float) -> nn.Module:
    ctor, weights = ARCHS[arch]
    model = ctor(weights=weights)  # ImageNet-pretrained initialization
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


def save_checkpoint(path: str, model: nn.Module, cfg: dict,
                    epoch: int, val_auc: float) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "arch": cfg["arch"],
        "img_size": cfg["img_size"],
        "threshold": cfg["threshold"],
        "epoch": epoch,
        "val_auc": val_auc,
    }, path)


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    # weights_only=False is explicit because this checkpoint stores a dictionary
    # containing metadata as well as the state_dict.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


# =============================================================================
# Training and evaluation
# =============================================================================
def run_epoch(model, loader, criterion, optimizer, device):
    """One pass over a loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, probs, targets = 0.0, [], []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            targets.extend(y.cpu().numpy())

    probs, targets = np.array(probs), np.array(targets)
    return total_loss / len(targets), probs, targets


def compute_metrics(targets: np.ndarray, probs: np.ndarray,
                    threshold: float) -> dict:
    """Accuracy, error rate, sensitivity, specificity, and AUC."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    return {
        "auc": float(roc_auc_score(targets, probs)),
        "accuracy": float(accuracy),
        "error": float(1.0 - accuracy),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    cfg = CONFIG

    # ---- Output folder, config, and logging ---------------------------------
    run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{cfg['arch']}_5fold"
    run_dir = os.path.join(cfg["output_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = setup_logger(os.path.join(run_dir, "training.log"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"Run directory: {run_dir}")

    # ---- Reproducibility and device -----------------------------------------
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ---- Hold out the test set; use 5-fold CV on development data ------------
    df = load_labels(cfg, log)
    dev_df, test_df = train_test_split(
        df,
        test_size=cfg["test_size"],
        stratify=df["label"],
        random_state=cfg["seed"],
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_rows = [
        dev_df.assign(split="development"),
        test_df.assign(split="test"),
    ]
    pd.concat(split_rows).to_csv(
        os.path.join(run_dir, "split.csv"), index=False
    )
    log.info(
        f"Development set: {len(dev_df)} | held-out test set: {len(test_df)}"
    )

    train_tf, eval_tf = build_transforms(cfg["img_size"])
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
    }

    skf = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )
    fold_results = []
    fold_best_epochs = []
    fold_checkpoints = []

    # ---- Cross-validation ----------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(dev_df, dev_df["label"]), start=1
    ):
        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        log.info(f"\n========== Fold {fold}/{cfg['n_splits']} ==========")

        train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        val_df = dev_df.iloc[val_idx].reset_index(drop=True)

        train_loader = DataLoader(
            JSRTDataset(train_df, cfg["image_dir"], train_tf),
            shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            JSRTDataset(val_df, cfg["image_dir"], eval_tf),
            shuffle=False, **loader_kwargs
        )

        # Start every fold from the same pretrained weights, not another fold.
        set_seed(cfg["seed"] + fold)
        model = build_model(cfg["arch"], cfg["dropout"]).to(device)

        n_pos = int(train_df["label"].sum())
        n_neg = len(train_df) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"]
        )

        history = []
        best_auc, best_epoch, epochs_without_gain = -1.0, 0, 0
        best_path = os.path.join(fold_dir, "best_model.pt")
        last_path = os.path.join(fold_dir, "last_model.pt")

        for epoch in range(1, cfg["epochs"] + 1):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]

            tr_loss, tr_probs, tr_targets = run_epoch(
                model, train_loader, criterion, optimizer, device
            )
            va_loss, va_probs, va_targets = run_epoch(
                model, val_loader, criterion, None, device
            )
            scheduler.step()

            tr = compute_metrics(tr_targets, tr_probs, cfg["threshold"])
            va = compute_metrics(va_targets, va_probs, cfg["threshold"])
            elapsed = time.time() - start

            history.append({
                "fold": fold, "epoch": epoch, "lr": lr,
                "time_sec": round(elapsed, 1),
                "train_loss": tr_loss, "train_acc": tr["accuracy"],
                "train_error": tr["error"], "train_auc": tr["auc"],
                "val_loss": va_loss, "val_acc": va["accuracy"],
                "val_error": va["error"], "val_auc": va["auc"],
                "val_sensitivity": va["sensitivity"],
                "val_specificity": va["specificity"],
            })
            pd.DataFrame(history).to_csv(
                os.path.join(fold_dir, "history.csv"), index=False
            )

            log.info(
                f"Fold {fold} | Epoch {epoch:03d}/{cfg['epochs']} | "
                f"lr {lr:.2e} | {elapsed:.1f}s\n"
                f"    train: loss {tr_loss:.4f} acc {tr['accuracy']:.4f} "
                f"error {tr['error']:.4f} AUC {tr['auc']:.4f}\n"
                f"    val  : loss {va_loss:.4f} acc {va['accuracy']:.4f} "
                f"error {va['error']:.4f} AUC {va['auc']:.4f} "
                f"sens {va['sensitivity']:.4f} spec {va['specificity']:.4f}"
            )

            save_checkpoint(last_path, model, cfg, epoch, va["auc"])

            if va["auc"] > best_auc:
                best_auc = va["auc"]
                best_epoch = epoch
                epochs_without_gain = 0
                save_checkpoint(best_path, model, cfg, epoch, va["auc"])
                log.info(
                    f"    New best model saved (val AUC {best_auc:.4f})"
                )
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= cfg["patience"]:
                    log.info(
                        f"Early stopping fold {fold} at epoch {epoch} "
                        f"(best val AUC {best_auc:.4f})"
                    )
                    break

        # Record best validation metrics for this fold from its saved checkpoint.
        best_model = load_checkpoint(best_path, device)
        _, best_probs, best_targets = run_epoch(
            best_model, val_loader, criterion, None, device
        )
        best_metrics = compute_metrics(
            best_targets, best_probs, cfg["threshold"]
        )
        best_metrics["fold"] = fold
        best_metrics["best_epoch"] = best_epoch
        best_metrics["val_loss"] = float(
            min(row["val_loss"] for row in history
                if row["epoch"] == best_epoch)
        )
        fold_results.append(best_metrics)
        fold_best_epochs.append(best_epoch)
        fold_checkpoints.append(best_path)

        log.info(
            f"Fold {fold} best: epoch {best_epoch}, "
            f"AUC {best_metrics['auc']:.4f}, "
            f"accuracy {best_metrics['accuracy']:.4f}, "
            f"sensitivity {best_metrics['sensitivity']:.4f}, "
            f"specificity {best_metrics['specificity']:.4f}"
        )

        del model, best_model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Save and summarise cross-validation results ------------------------
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(run_dir, "cv_results.csv"), index=False)

    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "val_loss"]
    summary = results_df[metric_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(run_dir, "cv_summary.csv"))

    log.info("\n===== Cross-validation summary (mean ± std) =====")
    for metric in metric_cols:
        log.info(
            f"{metric}: {results_df[metric].mean():.4f} ± "
            f"{results_df[metric].std():.4f}"
        )
    log.info(
        f"Best epochs by fold: {fold_best_epochs}; "
        f"median best epoch: {int(np.median(fold_best_epochs))}"
    )

    # ---- Evaluate the ensemble of fold-best models on held-out test set -----
    # The test set was never used for training, checkpoint selection or tuning.
    test_loader = DataLoader(
        JSRTDataset(test_df, cfg["image_dir"], eval_tf),
        shuffle=False, **loader_kwargs
    )
    test_targets = None
    fold_test_probs = []

    log.info("Evaluating fold-model ensemble on the held-out test set...")
    for fold, checkpoint in enumerate(fold_checkpoints, start=1):
        model = load_checkpoint(checkpoint, device)
        te_loss_fold, te_probs, te_targets = run_epoch(
            model, test_loader,
            # Same training-fold criterion used for that model is not needed
            # for probability inference; BCE loss is computed below without weighting.
            nn.BCEWithLogitsLoss(), None, device
        )
        if test_targets is None:
            test_targets = te_targets
        fold_test_probs.append(te_probs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    te_probs = np.mean(np.stack(fold_test_probs, axis=0), axis=0)
    te = compute_metrics(test_targets, te_probs, cfg["threshold"])

    # Calculate unweighted BCE loss for the ensemble probabilities.
    eps = 1e-7
    te["loss"] = float(
        -np.mean(
            test_targets * np.log(np.clip(te_probs, eps, 1 - eps))
            + (1 - test_targets) * np.log(np.clip(1 - te_probs, eps, 1 - eps))
        )
    )

    report = classification_report(
        test_targets, (te_probs >= cfg["threshold"]).astype(int),
        target_names=["Benign", "Malignant"], digits=4, zero_division=0,
    )
    log.info(
        f"\n===== Test results (mean probabilities from {cfg['n_splits']} fold models) =====\n"
        f"loss {te['loss']:.4f} | acc {te['accuracy']:.4f} | "
        f"error {te['error']:.4f} | AUC {te['auc']:.4f} | "
        f"sens {te['sensitivity']:.4f} | spec {te['specificity']:.4f}\n"
        f"{report}"
    )

    with open(os.path.join(run_dir, "test_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(te, f, indent=2)

    pred_df = test_df.reset_index(drop=True).copy()
    pred_df["probability"] = te_probs
    pred_df["prediction"] = (te_probs >= cfg["threshold"]).astype(int)
    pred_df.to_csv(
        os.path.join(run_dir, "test_predictions.csv"), index=False
    )

    log.info(f"Done. All outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
