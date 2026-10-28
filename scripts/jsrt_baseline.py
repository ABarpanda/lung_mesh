"""
JSRT lung nodule classification with a pretrained ResNet (PyTorch).

Task
----
Binary classification of the nodules in the JSRT dataset using the
malignancy labels from the metadata CSV (malignant = 1, benign = 0).

How to run
----------
Edit CONFIG below if needed, then run:

    python jsrt_resnet_train_v2.py

Outputs
-------
Each run creates runs/<timestamp>_<arch>/ containing:
    training.log         full log (console output is saved here too)
    config.json          settings used for the run
    split.csv            train / val / test assignment for every image
    history.csv          one row per epoch: loss, accuracy, error, AUC,
                         sensitivity, specificity, learning rate, time
    best_model.pt        checkpoint with the best validation AUC
    last_model.pt        checkpoint from the last epoch trained
    test_metrics.json    final metrics on the held-out test set
    test_predictions.csv per-image test probabilities and predictions
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "image_dir": "./dataset/JSRT_images",          # folder with the JSRT PNG images
    "label_csv": "./dataset/jsrt_metadata.csv",    # metadata file with study_id and state
    "label_column": "state",                       # column used for the label
    "positive_label": "malignant",                 # value in label_column that means class 1
    "output_root": "./runs",                       # where run folders are written
    "arch": "resnet18",                            # "resnet18", "resnet34", or "resnet50"
    "img_size": 512,                               # network input size (images are 2048x2048)
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "patience": 8,                                 # early stopping patience (epochs)
    "threshold": 0.5,                              # probability cutoff for the positive class
    "seed": 42,
    "num_workers": 2,
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
    """Load a JSRT image as 8-bit grayscale. Handles PNG and the raw .IMA format."""
    if path.lower().endswith(".ima"):
        raw = np.fromfile(path, dtype=">u2").reshape(2048, 2048).astype(np.float32)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return Image.fromarray((raw * 255).astype(np.uint8))
    return Image.open(path).convert("L")


def load_labels(cfg: dict, log: logging.Logger) -> pd.DataFrame:
    """Read the metadata CSV and return a DataFrame with filename and label columns."""
    meta = pd.read_csv(cfg["label_csv"])
    meta = meta.rename(columns={"study_id": "filename"})

    values = meta[cfg["label_column"]].astype(str).str.strip().str.lower()
    meta["label"] = (values == cfg["positive_label"].lower()).astype(int)
    meta = meta[["filename", "label"]]

    exists = meta["filename"].apply(
        lambda f: os.path.exists(os.path.join(cfg["image_dir"], f))
    )
    if (~exists).any():
        log.warning(f"{(~exists).sum()} images listed in the metadata were not found "
                    f"in {cfg['image_dir']} and will be skipped")
    meta = meta[exists].reset_index(drop=True)

    n_pos = int(meta["label"].sum())
    log.info(f"Labels from '{cfg['label_column']}' (positive = '{cfg['positive_label']}'): "
             f"{len(meta)} images | positive: {n_pos} | negative: {len(meta) - n_pos}")
    return meta


# =============================================================================
# Data
# =============================================================================
def build_transforms(img_size: int):
    # No horizontal flips: chest anatomy is asymmetric (heart on the left).
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
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
        nn.Linear(model.fc.in_features, 1),  # one logit for binary classification
    )
    return model


def save_checkpoint(path: str, model: nn.Module, cfg: dict, epoch: int, val_auc: float) -> None:
    """Save weights with the settings needed to reload the model later."""
    torch.save({
        "model_state": model.state_dict(),
        "arch": cfg["arch"],
        "img_size": cfg["img_size"],
        "threshold": cfg["threshold"],
        "epoch": epoch,
        "val_auc": val_auc,
    }, path)


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
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


def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
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
    run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{cfg['arch']}"
    run_dir = os.path.join(cfg["output_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = setup_logger(os.path.join(run_dir, "training.log"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"Run directory: {run_dir}")

    # ---- Reproducibility and device -----------------------------------------
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ---- Labels and stratified 70 / 15 / 15 split ---------------------------
    df = load_labels(cfg, log)

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"],
                                         random_state=cfg["seed"])
    train_df, val_df = train_test_split(train_df, test_size=0.1765,
                                        stratify=train_df["label"], random_state=cfg["seed"])

    split_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])
    split_df.to_csv(os.path.join(run_dir, "split.csv"), index=False)
    log.info(f"Split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ---- Data loaders -------------------------------------------------------
    train_tf, eval_tf = build_transforms(cfg["img_size"])
    loader_kwargs = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    train_loader = DataLoader(JSRTDataset(train_df, cfg["image_dir"], train_tf),
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(JSRTDataset(val_df, cfg["image_dir"], eval_tf),
                            shuffle=False, **loader_kwargs)
    test_loader = DataLoader(JSRTDataset(test_df, cfg["image_dir"], eval_tf),
                             shuffle=False, **loader_kwargs)

    # ---- Model, loss, optimizer, scheduler ----------------------------------
    model = build_model(cfg["arch"], cfg["dropout"]).to(device)

    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # ---- Training loop ------------------------------------------------------
    history = []
    best_auc, epochs_without_gain = -1.0, 0
    best_path = os.path.join(run_dir, "best_model.pt")
    last_path = os.path.join(run_dir, "last_model.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_probs, tr_targets = run_epoch(model, train_loader, criterion,
                                                  optimizer, device)
        va_loss, va_probs, va_targets = run_epoch(model, val_loader, criterion,
                                                  None, device)
        scheduler.step()

        tr = compute_metrics(tr_targets, tr_probs, cfg["threshold"])
        va = compute_metrics(va_targets, va_probs, cfg["threshold"])
        elapsed = time.time() - start

        history.append({
            "epoch": epoch, "lr": lr, "time_sec": round(elapsed, 1),
            "train_loss": tr_loss, "train_acc": tr["accuracy"],
            "train_error": tr["error"], "train_auc": tr["auc"],
            "val_loss": va_loss, "val_acc": va["accuracy"],
            "val_error": va["error"], "val_auc": va["auc"],
            "val_sensitivity": va["sensitivity"], "val_specificity": va["specificity"],
        })
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

        log.info(
            f"Epoch {epoch:03d}/{cfg['epochs']} | lr {lr:.2e} | {elapsed:.1f}s\n"
            f"    train: loss {tr_loss:.4f}  acc {tr['accuracy']:.4f}  "
            f"error {tr['error']:.4f}  AUC {tr['auc']:.4f}\n"
            f"    val  : loss {va_loss:.4f}  acc {va['accuracy']:.4f}  "
            f"error {va['error']:.4f}  AUC {va['auc']:.4f}  "
            f"sens {va['sensitivity']:.4f}  spec {va['specificity']:.4f}"
        )

        save_checkpoint(last_path, model, cfg, epoch, va["auc"])

        if va["auc"] > best_auc:
            best_auc = va["auc"]
            epochs_without_gain = 0
            save_checkpoint(best_path, model, cfg, epoch, va["auc"])
            log.info(f"    New best model saved (val AUC {best_auc:.4f})")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg["patience"]:
                log.info(f"Early stopping at epoch {epoch} (best val AUC {best_auc:.4f})")
                break

    # ---- Final test evaluation with the best checkpoint ---------------------
    log.info("Evaluating the best model on the test set...")
    model = load_checkpoint(best_path, device)
    te_loss, te_probs, te_targets = run_epoch(model, test_loader, criterion, None, device)
    te = compute_metrics(te_targets, te_probs, cfg["threshold"])
    te["loss"] = te_loss

    report = classification_report(
        te_targets, (te_probs >= cfg["threshold"]).astype(int),
        target_names=["Benign", "Malignant"], digits=4, zero_division=0,
    )
    log.info(f"\n===== Test results =====\n"
             f"loss {te['loss']:.4f} | acc {te['accuracy']:.4f} | error {te['error']:.4f} | "
             f"AUC {te['auc']:.4f} | sens {te['sensitivity']:.4f} | spec {te['specificity']:.4f}\n"
             f"{report}")

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(te, f, indent=2)

    pred_df = test_df.reset_index(drop=True).copy()
    pred_df["probability"] = te_probs
    pred_df["prediction"] = (te_probs >= cfg["threshold"]).astype(int)
    pred_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

    log.info(f"Done. All outputs saved to {run_dir}")


if __name__ == "__main__":
    main()