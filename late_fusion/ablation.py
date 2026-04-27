import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_root))
    __package__ = "late_fusion"

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    AMP_ENABLED,
    CLINICAL_AUX_LOSS_WEIGHT,
    DEFAULT_THRESHOLD,
    DROPOUT_RATE,
    ECG_AUX_LOSS_WEIGHT,
    ECG_LENGTH,
    ECG_LEADS,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    LEARNING_RATE,
    METRICS_DIR,
    NUM_CLINICAL_FEATURES,
    NUM_EPOCHS,
    SEED,
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_MIN,
    THRESHOLD_SEARCH_STEPS,
    WEIGHT_DECAY,
)
from .dataset import load_and_prepare_data
from .engine import _autocast_context
from .losses import FocalLoss
from .model import ClinicalOnlyModel, ECGOnlyModel, LateFusionModel

try:
    GradScaler = torch.amp.GradScaler
except AttributeError:
    GradScaler = torch.cuda.amp.GradScaler


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def find_best_threshold(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    for threshold in np.linspace(THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX, THRESHOLD_SEARCH_STEPS):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, max(best_f1, 0.0)


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    try:
        average_precision = average_precision_score(labels, probs)
    except ValueError:
        average_precision = 0.0

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
        "average_precision": average_precision,
        "threshold": float(threshold),
    }


def compute_loss_and_logits(criterion, outputs, labels):
    if isinstance(outputs, tuple):
        final_logits, ecg_logits, clinical_logits = outputs
        final_loss = criterion(final_logits, labels)
        ecg_loss = criterion(ecg_logits, labels)
        clinical_loss = criterion(clinical_logits, labels)
        total_loss = (
            final_loss
            + ECG_AUX_LOSS_WEIGHT * ecg_loss
            + CLINICAL_AUX_LOSS_WEIGHT * clinical_loss
        )
        return total_loss, final_logits

    loss = criterion(outputs, labels)
    return loss, outputs


def train_one_epoch(model, loader, criterion, optimizer, device, threshold, use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with _autocast_context(use_amp):
            outputs = model(ecg, clinical)
            loss, final_logits = compute_loss_and_logits(criterion, outputs, labels)

        if torch.isnan(loss):
            continue

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_probs.extend(torch.sigmoid(final_logits.detach()).cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    metrics = compute_metrics(all_labels, all_probs, threshold)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=DEFAULT_THRESHOLD, auto_threshold=False, use_amp=False):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        with _autocast_context(use_amp):
            outputs = model(ecg, clinical)
            loss, final_logits = compute_loss_and_logits(criterion, outputs, labels)

        running_loss += loss.item() * labels.size(0)
        all_probs.extend(torch.sigmoid(final_logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    if auto_threshold:
        threshold, _best_f1 = find_best_threshold(all_labels, all_probs)
    metrics = compute_metrics(all_labels, all_probs, threshold)
    return avg_loss, metrics


def build_model(name: str):
    if name == "ecg_only":
        return ECGOnlyModel(
            n_leads=ECG_LEADS,
            ecg_length=ECG_LENGTH,
            dropout=DROPOUT_RATE,
        )
    if name == "clinical_only":
        return ClinicalOnlyModel(
            n_clinical=NUM_CLINICAL_FEATURES,
            dropout=DROPOUT_RATE,
        )
    if name == "fusion":
        return LateFusionModel(
            n_leads=ECG_LEADS,
            ecg_length=ECG_LENGTH,
            n_clinical=NUM_CLINICAL_FEATURES,
            dropout=DROPOUT_RATE,
        )
    raise ValueError(f"Unknown ablation model: {name}")


def run_single_ablation(name, train_loader, val_loader, pos_weight, device, epochs):
    print("\n" + "=" * 70)
    print(f" ABLATION: {name}")
    print("=" * 70)

    model = build_model(name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Total parameters : {total_params:,}")

    criterion = FocalLoss(
        alpha=FOCAL_LOSS_ALPHA,
        gamma=FOCAL_LOSS_GAMMA,
        pos_weight=pos_weight.to(device),
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = AMP_ENABLED and device.type == "cuda"
    try:
        scaler = GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    best_val_f1 = 0.0
    best_threshold = DEFAULT_THRESHOLD
    best_epoch = 1
    best_state = None
    active_threshold = DEFAULT_THRESHOLD

    print(f"\n{'Epoch':>5} | {'Tr Loss':>8} | {'Val Loss':>8} | {'Val F1':>6} | {'Th':>5}")
    print("-" * 45)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, _train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            threshold=active_threshold, use_amp=use_amp, scaler=scaler,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            auto_threshold=True, use_amp=use_amp,
        )
        active_threshold = val_metrics["threshold"]
        print(
            f"{epoch:5d} | {train_loss:8.4f} | {val_loss:8.4f} | "
            f"{val_metrics['f1']:6.4f} | {active_threshold:5.2f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = active_threshold
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    elapsed = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_metrics = evaluate(
        model, val_loader, criterion, device, threshold=best_threshold, use_amp=use_amp
    )

    print("-" * 45)
    print(
        f"[DONE] {name} finished in {elapsed:.1f}s | "
        f"Best Val F1: {best_val_f1:.4f} at epoch {best_epoch} | "
        f"Final reported Val F1: {final_val_metrics['f1']:.4f}"
    )

    return {
        "model": name,
        "params": total_params,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 6),
        "best_threshold": round(best_threshold, 6),
        "val_loss": round(final_val_loss, 6),
        "val_accuracy": round(final_val_metrics["accuracy"], 6),
        "val_precision": round(final_val_metrics["precision"], 6),
        "val_recall": round(final_val_metrics["recall"], 6),
        "val_f1": round(final_val_metrics["f1"], 6),
        "val_auc": round(final_val_metrics["auc"], 6),
        "val_average_precision": round(final_val_metrics["average_precision"], 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Run late-fusion modality ablation")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override training DataLoader worker count")
    parser.add_argument("--val-workers", type=int, default=None,
                        help="Override validation DataLoader worker count")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Epochs to train each model")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("\n" + "=" * 70)
    print(" DATA LOADING")
    print("=" * 70)
    train_loader, val_loader, pos_weight, _test_metadata = load_and_prepare_data(
        subset=args.subset,
        train_num_workers=args.num_workers,
        val_num_workers=args.val_workers,
    )

    results = []
    for name in ["ecg_only", "clinical_only", "fusion"]:
        seed_everything()
        results.append(
            run_single_ablation(
                name=name,
                train_loader=train_loader,
                val_loader=val_loader,
                pos_weight=pos_weight,
                device=device,
                epochs=args.epochs,
            )
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = METRICS_DIR / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(" ABLATION SUMMARY")
    print("=" * 70)
    for row in results:
        print(
            f"{row['model']:>13} | F1={row['val_f1']:.4f} | "
            f"Best={row['best_val_f1']:.4f} | "
            f"AUC={row['val_auc']:.4f} | Params={row['params']:,}"
        )
    print(f"[SAVE] Ablation results -> {results_path}")


if __name__ == "__main__":
    main()
