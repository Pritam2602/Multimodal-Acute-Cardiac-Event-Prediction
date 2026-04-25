# ==========================================
# TRAIN - Main entry point
# ==========================================
# Usage:
#   cd D:\MINI_PROJECT
#   python -m early_fusion.train
#   python -m early_fusion.train --subset 1000   (quick test)
#   python -m early_fusion.train --num-workers 2 --val-workers 0
# ==========================================

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE,
    DEFAULT_THRESHOLD, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA,
)
from .dataset import load_and_prepare_data
from .engine import train_one_epoch, evaluate
from .losses import FocalLoss
from .model import EarlyFusionModel
from .plots import (
    save_accuracy_plot,
    save_confusion_matrix_plot,
    save_loss_plot,
    save_metric_plot,
    save_model_comparison_table,
    save_pr_curve_plot,
    save_roc_curve_plot,
)


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # safe with fixed input sizes, faster conv kernels


def _build_empty_history() -> dict:
    return {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": [],
    }


def _latest_checkpoint_path() -> Path:
    return MODELS_DIR / "latest_checkpoint.pth"


def _best_model_path() -> Path:
    return MODELS_DIR / "early_fusion_model.pth"


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_f1: float,
    best_threshold: float,
    history: dict,
    args: argparse.Namespace,
):
    payload = {
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "args": vars(args),
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, checkpoint_path)
    print(f"[SAVE] Checkpoint -> {checkpoint_path}")


def _resume_from_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
):
    if not checkpoint_path.exists():
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError as exc:
        print(f"[LOAD] Checkpoint incompatible with current model: {exc}")
        print("[LOAD] Starting fresh training run with the updated architecture.")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history()

    # Restore scheduler state if available
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception:
            pass  # scheduler config may have changed; let it re-initialize

    history = checkpoint.get("history", _build_empty_history())
    best_val_f1 = float(checkpoint.get("best_val_f1", 0.0))
    best_threshold = float(checkpoint.get("best_threshold", DEFAULT_THRESHOLD))
    last_epoch = int(checkpoint.get("epoch", 0))

    random_state = checkpoint.get("random_state")
    if random_state is not None:
        random.setstate(random_state)

    numpy_state = checkpoint.get("numpy_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)

    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    next_epoch = last_epoch + 1
    print(f"[LOAD] Resuming from epoch {next_epoch} using {checkpoint_path}")
    print(f"[LOAD] Best Val F1 so far: {best_val_f1:.4f}")
    print(f"[LOAD] Best threshold so far: {best_threshold:.4f}")
    return next_epoch, best_val_f1, best_threshold, history


def _should_continue(next_epoch: int, total_epochs: int) -> bool:
    while True:
        response = input(f"[PROMPT] Continue to epoch {next_epoch}/{total_epochs}? [Y/n]: ").strip().lower()
        if response in {"", "y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("[PROMPT] Please enter 'y' or 'n'.")


def main():
    parser = argparse.ArgumentParser(description="Train Early Fusion AMI model")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples (for quick testing)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override training DataLoader worker count")
    parser.add_argument("--val-workers", type=int, default=None,
                        help="Override validation DataLoader worker count")
    args = parser.parse_args()

    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    for directory in [MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {directory}")

    print("\n" + "=" * 70)
    print(" DATA LOADING")
    print("=" * 70)
    train_loader, val_loader, pos_weight, _test_metadata = load_and_prepare_data(
        subset=args.subset,
        train_num_workers=args.num_workers,
        val_num_workers=args.val_workers,
    )

    print("\n" + "=" * 70)
    print(" MODEL")
    print("=" * 70)
    model = EarlyFusionModel(
        n_leads=ECG_LEADS,
        ecg_length=ECG_LENGTH,
        n_clinical=NUM_CLINICAL_FEATURES,
        dropout=DROPOUT_RATE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters : {total_params:,}")
    print(f"[MODEL] Trainable params : {train_params:,}")

    criterion = FocalLoss(
        alpha=FOCAL_LOSS_ALPHA,
        gamma=FOCAL_LOSS_GAMMA,
        pos_weight=pos_weight.to(device),
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    checkpoint_path = _latest_checkpoint_path()
    best_model_path = _best_model_path()
    resumed_from_checkpoint = checkpoint_path.exists()
    start_epoch, best_val_f1, best_threshold, history = _resume_from_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    )
    active_threshold = best_threshold

    print(f"\n{'Epoch':>5} | {'Tr Loss':>8} | {'Val Loss':>8} | "
          f"{'Tr Acc':>7} | {'Val Acc':>7} | {'Val P':>6} | {'Val R':>6} | {'Val F1':>6} | {'Th':>5}")
    print("-" * 86)

    start_time = time.time()
    completed_epochs = max(start_epoch - 1, 0)

    # Mixed precision scaler (float16 forward pass for ~2x GPU speedup)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            threshold=active_threshold, scaler=scaler,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, auto_threshold=True
        )
        active_threshold = val_metrics["threshold"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{split}_acc"].append(metrics["accuracy"])
            history[f"{split}_precision"].append(metrics["precision"])
            history[f"{split}_recall"].append(metrics["recall"])
            history[f"{split}_f1"].append(metrics["f1"])

        print(
            f"{epoch:5d} | {train_loss:8.4f} | {val_loss:8.4f} | "
            f"{train_metrics['accuracy']:7.4f} | {val_metrics['accuracy']:7.4f} | "
            f"{val_metrics['precision']:6.4f} | {val_metrics['recall']:6.4f} | {val_metrics['f1']:6.4f} | "
            f"{active_threshold:5.2f}"
        )

        completed_epochs = epoch
        if (not resumed_from_checkpoint and epoch == 1) or (val_metrics["f1"] > best_val_f1):
            best_val_f1 = val_metrics["f1"]
            best_threshold = active_threshold
            torch.save(model.state_dict(), best_model_path)
            print(
                f"         ^ New best Val F1 = {best_val_f1:.4f} at threshold {best_threshold:.4f}"
                " -- best model updated"
            )

        scheduler.step()

        _save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, epoch, best_val_f1, best_threshold, history, args
        )

        if epoch < NUM_EPOCHS and not _should_continue(epoch + 1, NUM_EPOCHS):
            print(f"[STOP] Training paused after epoch {epoch}. Rerun the script to resume from epoch {epoch + 1}.")
            break

    elapsed = time.time() - start_time
    print("-" * 86)
    print(
        f"[DONE] Training session finished in {elapsed:.1f}s  |  "
        f"Completed epochs: {completed_epochs} | Best Val F1: {best_val_f1:.4f} | "
        f"Best threshold: {best_threshold:.4f}"
    )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"[LOAD] Best model restored from {best_model_path}")

    final_val_loss, final_val_metrics, final_val_outputs = evaluate(
        model, val_loader, criterion, device, threshold=best_threshold, return_outputs=True
    )
    final_train_loss, final_train_metrics = evaluate(
        model, train_loader, criterion, device, threshold=best_threshold
    )

    metrics_payload = {
        "train": {
            "loss": round(final_train_loss, 6),
            **{k: round(v, 6) for k, v in final_train_metrics.items()},
        },
        "val": {
            "loss": round(final_val_loss, 6),
            **{k: round(v, 6) for k, v in final_val_metrics.items()},
        },
        "best_val_f1": round(best_val_f1, 6),
        "best_threshold": round(best_threshold, 6),
        "epochs_completed": completed_epochs,
        "target_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "loss_name": "FocalLoss",
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
        "validation_outputs": {
            "labels": [int(v) for v in final_val_outputs["labels"]],
            "preds": [int(v) for v in final_val_outputs["preds"]],
            "probs": [round(float(v), 6) for v in final_val_outputs["probs"]],
        },
    }

    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[SAVE] Metrics -> {metrics_path}")

    comparison_rows = [
        {"split": "Train", "loss": final_train_loss, **final_train_metrics},
        {"split": "Validation", "loss": final_val_loss, **final_val_metrics},
    ]

    comparison_csv_path = METRICS_DIR / "comparison_table.csv"
    with open(comparison_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "loss", "accuracy", "precision", "recall",
                "f1", "auc", "average_precision", "threshold",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    print(f"[SAVE] Comparison table -> {comparison_csv_path}")

    save_loss_plot(history, str(PLOTS_DIR / "loss.png"))
    save_accuracy_plot(history, str(PLOTS_DIR / "accuracy.png"))
    save_metric_plot(history, "precision", "Precision", str(PLOTS_DIR / "precision.png"))
    save_metric_plot(history, "recall", "Recall", str(PLOTS_DIR / "recall.png"))
    save_metric_plot(history, "f1", "F1 Score", str(PLOTS_DIR / "f1.png"))
    save_confusion_matrix_plot(
        final_val_outputs["labels"],
        final_val_outputs["preds"],
        str(PLOTS_DIR / "confusion_matrix.png"),
    )
    save_roc_curve_plot(
        final_val_outputs["labels"],
        final_val_outputs["probs"],
        final_val_metrics["auc"],
        str(PLOTS_DIR / "roc_curve.png"),
    )
    save_pr_curve_plot(
        final_val_outputs["labels"],
        final_val_outputs["probs"],
        final_val_metrics["average_precision"],
        str(PLOTS_DIR / "pr_curve.png"),
    )
    save_model_comparison_table(comparison_rows, str(PLOTS_DIR / "model_comparison_table.png"))

    print("\n" + "=" * 70)
    print(" ARTIFACTS SAVED")
    print("=" * 70)
    print(f"  Best model : {best_model_path}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {metrics_path}")
    print(f"               {comparison_csv_path}")
    print(f"  Plots      : {PLOTS_DIR / 'loss.png'}")
    print(f"               {PLOTS_DIR / 'accuracy.png'}")
    print(f"               {PLOTS_DIR / 'precision.png'}")
    print(f"               {PLOTS_DIR / 'recall.png'}")
    print(f"               {PLOTS_DIR / 'f1.png'}")
    print(f"               {PLOTS_DIR / 'confusion_matrix.png'}")
    print(f"               {PLOTS_DIR / 'roc_curve.png'}")
    print(f"               {PLOTS_DIR / 'pr_curve.png'}")
    print(f"               {PLOTS_DIR / 'model_comparison_table.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
