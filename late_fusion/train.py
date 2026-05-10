# ==========================================
# TRAIN - Main entry point
# ==========================================
# Usage:
#   cd D:\MINI_PROJECT
#   python -m late_fusion.train
#   python -m late_fusion.train --subset 1000   (quick test)
#   python -m late_fusion.train --num-workers 2 --val-workers 0
# ==========================================

import argparse
import csv
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
import torch.nn as nn
import torch.optim as optim

from .config import (
    CLINICAL_AUX_LOSS_WEIGHT, EARLY_STOPPING_PATIENCE, ECG_AUX_LOSS_WEIGHT,
    ECG_BRANCH_LR_MULTIPLIER, ECG_BRANCH_TYPE, ECG_CNN_FILTERS, ECG_LSTM_HIDDEN_SIZE,
    FORCE_FRESH_TRAIN, FUSION_HIDDEN_DIM, FUSION_TYPE,
    HARD_NEGATIVE_POWER, HARD_NEGATIVE_WEIGHT,
    SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHTED_SAMPLING_DEFAULT,
    MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    ECG_LEADS, ECG_LENGTH, NUM_CLINICAL_FEATURES, DROPOUT_RATE,
    DEFAULT_THRESHOLD, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA,
)
from .dataset import load_and_prepare_data
from .engine import train_one_epoch, evaluate
from .losses import BCEWithLogitsLossWrapper, FocalLoss, HybridBCELoss
from .model import LateFusionModel, get_architecture_metadata
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
    torch.backends.cudnn.benchmark = False


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
    return MODELS_DIR / "late_fusion_model.pth"


def _build_criterion(args: argparse.Namespace, pos_weight: torch.Tensor):
    pos_weight = pos_weight.to(args.device)

    if args.loss_mode == "bce":
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight), "BCEWithLogitsLoss"

    if args.loss_mode == "hybrid":
        return HybridBCELoss(
            alpha=FOCAL_LOSS_ALPHA,
            gamma=FOCAL_LOSS_GAMMA,
            pos_weight=pos_weight,
            bce_weight=args.hybrid_bce_weight,
            focal_weight=args.hybrid_focal_weight,
        ), (
            f"HybridBCELoss(bce={args.hybrid_bce_weight:.2f},"
            f"focal={args.hybrid_focal_weight:.2f})"
        )

    return FocalLoss(
        alpha=FOCAL_LOSS_ALPHA,
        gamma=FOCAL_LOSS_GAMMA,
        pos_weight=pos_weight,
    ), "FocalLoss"


def _make_grad_scaler(amp_enabled: bool):
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler(enabled=amp_enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)
    return None


def _build_optimizer(model: nn.Module) -> optim.Optimizer:
    ecg_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("ecg_branch."):
            ecg_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if ecg_params:
        param_groups.append(
            {
                "params": ecg_params,
                "lr": LEARNING_RATE * ECG_BRANCH_LR_MULTIPLIER,
                "name": "ecg_branch",
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": LEARNING_RATE,
                "name": "non_ecg",
            }
        )
    return optim.Adam(param_groups, lr=LEARNING_RATE)


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    epoch: int,
    best_val_f1: float,
    best_threshold: float,
    history: dict,
    args: argparse.Namespace,
):
    architecture_metadata = get_architecture_metadata(model)
    payload = {
        "model_name": "late_fusion",
        "num_clinical_features": NUM_CLINICAL_FEATURES,
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "architecture_metadata": architecture_metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "args": vars(args),
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, checkpoint_path)
    print(f"[SAVE] Checkpoint -> {checkpoint_path}")


def _current_architecture_metadata(model: nn.Module) -> dict:
    return get_architecture_metadata(model)


def _architecture_mismatch_reasons(checkpoint_metadata: dict | None, current_metadata: dict) -> list[str]:
    if not checkpoint_metadata:
        return ["missing architecture metadata"]

    reasons = []
    keys_to_compare = [
        "ecg_branch_type",
        "fusion_type",
        "ecg_length",
        "n_leads",
        "n_clinical",
        "cnn_filters",
        "lstm_hidden_size",
        "fusion_hidden_dim",
        "ecg_embedding_dim",
    ]
    for key in keys_to_compare:
        if checkpoint_metadata.get(key) != current_metadata.get(key):
            reasons.append(
                f"{key}: checkpoint={checkpoint_metadata.get(key)!r}, current={current_metadata.get(key)!r}"
            )
    return reasons


def _resume_from_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
):
    current_architecture = _current_architecture_metadata(model)

    if FORCE_FRESH_TRAIN:
        print("[LOAD] FORCE_FRESH_TRAIN=True -> ignoring any previous checkpoint and starting from scratch.")
        print("[LOAD] Training mode: fresh-started")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history(), "fresh-started"

    if not checkpoint_path.exists():
        print("[LOAD] No checkpoint found -> fresh start.")
        print("[LOAD] Training mode: fresh-started")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history(), "fresh-started"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "late_fusion":
        print(f"[LOAD] Ignoring checkpoint not created by late_fusion: {checkpoint_path}")
        print("[LOAD] Starting fresh late-fusion training run.")
        print("[LOAD] Training mode: fresh-started")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history(), "fresh-started"
    if checkpoint.get("num_clinical_features") not in (None, NUM_CLINICAL_FEATURES):
        print(
            "[LOAD] Checkpoint clinical feature count "
            f"({checkpoint.get('num_clinical_features')}) does not match current config "
            f"({NUM_CLINICAL_FEATURES})."
        )
        print("[LOAD] Starting fresh training run with the updated engineered feature set.")
        print("[LOAD] Training mode: fresh-started")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history(), "fresh-started"

    architecture_mismatch_reasons = _architecture_mismatch_reasons(
        checkpoint.get("architecture_metadata"),
        current_architecture,
    )
    if architecture_mismatch_reasons:
        print("[LOAD] Checkpoint architecture differs from the current model.")
        for reason in architecture_mismatch_reasons:
            print(f"[LOAD]   - {reason}")
        print("[LOAD] Refusing to load incompatible checkpoint. Starting fresh with the new ECG architecture.")
        print("[LOAD] Training mode: fresh-started")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history(), "fresh-started"

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state_dict is not None:
            scaler.load_state_dict(scaler_state_dict)
    except RuntimeError as exc:
        print(f"[LOAD] Checkpoint incompatible with current model: {exc}")
        print("[LOAD] Starting fresh training run with the updated architecture.")
        print("[LOAD] Training mode: fresh-started")
        return 1, 0.0, DEFAULT_THRESHOLD, _build_empty_history(), "fresh-started"

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
    print("[LOAD] Training mode: resumed")
    return next_epoch, best_val_f1, best_threshold, history, "resumed"


def main():
    parser = argparse.ArgumentParser(description="Train Late Fusion AMI model")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples (for quick testing)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override training DataLoader worker count")
    parser.add_argument("--val-workers", type=int, default=None,
                        help="Override validation DataLoader worker count")
    parser.add_argument("--disable-amp", action="store_true",
                        help="Disable CUDA automatic mixed precision")
    parser.add_argument("--weighted-sampling", action="store_true",
                        help="Use a weighted sampler for the training split")
    parser.add_argument("--no-weighted-sampling", action="store_true",
                        help="Disable weighted sampling even if enabled by config")
    parser.add_argument("--loss-mode", choices=["bce", "focal", "hybrid"], default="hybrid",
                        help="Training loss to use")
    parser.add_argument("--hybrid-bce-weight", type=float, default=0.5,
                        help="BCE contribution when --loss-mode=hybrid")
    parser.add_argument("--hybrid-focal-weight", type=float, default=0.5,
                        help="Focal contribution when --loss-mode=hybrid")
    args = parser.parse_args()

    if args.hybrid_bce_weight < 0 or args.hybrid_focal_weight < 0:
        raise ValueError("Hybrid loss weights must be non-negative.")
    if args.loss_mode == "hybrid" and (args.hybrid_bce_weight + args.hybrid_focal_weight) == 0:
        raise ValueError("Hybrid loss weights cannot both be zero.")
    if args.weighted_sampling and args.no_weighted_sampling:
        raise ValueError("Choose either --weighted-sampling or --no-weighted-sampling, not both.")

    seed_everything()

    weighted_sampling = WEIGHTED_SAMPLING_DEFAULT
    if args.weighted_sampling:
        weighted_sampling = True
    if args.no_weighted_sampling:
        weighted_sampling = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = _make_grad_scaler(amp_enabled)
    args.device = device
    print(f"[INFO] Device: {device}")
    print(f"[INFO] AMP: {'enabled' if amp_enabled else 'disabled'}")

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
        weighted_sampling=weighted_sampling,
    )

    print("\n" + "=" * 70)
    print(" MODEL")
    print("=" * 70)
    model = LateFusionModel(
        n_leads=ECG_LEADS,
        ecg_length=ECG_LENGTH,
        n_clinical=NUM_CLINICAL_FEATURES,
        dropout=DROPOUT_RATE,
        cnn_filters=ECG_CNN_FILTERS,
        lstm_hidden_size=ECG_LSTM_HIDDEN_SIZE,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
    ).to(device)

    architecture_metadata = _current_architecture_metadata(model)
    print(f"[MODEL] ECG branch type  : {ECG_BRANCH_TYPE}")
    print(f"[MODEL] Fusion type      : {FUSION_TYPE}")
    print(f"[MODEL] ECG length       : {ECG_LENGTH}")
    print(f"[MODEL] CNN filters      : {ECG_CNN_FILTERS}")
    print(f"[MODEL] LSTM hidden size : {ECG_LSTM_HIDDEN_SIZE}")
    print(f"[MODEL] Fusion hidden dim: {FUSION_HIDDEN_DIM}")
    print(f"[MODEL] Dropout          : {DROPOUT_RATE}")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters : {total_params:,}")
    print(f"[MODEL] Trainable params : {train_params:,}")

    criterion, loss_name = _build_criterion(args, pos_weight)
    print(f"[INFO] Loss: {loss_name}")
    print(
        f"[INFO] Aux supervision: fusion + {ECG_AUX_LOSS_WEIGHT:.2f}*ecg + "
        f"{CLINICAL_AUX_LOSS_WEIGHT:.2f}*clinical"
    )
    print(f"[INFO] Base LR: {LEARNING_RATE:.6f}")
    print(f"[INFO] ECG branch LR multiplier: {ECG_BRANCH_LR_MULTIPLIER:.2f}")
    print(f"[INFO] Weighted sampling: {'enabled' if weighted_sampling else 'disabled'}")
    print(f"[INFO] Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(
        f"[INFO] Hard-negative focus: weight={HARD_NEGATIVE_WEIGHT:.2f} | "
        f"power={HARD_NEGATIVE_POWER:.2f}"
    )
    optimizer = _build_optimizer(model)

    checkpoint_path = _latest_checkpoint_path()
    best_model_path = _best_model_path()
    resumed_from_checkpoint = checkpoint_path.exists()
    start_epoch, best_val_f1, best_threshold, history, training_mode = _resume_from_checkpoint(
        checkpoint_path, model, optimizer, scaler
    )
    resumed_from_checkpoint = training_mode == "resumed"
    active_threshold = best_threshold
    epochs_without_improvement = 0

    print(f"\n{'Epoch':>5} | {'Tr Loss':>8} | {'Val Loss':>8} | "
          f"{'Tr Acc':>7} | {'Val Acc':>7} | {'Val P':>6} | {'Val R':>6} | {'Val F1':>6} | {'Th':>5}")
    print("-" * 86)

    start_time = time.time()
    completed_epochs = max(start_epoch - 1, 0)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            threshold=active_threshold,
            amp_enabled=amp_enabled,
            scaler=scaler,
            ecg_aux_weight=ECG_AUX_LOSS_WEIGHT,
            clinical_aux_weight=CLINICAL_AUX_LOSS_WEIGHT,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            auto_threshold=True,
            amp_enabled=amp_enabled,
            ecg_aux_weight=ECG_AUX_LOSS_WEIGHT,
            clinical_aux_weight=CLINICAL_AUX_LOSS_WEIGHT,
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
        print(
            f"         losses: fusion={val_metrics['fusion_loss']:.4f} | "
            f"ecg={val_metrics['ecg_loss']:.4f} | clinical={val_metrics['clinical_loss']:.4f}"
        )
        print(
            f"         hard-neg weights: mean={val_metrics['hard_negative_weight_mean']:.4f} | "
            f"max={val_metrics['hard_negative_weight_max']:.4f}"
        )

        completed_epochs = epoch
        if (not resumed_from_checkpoint and epoch == 1) or (val_metrics["f1"] > best_val_f1):
            best_val_f1 = val_metrics["f1"]
            best_threshold = active_threshold
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
            print(
                f"         ^ New best Val F1 = {best_val_f1:.4f} at threshold {best_threshold:.4f}"
                " -- best model updated"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"         no val F1 improvement for {epochs_without_improvement} epoch(s)"
            )

        _save_checkpoint(
            checkpoint_path, model, optimizer, scaler, epoch, best_val_f1, best_threshold, history, args
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"[STOP] Early stopping triggered after {EARLY_STOPPING_PATIENCE} "
                "epochs without validation F1 improvement."
            )
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
        model,
        val_loader,
        criterion,
        device,
        threshold=best_threshold,
        return_outputs=True,
        amp_enabled=amp_enabled,
        ecg_aux_weight=ECG_AUX_LOSS_WEIGHT,
        clinical_aux_weight=CLINICAL_AUX_LOSS_WEIGHT,
    )
    final_train_loss, final_train_metrics = evaluate(
        model,
        train_loader,
        criterion,
        device,
        threshold=best_threshold,
        amp_enabled=amp_enabled,
        ecg_aux_weight=ECG_AUX_LOSS_WEIGHT,
        clinical_aux_weight=CLINICAL_AUX_LOSS_WEIGHT,
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
        "ecg_branch_learning_rate": LEARNING_RATE * ECG_BRANCH_LR_MULTIPLIER,
        "ecg_aux_loss_weight": ECG_AUX_LOSS_WEIGHT,
        "clinical_aux_loss_weight": CLINICAL_AUX_LOSS_WEIGHT,
        "weighted_sampling": weighted_sampling,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
        "hard_negative_power": HARD_NEGATIVE_POWER,
        "loss_name": loss_name,
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
                "fusion_loss", "ecg_loss", "clinical_loss",
                "hard_negative_weight_mean", "hard_negative_weight_max",
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
