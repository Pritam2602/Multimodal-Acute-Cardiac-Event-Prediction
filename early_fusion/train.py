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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.optim as optim

from .config import (
    SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    ECG_LEADS, ECG_LENGTH, CLINICAL_FEATURES, NUM_CLINICAL_FEATURES, DROPOUT_RATE,
    DEFAULT_THRESHOLD, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, LOSS_NAME,
    LABEL_SMOOTHING, SCHEDULER_TYPE,
    ECG_TIME_WINDOW_HOURS, MAX_ECGS_PER_PATIENT,
)
from .dataset import load_and_prepare_data
from .engine import train_one_epoch, evaluate
from .losses import build_loss
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


def _resolve_artifact_dirs(run_name: str | None) -> tuple[Path, Path, Path]:
    if not run_name:
        return MODELS_DIR, PLOTS_DIR, METRICS_DIR

    run_root = MODELS_DIR.parent / "runs" / run_name
    return run_root / "models", run_root / "plots", run_root / "metrics"


def _parse_excluded_features(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _latest_checkpoint_path_for(models_dir: Path) -> Path:
    return models_dir / "latest_checkpoint.pth"


def _best_model_path_for(models_dir: Path) -> Path:
    return models_dir / "early_fusion_model.pth"


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_f1: float,
    best_epoch: int,
    best_threshold: float,
    epochs_without_improvement: int,
    history: dict,
    args: argparse.Namespace,
):
    payload = {
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "epochs_without_improvement": epochs_without_improvement,
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
        return 1, 0.0, 0, DEFAULT_THRESHOLD, 0, _build_empty_history()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError as exc:
        print(f"[LOAD] Checkpoint incompatible with current model: {exc}")
        print("[LOAD] Starting fresh training run with the updated architecture.")
        return 1, 0.0, 0, DEFAULT_THRESHOLD, 0, _build_empty_history()

    # Restore scheduler state if available
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception:
            pass  # scheduler config may have changed; let it re-initialize

    history = checkpoint.get("history", _build_empty_history())
    best_val_f1 = float(checkpoint.get("best_val_f1", 0.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best_threshold = float(checkpoint.get("best_threshold", DEFAULT_THRESHOLD))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
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
    if best_epoch:
        print(f"[LOAD] Best epoch so far: {best_epoch}")
    print(f"[LOAD] Best threshold so far: {best_threshold:.4f}")
    return next_epoch, best_val_f1, best_epoch, best_threshold, epochs_without_improvement, history


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
    parser.add_argument("--weighted-sampling", action="store_true",
                        help="Use a weighted sampler for training batches")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Total training epochs")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE,
                        help="Dropout rate for the model")
    parser.add_argument("--loss-name", choices=["focal", "bce", "ohem_focal"], default=LOSS_NAME,
                        help="Loss function to use")
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_LOSS_ALPHA,
                        help="Alpha parameter for focal loss")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_LOSS_GAMMA,
                        help="Gamma parameter for focal loss")
    parser.add_argument("--pos-weight-scale", type=float, default=1.0,
                        help="Multiply the training positive-class weight by this factor")
    parser.add_argument("--exclude-clinical-features", type=str, default=None,
                        help="Comma-separated clinical features to omit for an ablation run")
    parser.add_argument("--model-arch", type=str, default="baseline",
                        choices=["baseline", "resnet", "filmed", "crossattn", "clinical_gated", "clinical_only", "siamese_delta", "siamese_crossattn"],
                        help="Neural network architecture to use.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup alpha (0=off, 0.4=recommended). Interpolates training samples.")
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help="Stop after this many non-improving epochs; set 0 to disable")
    parser.add_argument("--early-stopping-min-delta", type=float, default=EARLY_STOPPING_MIN_DELTA,
                        help="Minimum F1 improvement to reset early stopping")
    parser.add_argument("--auto-continue", action="store_true",
                        help="Run through epochs without interactive prompts")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional subdirectory name under artifacts/runs for this experiment")
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING,
                        help="Label smoothing factor (0=off, 0.05=default)")
    parser.add_argument("--scheduler", choices=["onecycle", "cosine", "cosine_warmup"], default=SCHEDULER_TYPE,
                        help="LR scheduler type")
    parser.add_argument("--entropy-lambda", type=float, default=0.0,
                        help="Multiplier for the Attention Entropy Constraint penalty (e.g. 0.1)")
    parser.add_argument("--contrastive-lambda", type=float, default=0.0,
                        help="Multiplier for False-Positive Adversarial Separation loss (e.g. 0.5)")
    parser.add_argument("--ecg-time-window", type=int, default=ECG_TIME_WINDOW_HOURS,
                        help="Keep ECGs within ±N hours of admission (0=off, recommended=72)")
    parser.add_argument("--max-ecgs-per-patient", type=int, default=MAX_ECGS_PER_PATIENT,
                        help="Cap repeated ECGs per patient (0=off, recommended=3)")
    args = parser.parse_args()

    seed_everything()

    excluded_features = _parse_excluded_features(args.exclude_clinical_features)
    unknown_excluded = [feat for feat in excluded_features if feat not in CLINICAL_FEATURES and feat != "ALL"]
    if unknown_excluded:
        raise ValueError(f"Unknown --exclude-clinical-features values: {unknown_excluded}")
    if "ALL" in excluded_features:
        selected_clinical_features = []
    else:
        selected_clinical_features = [feat for feat in CLINICAL_FEATURES if feat not in set(excluded_features)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    models_dir, plots_dir, metrics_dir = _resolve_artifact_dirs(args.run_name)

    print(
        f"[RUN] epochs={args.epochs} lr={args.learning_rate} wd={args.weight_decay} "
        f"dropout={args.dropout} loss={args.loss_name} weighted_sampling={args.weighted_sampling} "
        f"model_arch={args.model_arch}"
    )
    if args.loss_name == "focal":
        print(f"[RUN] focal_alpha={args.focal_alpha} focal_gamma={args.focal_gamma}")
    print(f"[RUN] pos_weight_scale={args.pos_weight_scale}")
    if excluded_features:
        print(
            f"[RUN] excluded_clinical_features={excluded_features} | "
            f"n_clinical={len(selected_clinical_features)}/{NUM_CLINICAL_FEATURES}"
        )
    print(
        f"[RUN] early_stopping_patience={args.early_stopping_patience} "
        f"min_delta={args.early_stopping_min_delta}"
    )
    print(f"[RUN] scheduler={args.scheduler} label_smoothing={args.label_smoothing}")
    print(f"[RUN] ecg_time_window={args.ecg_time_window}h max_ecgs_per_patient={args.max_ecgs_per_patient}")

    for directory in [models_dir, plots_dir, metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {directory}")

    print("\n" + "=" * 70)
    print(" DATA LOADING")
    print("=" * 70)
    train_loader, val_loader, pos_weight, _test_metadata = load_and_prepare_data(
        subset=args.subset,
        train_num_workers=args.num_workers,
        val_num_workers=args.val_workers,
        weighted_sampling=args.weighted_sampling,
        clinical_features=selected_clinical_features,
        ecg_time_window=args.ecg_time_window,
        max_ecgs_per_patient=args.max_ecgs_per_patient,
    )

    print("\n" + "=" * 70)
    print(" MODEL")
    print("=" * 70)
    model = EarlyFusionModel(
        n_leads=ECG_LEADS,
        ecg_length=ECG_LENGTH,
        n_clinical=len(selected_clinical_features),
        dropout=args.dropout,
        extractor_arch=args.model_arch,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters : {total_params:,}")
    print(f"[MODEL] Trainable params : {train_params:,}")

    raw_pos_weight = pos_weight.clone()
    pos_weight = pos_weight * args.pos_weight_scale
    print(
        f"[LOSS] raw_pos_weight={raw_pos_weight.item():.4f} | "
        f"effective_pos_weight={pos_weight.item():.4f}"
    )

    criterion = build_loss(
        args.loss_name,
        pos_weight=pos_weight.to(device),
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # ── Scheduler ────────────────────────────────────────────────────────
    # OneCycleLR steps per-batch inside train_one_epoch (warmup → peak → decay).
    # CosineAnnealing steps per-epoch after validation (legacy behaviour).
    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1000,
        )
    elif args.scheduler == "cosine_warmup":
        warmup_epochs = 3
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    checkpoint_path = _latest_checkpoint_path_for(models_dir)
    best_model_path = _best_model_path_for(models_dir)
    resumed_from_checkpoint = checkpoint_path.exists()
    start_epoch, best_val_f1, best_epoch, best_threshold, epochs_without_improvement, history = _resume_from_checkpoint(
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

    stop_reason = "completed"

    # OneCycleLR is passed into engine for per-batch stepping; cosine stays epoch-level
    batch_scheduler = scheduler if args.scheduler == "onecycle" else None

    ema_avg = get_ema_multi_avg_fn(0.99)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            threshold=active_threshold, scaler=scaler,
            label_smoothing=args.label_smoothing,
            scheduler=batch_scheduler,
            mixup_alpha=args.mixup_alpha,
            entropy_lambda=args.entropy_lambda,
            contrastive_lambda=args.contrastive_lambda
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
        improved = ((not resumed_from_checkpoint and epoch == 1) or
                    (val_metrics["f1"] > best_val_f1 + args.early_stopping_min_delta))

        if improved:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_threshold = active_threshold
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"         ^ New best Val F1 = {best_val_f1:.4f} at threshold {best_threshold:.4f}"
                " -- best model updated"
            )
        else:
            epochs_without_improvement += 1

        # Only step epoch-level schedulers here (OneCycleLR steps per-batch in engine)
        if args.scheduler != "onecycle":
            scheduler.step()

        ema_model.update_parameters(model)

        _save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_f1,
            best_epoch,
            best_threshold,
            epochs_without_improvement,
            history,
            args,
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"[STOP] Early stopping triggered at epoch {epoch} "
                f"(best epoch: {best_epoch}, best Val F1: {best_val_f1:.4f})."
            )
            break

        if args.auto_continue:
            continue

        if epoch < args.epochs and not _should_continue(epoch + 1, args.epochs):
            stop_reason = "user_stop"
            print(f"[STOP] Training paused after epoch {epoch}. Rerun the script to resume from epoch {epoch + 1}.")
            break

    elapsed = time.time() - start_time
    print("-" * 86)
    print(
        f"[DONE] Training session finished in {elapsed:.1f}s  |  "
        f"Completed epochs: {completed_epochs} | Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f} | "
        f"Best threshold: {best_threshold:.4f}"
    )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"[LOAD] Best model restored from {best_model_path}")

    ema_model_path = models_dir / "ema_model.pth"
    torch.save(ema_model.module.state_dict(), ema_model_path)
    print(f"[SAVE] Final EMA model saved to {ema_model_path}")

    print("\n--- Evaluating Best Model ---")
    final_val_loss, final_val_metrics, final_val_outputs = evaluate(
        model, val_loader, criterion, device, threshold=best_threshold, return_outputs=True
    )
    final_train_loss, final_train_metrics = evaluate(
        model, train_loader, criterion, device, threshold=best_threshold
    )

    print("\n--- Evaluating EMA Model ---")
    ema_val_loss, ema_val_metrics = evaluate(
        ema_model, val_loader, criterion, device, auto_threshold=True
    )
    print(f"[EMA] Val F1: {ema_val_metrics['f1']:.4f} @ Threshold: {ema_val_metrics['threshold']:.4f}")

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
        "best_epoch": best_epoch,
        "best_threshold": round(best_threshold, 6),
        "epochs_completed": completed_epochs,
        "target_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "model_arch": args.model_arch,
        "loss_name": args.loss_name,
        "weighted_sampling": args.weighted_sampling,
        "clinical_features": selected_clinical_features,
        "excluded_clinical_features": excluded_features,
        "pos_weight": round(float(raw_pos_weight.item()), 6),
        "pos_weight_scale": args.pos_weight_scale,
        "effective_pos_weight": round(float(pos_weight.item()), 6),
        "focal_alpha": args.focal_alpha if args.loss_name == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss_name == "focal" else None,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "ecg_time_window": args.ecg_time_window,
        "max_ecgs_per_patient": args.max_ecgs_per_patient,
        "stopping_reason": stop_reason,
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
        "validation_outputs": {
            "labels": [int(v) for v in final_val_outputs["labels"]],
            "preds": [int(v) for v in final_val_outputs["preds"]],
            "probs": [round(float(v), 6) for v in final_val_outputs["probs"]],
        },
    }

    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[SAVE] Metrics -> {metrics_path}")

    comparison_rows = [
        {"split": "Train", "loss": final_train_loss, **final_train_metrics},
        {"split": "Validation", "loss": final_val_loss, **final_val_metrics},
    ]

    comparison_csv_path = metrics_dir / "comparison_table.csv"
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

    save_loss_plot(history, str(plots_dir / "loss.png"))
    save_accuracy_plot(history, str(plots_dir / "accuracy.png"))
    save_metric_plot(history, "precision", "Precision", str(plots_dir / "precision.png"))
    save_metric_plot(history, "recall", "Recall", str(plots_dir / "recall.png"))
    save_metric_plot(history, "f1", "F1 Score", str(plots_dir / "f1.png"))
    save_confusion_matrix_plot(
        final_val_outputs["labels"],
        final_val_outputs["preds"],
        str(plots_dir / "confusion_matrix.png"),
    )
    save_roc_curve_plot(
        final_val_outputs["labels"],
        final_val_outputs["probs"],
        final_val_metrics["auc"],
        str(plots_dir / "roc_curve.png"),
    )
    save_pr_curve_plot(
        final_val_outputs["labels"],
        final_val_outputs["probs"],
        final_val_metrics["average_precision"],
        str(plots_dir / "pr_curve.png"),
    )
    save_model_comparison_table(comparison_rows, str(plots_dir / "model_comparison_table.png"))

    print("\n" + "=" * 70)
    print(" ARTIFACTS SAVED")
    print("=" * 70)
    print(f"  Best model : {best_model_path}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {metrics_path}")
    print(f"               {comparison_csv_path}")
    print(f"  Plots      : {plots_dir / 'loss.png'}")
    print(f"               {plots_dir / 'accuracy.png'}")
    print(f"               {plots_dir / 'precision.png'}")
    print(f"               {plots_dir / 'recall.png'}")
    print(f"               {plots_dir / 'f1.png'}")
    print(f"               {plots_dir / 'confusion_matrix.png'}")
    print(f"               {plots_dir / 'roc_curve.png'}")
    print(f"               {plots_dir / 'pr_curve.png'}")
    print(f"               {plots_dir / 'model_comparison_table.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
