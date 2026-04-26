import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import (
    DEFAULT_THRESHOLD,
    DROPOUT_RATE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    ECG_LEADS,
    ECG_LENGTH,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    LEARNING_RATE,
    LOSS_NAME,
    METRICS_DIR,
    MODELS_DIR,
    NUM_CLINICAL_FEATURES,
    NUM_EPOCHS,
    SEED,
    TEST_SPLIT,
    VAL_NUM_WORKERS,
    WEIGHT_DECAY,
)
from .dataset import make_split_dataloaders, prepare_full_dataset
from .engine import evaluate, train_one_epoch
from .losses import build_loss
from .model import EarlyFusionModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _resolve_metrics_dir(run_name: str | None) -> Path:
    if not run_name:
        return METRICS_DIR / "cross_validation"
    return MODELS_DIR.parent / "runs" / run_name / "metrics"


def main():
    parser = argparse.ArgumentParser(description="Cross-validation for Early Fusion AMI model")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of stratified folds")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Max epochs per fold")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE,
                        help="Model dropout")
    parser.add_argument("--loss-name", choices=["focal", "bce"], default=LOSS_NAME,
                        help="Loss function to use")
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_LOSS_ALPHA,
                        help="Alpha parameter for focal loss")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_LOSS_GAMMA,
                        help="Gamma parameter for focal loss")
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help="Stop after this many non-improving epochs; set 0 to disable")
    parser.add_argument("--early-stopping-min-delta", type=float, default=EARLY_STOPPING_MIN_DELTA,
                        help="Minimum F1 improvement to count as better")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override training DataLoader worker count")
    parser.add_argument("--val-workers", type=int, default=VAL_NUM_WORKERS,
                        help="Override validation DataLoader worker count")
    parser.add_argument("--weighted-sampling", action="store_true",
                        help="Use a weighted sampler for training folds")
    parser.add_argument("--run-name", type=str, default="crossval",
                        help="Artifacts subdirectory under artifacts/runs")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_dir = _resolve_metrics_dir(args.run_name)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Cross-validation metrics dir: {metrics_dir}")

    prepared = prepare_full_dataset(subset=args.subset)
    labels = prepared["labels"]
    indices = np.arange(len(labels))

    train_pool_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=SEED, stratify=labels
    )
    pool_labels = labels[train_pool_idx]
    print(f"[DATA] Held-out test size kept fixed at {len(test_idx):,} samples")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    rows = []

    for fold, (train_sub_idx, val_sub_idx) in enumerate(
        skf.split(np.zeros(len(train_pool_idx)), pool_labels),
        start=1,
    ):
        seed_everything(SEED + fold)
        print("\n" + "=" * 70)
        print(f" FOLD {fold}/{args.folds}")
        print("=" * 70)

        train_idx = train_pool_idx[train_sub_idx]
        val_idx = train_pool_idx[val_sub_idx]
        split_data = make_split_dataloaders(
            prepared,
            train_idx=train_idx,
            val_idx=val_idx,
            train_num_workers=args.num_workers,
            val_num_workers=args.val_workers,
            augment_train=True,
            weighted_sampling=args.weighted_sampling,
        )

        model = EarlyFusionModel(
            n_leads=ECG_LEADS,
            ecg_length=ECG_LENGTH,
            n_clinical=NUM_CLINICAL_FEATURES,
            dropout=args.dropout,
        ).to(device)
        criterion = build_loss(
            args.loss_name,
            pos_weight=split_data["pos_weight"].to(device),
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_val_f1 = 0.0
        best_epoch = 0
        best_threshold = DEFAULT_THRESHOLD
        active_threshold = DEFAULT_THRESHOLD
        epochs_without_improvement = 0
        best_state_dict = copy.deepcopy(model.state_dict())

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_one_epoch(
                model,
                split_data["train_loader"],
                criterion,
                optimizer,
                device,
                threshold=active_threshold,
                scaler=scaler,
            )
            val_loss, val_metrics = evaluate(
                model,
                split_data["val_loader"],
                criterion,
                device,
                auto_threshold=True,
            )
            active_threshold = val_metrics["threshold"]

            print(
                f"{epoch:3d} | tr_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"tr_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} th={active_threshold:.2f}"
            )

            if val_metrics["f1"] > best_val_f1 + args.early_stopping_min_delta:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_threshold = active_threshold
                epochs_without_improvement = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            scheduler.step()

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"[STOP] Early stopping at epoch {epoch} for fold {fold}")
                break

        model.load_state_dict(best_state_dict)
        final_val_loss, final_val_metrics = evaluate(
            model,
            split_data["val_loader"],
            criterion,
            device,
            threshold=best_threshold,
        )

        row = {
            "fold": fold,
            "best_epoch": best_epoch,
            "val_loss": round(final_val_loss, 6),
            "accuracy": round(final_val_metrics["accuracy"], 6),
            "precision": round(final_val_metrics["precision"], 6),
            "recall": round(final_val_metrics["recall"], 6),
            "f1": round(final_val_metrics["f1"], 6),
            "auc": round(final_val_metrics["auc"], 6),
            "average_precision": round(final_val_metrics["average_precision"], 6),
            "threshold": round(best_threshold, 6),
        }
        rows.append(row)

        print(
            f"[FOLD {fold}] best_epoch={best_epoch} f1={row['f1']:.4f} "
            f"auc={row['auc']:.4f} ap={row['average_precision']:.4f}"
        )

    summary = {
        "folds": args.folds,
        "epochs": args.epochs,
        "loss_name": args.loss_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "weighted_sampling": args.weighted_sampling,
        "focal_alpha": args.focal_alpha if args.loss_name == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss_name == "focal" else None,
        "mean_f1": round(float(np.mean([row["f1"] for row in rows])), 6),
        "std_f1": round(float(np.std([row["f1"] for row in rows])), 6),
        "mean_auc": round(float(np.mean([row["auc"] for row in rows])), 6),
        "std_auc": round(float(np.std([row["auc"] for row in rows])), 6),
        "mean_average_precision": round(float(np.mean([row["average_precision"] for row in rows])), 6),
        "std_average_precision": round(float(np.std([row["average_precision"] for row in rows])), 6),
        "rows": rows,
    }

    csv_path = metrics_dir / "cross_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = metrics_dir / "cross_validation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Mean F1: {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}")
    print(f"Mean AUC: {summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f}")
    print(f"Mean AP : {summary['mean_average_precision']:.4f} +/- {summary['std_average_precision']:.4f}")
    print(f"[SAVE] CSV  -> {csv_path}")
    print(f"[SAVE] JSON -> {json_path}")


if __name__ == "__main__":
    main()
