import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve

from early_fusion.config import (
    SEED, ARTIFACT_ROOT, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, DROPOUT_RATE, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)
from early_fusion.temporal_dataset import load_and_prepare_temporal_data
from .temporal_model import TemporalHybridLateFusion
from .temporal_engine import train_one_epoch, evaluate
from early_fusion.losses import FocalLoss, OHEMFocalLoss

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(labels, preds, output_path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control/Chronic", "Acute MI"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--contrastive-lambda", type=float, default=0.1)
    parser.add_argument("--entropy-lambda", type=float, default=0.001)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    run_dir = ARTIFACT_ROOT / "runs" / args.run_name
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    metrics_dir = run_dir / "metrics"
    
    for d in [models_dir, plots_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, pos_weight, n_clinical = load_and_prepare_temporal_data(
        batch_size=args.batch_size, train_workers=0, val_workers=0
    )

    model = TemporalHybridLateFusion(
        n_leads=12,
        ecg_length=5000,
        n_static_clinical=77,
        n_dynamic_clinical=3,
        dropout=DROPOUT_RATE
    ).to(device)

    # Loss - Clamp pos_weight and soften focal loss for better calibration
    effective_pos_weight = min(pos_weight, 4.0)
    print(f"[TRAIN] Using Effective Positive Weight: {effective_pos_weight:.2f}")
    pos_weight_tensor = torch.tensor([effective_pos_weight], device=device, dtype=torch.float32)
    criterion = FocalLoss(alpha=0.25, gamma=1.5, pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_f1 = 0.0
    best_physio_score = 0.0
    epochs_no_improve = 0
    EARLY_STOPPING_PATIENCE = 3 # Aggressive early stopping
    history = {"train": [], "val": []}

    print("\n[TRAIN] Starting Temporal Phase 5 Training...")
    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        
        # Entropy Band Regularization is safe to leave on for all epochs
        current_entropy_lambda = args.entropy_lambda
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, scheduler=scheduler,
            contrastive_lambda=args.contrastive_lambda,
            entropy_lambda=current_entropy_lambda
        )

        val_metrics, val_probs, val_labels, val_preds, _ = evaluate(
            model, val_loader, criterion, device
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Epoch [{epoch:2d}/{args.epochs}]  "
              f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}  --  "
              f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | Threshold: {val_metrics['threshold']:.3f}")
              
        if "attn_entropy" in val_metrics:
            entropy = val_metrics['attn_entropy']
            dominance = val_metrics['attn_dominance']
            sparsity = val_metrics['attn_sparsity']
            
            bio_str = f"  -> Attn Bio: Temp_Entropy={entropy:.3f} | Dominance={dominance:.3f} | Sparsity={sparsity:.3f}"
            if "lead_entropy" in val_metrics:
                bio_str += f" | Lead_Entropy={val_metrics['lead_entropy']:.3f}"
            print(bio_str)
            
            # Physiological Checkpointing
            if (0.45 <= entropy <= 0.65) and (0.6 <= dominance <= 0.75) and (val_metrics['auc'] > 0.91):
                # Calculate a custom physiological score based on F1 + Health
                physio_score = val_metrics['f1'] + (0.65 - abs(0.55 - entropy)) # reward center of entropy range
                if physio_score > best_physio_score:
                    best_physio_score = physio_score
                    torch.save(model.state_dict(), models_dir / "best_physiological_model.pth")
                    print(f"  -> Saved new BEST PHYSIOLOGICAL model (F1: {val_metrics['f1']:.4f})")

        # Checkpointing (Standard F1)
        if val_metrics["f1"] > best_val_f1 + EARLY_STOPPING_MIN_DELTA:
            best_val_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            plot_confusion_matrix(val_labels, val_preds, plots_dir / "best_val_confusion_matrix.png")
            
            metrics_summary = {
                "train": train_metrics,
                "val": val_metrics,
                "best_val_f1": best_val_f1,
                "best_epoch": epoch
            }
            save_metrics(metrics_summary, metrics_dir / "metrics.json")
            print(f"  -> Saved new best model (F1: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[TRAIN] Early stopping triggered at epoch {epoch}")
                with open(metrics_dir / "history.json", "w") as f:
                    import json
                    json.dump(history, f, indent=4)
                break

        with open(metrics_dir / "history.json", "w") as f:
            import json
            json.dump(history, f, indent=4)

    print("============================================================")
    print(f"Training Complete. Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
