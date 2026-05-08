# ==========================================
# PLOTS — Training Metric Visualisations
# ==========================================

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .config import PLOTS_DIR


def _base_axes(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_loss_plot(history: dict, save_path: str = None):
    """Plot and save training vs validation loss curves."""
    if save_path is None:
        save_path = str(PLOTS_DIR / "loss.png")

    fig, ax = _base_axes("Training vs Validation Loss", "BCE Loss")
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "o-", color="#4361EE", linewidth=2, markersize=4, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "s-", color="#F72585", linewidth=2, markersize=4, label="Val Loss")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Loss plot saved -> {save_path}")


def save_accuracy_plot(history: dict, save_path: str = None):
    """Plot and save training vs validation accuracy curves."""
    if save_path is None:
        save_path = str(PLOTS_DIR / "accuracy.png")

    fig, ax = _base_axes("Training vs Validation Accuracy", "Accuracy")
    epochs = range(1, len(history["train_acc"]) + 1)

    ax.plot(epochs, history["train_acc"], "o-", color="#3A0CA3", linewidth=2, markersize=4, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"],   "s-", color="#F77F00", linewidth=2, markersize=4, label="Val Accuracy")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Accuracy plot saved -> {save_path}")


def save_metric_plot(history: dict, metric_key: str, label: str, save_path: str = None):
    """Plot a train/validation metric curve across epochs."""
    if save_path is None:
        save_path = str(PLOTS_DIR / f"{metric_key}.png")

    train_key = f"train_{metric_key}"
    val_key = f"val_{metric_key}"
    epochs = range(1, len(history[train_key]) + 1)

    fig, ax = _base_axes(f"Training vs Validation {label}", label)
    ax.plot(epochs, history[train_key], "o-", color="#0F766E", linewidth=2, markersize=4, label=f"Train {label}")
    ax.plot(epochs, history[val_key],   "s-", color="#DC2626", linewidth=2, markersize=4, label=f"Val {label}")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {label} plot saved -> {save_path}")


def save_confusion_matrix_plot(labels, preds, save_path: str = None):
    """Save a 2x2 confusion matrix heatmap."""
    if save_path is None:
        save_path = str(PLOTS_DIR / "confusion_matrix.png")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=["Pred Non-AMI", "Pred AMI"])
    ax.set_yticks([0, 1], labels=["True Non-AMI", "True AMI"])
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > cm.max() / 2 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Confusion matrix saved -> {save_path}")


def save_roc_curve_plot(labels, probs, auc_score: float, save_path: str = None):
    """Save ROC curve."""
    if save_path is None:
        save_path = str(PLOTS_DIR / "roc_curve.png")

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] ROC curve saved -> {save_path}")


def save_pr_curve_plot(labels, probs, average_precision: float, save_path: str = None):
    """Save precision-recall curve."""
    if save_path is None:
        save_path = str(PLOTS_DIR / "pr_curve.png")

    precision, recall, _ = precision_recall_curve(labels, probs)
    positive_rate = float(np.mean(labels)) if len(labels) else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#9333EA", linewidth=2.5, label=f"PR Curve (AP = {average_precision:.3f})")
    ax.axhline(y=positive_rate, linestyle="--", color="#6B7280", label=f"Positive Rate = {positive_rate:.3f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] PR curve saved -> {save_path}")


def save_model_comparison_table(rows, save_path: str = None):
    """Save a compact comparison table as a PNG figure."""
    if save_path is None:
        save_path = str(PLOTS_DIR / "model_comparison_table.png")

    columns = ["Split", "Loss", "Accuracy", "Precision", "Recall", "F1", "AUC", "AP"]
    cell_text = []
    for row in rows:
        cell_text.append([
            row["split"],
            f"{row['loss']:.4f}",
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['auc']:.4f}",
            f"{row['average_precision']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1D4ED8")
        else:
            cell.set_facecolor("#EFF6FF" if row % 2 else "#DBEAFE")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Comparison table saved -> {save_path}")


def save_optuna_history_plot(trials_df: pd.DataFrame, save_path: str):
    """Save trial-by-trial optimization history."""
    completed = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if completed.empty:
        print(f"[PLOT] No completed Optuna trials available for {save_path}")
        return

    completed = completed.sort_values("number")
    best_so_far = completed["value"].cummax()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(completed["number"], completed["value"], "o-", color="#2563EB", linewidth=2, label="Trial F1")
    ax.plot(completed["number"], best_so_far, "-", color="#DC2626", linewidth=2, label="Best So Far")
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Validation F1", fontsize=12)
    ax.set_title("Optuna Optimization History", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Optuna history saved -> {save_path}")


def save_optuna_param_importance_plot(importances: dict[str, float], save_path: str):
    """Save parameter importance bars from Optuna."""
    if not importances:
        print(f"[PLOT] No Optuna importances available for {save_path}")
        return

    items = sorted(importances.items(), key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    ax.barh(labels, values, color="#7C3AED")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Optuna Parameter Importances", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Optuna importances saved -> {save_path}")


def save_optuna_param_slices_plot(trials_df: pd.DataFrame, params: list[str], save_path: str):
    """Save simple parameter-vs-score scatter slices for completed trials."""
    completed = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if completed.empty or not params:
        print(f"[PLOT] No completed Optuna trials available for {save_path}")
        return

    n_params = len(params)
    ncols = 2
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(4, 4 * nrows)))
    axes = np.atleast_1d(axes).flatten()

    for ax, param in zip(axes, params):
        col = f"params_{param}"
        if col not in completed.columns:
            ax.axis("off")
            continue
        ax.scatter(completed[col], completed["value"], c=completed["number"], cmap="viridis", s=35, alpha=0.85)
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Validation F1", fontsize=11)
        ax.set_title(param, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.suptitle("Optuna Parameter Slices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Optuna parameter slices saved -> {save_path}")
