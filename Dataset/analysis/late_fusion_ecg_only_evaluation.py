import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from late_fusion.config import ECG_LENGTH, ECG_LEADS, METRICS_DIR, MODELS_DIR, NUM_CLINICAL_FEATURES
from late_fusion.dataset import load_and_prepare_data
from late_fusion.engine import _compute_metrics, find_best_threshold
from late_fusion.model import LateFusionModel


DEFAULT_OUTPUT_DIR = METRICS_DIR / "ecg_only_eval"
DEFAULT_CHECKPOINT = MODELS_DIR / "latest_checkpoint.pth"
DEFAULT_WEIGHTS = MODELS_DIR / "late_fusion_model.pth"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class LegacyECGBranch(nn.Module):
    def __init__(self, n_leads: int, ecg_length: int, dropout: float):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, ecg_length)
            conv_out = self.conv_block(dummy)
            lstm_input_size = conv_out.shape[1]
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(128, 1)

    def forward(self, ecg):
        x = self.conv_block(ecg)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        features = self.encoder_head(x)
        return self.logit_head(features).squeeze(-1)


class LegacyClinicalBranch(nn.Module):
    def __init__(self, n_clinical: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.logit_head = nn.Linear(32, 1)

    def forward(self, clinical):
        features = self.encoder(clinical)
        return self.logit_head(features).squeeze(-1)


class LegacyLateFusionModel(nn.Module):
    def __init__(self, n_leads: int = ECG_LEADS, ecg_length: int = ECG_LENGTH, n_clinical: int = NUM_CLINICAL_FEATURES):
        super().__init__()
        self.ecg_branch = LegacyECGBranch(n_leads=n_leads, ecg_length=ecg_length, dropout=0.4)
        self.clinical_branch = LegacyClinicalBranch(n_clinical=n_clinical, dropout=0.4)
        self.fusion_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(8, 1),
        )

    def forward(self, ecg, clinical):
        ecg_logit = self.ecg_branch(ecg)
        clinical_logit = self.clinical_branch(clinical)
        return self.fusion_head(torch.stack([ecg_logit, clinical_logit], dim=1)).squeeze(-1)


def _is_legacy_state_dict(state_dict: dict) -> bool:
    return "ecg_branch.attention.weight" not in state_dict


def _load_model(device: torch.device):
    if DEFAULT_CHECKPOINT.exists():
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is not None:
            model_cls = LegacyLateFusionModel if _is_legacy_state_dict(state_dict) else LateFusionModel
            model = model_cls(
                n_leads=ECG_LEADS,
                ecg_length=ECG_LENGTH,
                n_clinical=NUM_CLINICAL_FEATURES,
            ).to(device)
            model.load_state_dict(state_dict)
            return model

    if DEFAULT_WEIGHTS.exists():
        state_dict = torch.load(DEFAULT_WEIGHTS, map_location=device, weights_only=True)
        model_cls = LegacyLateFusionModel if _is_legacy_state_dict(state_dict) else LateFusionModel
        model = model_cls(
            n_leads=ECG_LEADS,
            ecg_length=ECG_LENGTH,
            n_clinical=NUM_CLINICAL_FEATURES,
        ).to(device)
        model.load_state_dict(state_dict)
        return model

    raise FileNotFoundError("No late-fusion model weights found.")


@torch.no_grad()
def _collect_outputs(model, loader, device, max_attention_plots: int):
    model.eval()
    labels = []
    probs = []
    attention_examples = []
    activation_sums = {}
    activation_counts = {}

    hooks = []

    def hook_factory(name):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            stats = torch.tensor(
                [
                    tensor.mean().item(),
                    tensor.std().item(),
                    tensor.min().item(),
                    tensor.max().item(),
                ],
                dtype=torch.float64,
            )
            activation_sums[name] = activation_sums.get(name, torch.zeros(4, dtype=torch.float64)) + stats
            activation_counts[name] = activation_counts.get(name, 0) + 1
        return _hook

    if hasattr(model.ecg_branch, "attention"):
        modules_to_hook = {
            "lead_temporal_stem": model.ecg_branch.lead_temporal_stem[0],
            "cross_lead_mixer": model.ecg_branch.cross_lead_mixer[0],
            "res_stage2": model.ecg_branch.res_stage2[-1],
            "res_stage4": model.ecg_branch.res_stage4[-1],
        }
    else:
        modules_to_hook = {
            "early_conv": model.ecg_branch.conv_block[0],
            "mid_conv": model.ecg_branch.conv_block[4],
            "late_conv": model.ecg_branch.conv_block[8],
        }
    for name, module in modules_to_hook.items():
        hooks.append(module.register_forward_hook(hook_factory(name)))

    for ecg, clinical, batch_labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        if hasattr(model.ecg_branch, "attention"):
            logits, debug = model.ecg_branch(ecg, return_debug=True)
        else:
            logits = model.ecg_branch(ecg)
            debug = None
        batch_probs = torch.sigmoid(logits).cpu().numpy().astype(float)
        probs.extend(batch_probs)
        labels.extend(batch_labels.numpy().astype(int))

        if debug is not None and len(attention_examples) < max_attention_plots:
            attention = debug["attention_weights"].cpu().numpy()
            ecg_np = ecg.cpu().numpy()
            remaining = max_attention_plots - len(attention_examples)
            for idx in range(min(remaining, ecg_np.shape[0])):
                attention_examples.append(
                    {
                        "label": int(batch_labels[idx].item()),
                        "signal": ecg_np[idx],
                        "attention": attention[idx],
                    }
                )

    for hook in hooks:
        hook.remove()

    activation_stats = {}
    for name, total in activation_sums.items():
        count = activation_counts[name]
        activation_stats[name] = {
            "mean": float(total[0] / count),
            "std": float(total[1] / count),
            "min": float(total[2] / count),
            "max": float(total[3] / count),
        }

    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float), attention_examples, activation_stats


def _plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("ECG-only Confusion Matrix", fontsize=13, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_probability_hist(labels: np.ndarray, probs: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, label="Negative", color="#2563EB")
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, label="Positive", color="#DC2626")
    ax.set_title("ECG-only Probability Histogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(labels: np.ndarray, probs: np.ndarray, save_path: Path):
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="ECG branch")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_title("ECG-only Calibration", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_attention_examples(attention_examples: list[dict], output_dir: Path):
    if not attention_examples:
        return
    for idx, item in enumerate(attention_examples, 1):
        signal = item["signal"]
        attention = item["attention"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        axes[0].plot(attention, color="#7C3AED")
        axes[0].set_title(f"Attention weights | sample {idx} | label={item['label']}")
        axes[0].grid(True, alpha=0.2)

        lead_importance = np.mean(np.abs(signal), axis=1)
        axes[1].bar(LEAD_NAMES, lead_importance, color="#0F766E")
        axes[1].set_title("Per-lead absolute activation proxy")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_dir / f"attention_example_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def _lead_occlusion_importance(model, loader, device, max_batches: int = 4):
    model.eval()
    lead_deltas = []
    batches_seen = 0
    for ecg, clinical, _labels in loader:
        if batches_seen >= max_batches:
            break
        ecg = ecg.to(device, non_blocking=True)
        base_probs = torch.sigmoid(model.ecg_branch(ecg)).cpu().numpy()
        batch_deltas = []
        for lead_idx in range(ecg.shape[1]):
            ablated = ecg.clone()
            ablated[:, lead_idx, :] = 0.0
            probs = torch.sigmoid(model.ecg_branch(ablated)).cpu().numpy()
            batch_deltas.append(np.mean(np.abs(base_probs - probs)))
        lead_deltas.append(batch_deltas)
        batches_seen += 1
    mean_deltas = np.mean(np.asarray(lead_deltas), axis=0)
    return pd.DataFrame({"lead": LEAD_NAMES, "mean_abs_prob_delta": mean_deltas})


def main():
    parser = argparse.ArgumentParser(description="Standalone ECG-only evaluation for late fusion")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attention-plots", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _pos_weight, _test_metadata = load_and_prepare_data(
        subset=args.subset,
        train_num_workers=args.num_workers,
        val_num_workers=args.val_workers,
        batch_size=args.batch_size,
    )
    del train_loader

    model = _load_model(device)
    labels, probs, attention_examples, activation_stats = _collect_outputs(
        model,
        val_loader,
        device,
        max_attention_plots=args.attention_plots,
    )

    threshold, best_f1 = find_best_threshold(labels, probs)
    metrics, preds = _compute_metrics(labels, probs, threshold)
    metrics["best_search_f1"] = best_f1

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    _plot_confusion_matrix(cm, args.output_dir / "confusion_matrix.png")
    _plot_probability_hist(labels, probs, args.output_dir / "probability_histogram.png")
    _plot_calibration(labels, probs, args.output_dir / "calibration_curve.png")
    _plot_attention_examples(attention_examples, args.output_dir)

    lead_importance_df = _lead_occlusion_importance(model, val_loader, device)
    lead_importance_df.to_csv(args.output_dir / "lead_importance.csv", index=False)

    activation_json = args.output_dir / "activation_stats.json"
    activation_json.write_text(json.dumps(activation_stats, indent=2))

    summary_csv = args.output_dir / "ecg_only_metrics.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "average_precision",
                "best_search_f1",
                "n_samples",
                "n_positive",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "validation",
                "threshold": round(float(metrics["threshold"]), 6),
                "accuracy": round(float(metrics["accuracy"]), 6),
                "precision": round(float(metrics["precision"]), 6),
                "recall": round(float(metrics["recall"]), 6),
                "f1": round(float(metrics["f1"]), 6),
                "auc": round(float(metrics["auc"]), 6),
                "average_precision": round(float(metrics["average_precision"]), 6),
                "best_search_f1": round(float(metrics["best_search_f1"]), 6),
                "n_samples": int(labels.shape[0]),
                "n_positive": int(labels.sum()),
            }
        )

    payload = {
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
        "best_search_f1": float(best_f1),
        "activation_stats": activation_stats,
    }
    (args.output_dir / "ecg_only_metrics.json").write_text(json.dumps(payload, indent=2))

    print(f"[SAVE] ECG-only evaluation -> {args.output_dir}")


if __name__ == "__main__":
    main()
