import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_root))

import numpy as np
import torch
import torch.nn as nn

from late_fusion.config import (
    ECG_LENGTH,
    ECG_LEADS,
    METRICS_DIR,
    MODELS_DIR,
    NUM_CLINICAL_FEATURES,
)
from late_fusion.dataset import load_and_prepare_data
from late_fusion.engine import _compute_metrics, find_best_threshold
from late_fusion.model import LateFusionModel


DEFAULT_SUMMARY_CSV = METRICS_DIR / "branch_f1_scores.csv"
DEFAULT_PREDICTIONS_CSV = METRICS_DIR / "branch_predictions.csv"
DEFAULT_CHECKPOINT = MODELS_DIR / "latest_checkpoint.pth"
DEFAULT_WEIGHTS = MODELS_DIR / "late_fusion_model.pth"


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


@torch.no_grad()
def collect_branch_outputs(model, loader, device):
    model.eval()
    all_labels = []
    all_ecg_probs = []
    all_clinical_probs = []
    all_fusion_probs = []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        if hasattr(model, "forward_with_branches"):
            outputs = model.forward_with_branches(ecg, clinical)
            ecg_logits = outputs["ecg_logits"]
            clinical_logits = outputs["clinical_logits"]
            fusion_logits = outputs["fusion_logits"]
        elif hasattr(model.ecg_branch, "forward_features") and hasattr(model.clinical_branch, "forward_features"):
            ecg_embedding = model.ecg_branch.forward_features(ecg)
            clinical_embedding = model.clinical_branch.forward_features(clinical)
            ecg_logits = model.ecg_branch.logit_head(ecg_embedding).squeeze(-1)
            clinical_logits = model.clinical_branch.logit_head(clinical_embedding).squeeze(-1)
            fusion_input = torch.cat([ecg_embedding, clinical_embedding], dim=1)
            fusion_logits = model.fusion_head(fusion_input).squeeze(-1)
        else:
            ecg_logits = model.ecg_branch(ecg)
            clinical_logits = model.clinical_branch(clinical)
            fusion_logits = model.fusion_head(torch.stack([ecg_logits, clinical_logits], dim=1)).squeeze(-1)

        all_ecg_probs.extend(torch.sigmoid(ecg_logits).cpu().numpy().astype(float))
        all_clinical_probs.extend(torch.sigmoid(clinical_logits).cpu().numpy().astype(float))
        all_fusion_probs.extend(torch.sigmoid(fusion_logits).cpu().numpy().astype(float))
        all_labels.extend(labels.cpu().numpy().astype(int))

    return {
        "labels": np.asarray(all_labels, dtype=int),
        "ecg_probs": np.asarray(all_ecg_probs, dtype=float),
        "clinical_probs": np.asarray(all_clinical_probs, dtype=float),
        "fusion_probs": np.asarray(all_fusion_probs, dtype=float),
    }


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

    raise FileNotFoundError(
        "No late-fusion checkpoint or model weights found. "
        f"Checked: {DEFAULT_CHECKPOINT} and {DEFAULT_WEIGHTS}"
    )


def _branch_metrics(labels: np.ndarray, probs: np.ndarray):
    threshold, best_f1 = find_best_threshold(labels, probs)
    metrics, preds = _compute_metrics(labels, probs, threshold)
    metrics["best_search_f1"] = best_f1
    return metrics, preds


def main():
    parser = argparse.ArgumentParser(description="Write separate clinical/ECG/fusion F1 metrics to CSV")
    parser.add_argument("--subset", type=int, default=None, help="Use only first N samples")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training DataLoader worker count")
    parser.add_argument("--val-workers", type=int, default=None, help="Override validation DataLoader worker count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _pos_weight, _test_metadata = load_and_prepare_data(
        subset=args.subset,
        train_num_workers=args.num_workers,
        val_num_workers=args.val_workers,
        batch_size=args.batch_size or 64,
    )
    del train_loader

    model = _load_model(device)
    outputs = collect_branch_outputs(model, val_loader, device)
    labels = outputs["labels"]

    rows = []
    prediction_columns = [
        "row_index",
        "label",
        "ecg_prob",
        "ecg_pred",
        "clinical_prob",
        "clinical_pred",
        "fusion_prob",
        "fusion_pred",
    ]

    branch_results = {}
    for branch_name, probs_key in [
        ("ecg", "ecg_probs"),
        ("clinical", "clinical_probs"),
        ("fusion", "fusion_probs"),
    ]:
        metrics, preds = _branch_metrics(labels, outputs[probs_key])
        branch_results[branch_name] = {"metrics": metrics, "preds": preds, "probs": outputs[probs_key]}
        rows.append(
            {
                "branch": branch_name,
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

    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "branch",
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
        writer.writerows(rows)

    with open(args.predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_columns)
        writer.writeheader()
        for idx in range(labels.shape[0]):
            writer.writerow(
                {
                    "row_index": idx,
                    "label": int(labels[idx]),
                    "ecg_prob": round(float(branch_results["ecg"]["probs"][idx]), 6),
                    "ecg_pred": int(branch_results["ecg"]["preds"][idx]),
                    "clinical_prob": round(float(branch_results["clinical"]["probs"][idx]), 6),
                    "clinical_pred": int(branch_results["clinical"]["preds"][idx]),
                    "fusion_prob": round(float(branch_results["fusion"]["probs"][idx]), 6),
                    "fusion_pred": int(branch_results["fusion"]["preds"][idx]),
                }
            )

    payload = {
        "device": str(device),
        "summary_csv": str(args.summary_csv),
        "predictions_csv": str(args.predictions_csv),
        "rows": rows,
    }
    json_path = args.summary_csv.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"[SAVE] Branch metrics -> {args.summary_csv}")
    print(f"[SAVE] Branch predictions -> {args.predictions_csv}")
    print(f"[SAVE] Branch metrics JSON -> {json_path}")


if __name__ == "__main__":
    main()
