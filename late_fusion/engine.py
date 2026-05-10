from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    DEFAULT_THRESHOLD,
    HARD_NEGATIVE_POWER,
    HARD_NEGATIVE_WEIGHT,
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_MIN,
    THRESHOLD_SEARCH_STEPS,
)


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()

    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        return amp_module.autocast(device_type=device.type, enabled=enabled)

    if device.type == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=enabled)

    return nullcontext()


def find_best_threshold(labels, probs, threshold_min=THRESHOLD_SEARCH_MIN,
                        threshold_max=THRESHOLD_SEARCH_MAX, steps=THRESHOLD_SEARCH_STEPS):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    candidate_thresholds = np.unique(probs)
    candidate_thresholds = candidate_thresholds[
        (candidate_thresholds >= threshold_min) & (candidate_thresholds <= threshold_max)
    ]

    if candidate_thresholds.size == 0:
        candidate_thresholds = np.linspace(threshold_min, threshold_max, steps)

    for threshold in candidate_thresholds:
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1 or (score == best_f1 and abs(float(threshold) - DEFAULT_THRESHOLD) < abs(best_threshold - DEFAULT_THRESHOLD)):
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, max(best_f1, 0.0)


def _compute_metrics(labels, probs, threshold):
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

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
        "average_precision": average_precision,
        "threshold": float(threshold),
    }
    return metrics, preds


def _forward_branch_outputs(model, ecg, clinical):
    if hasattr(model, "forward_with_branches"):
        return model.forward_with_branches(ecg, clinical)

    fusion_logits = model(ecg, clinical)
    return {
        "fusion_logits": fusion_logits,
        "ecg_logits": fusion_logits,
        "clinical_logits": fusion_logits,
    }


def _compute_hard_negative_weights(labels, fusion_logits):
    labels = labels.float()
    with torch.no_grad():
        probs = torch.sigmoid(fusion_logits.detach())
        negative_mask = (1.0 - labels)
        hard_negative_boost = 1.0 + (HARD_NEGATIVE_WEIGHT * probs.pow(HARD_NEGATIVE_POWER) * negative_mask)
    return hard_negative_boost


def _weighted_mean(loss_values, sample_weights):
    weighted_sum = torch.sum(loss_values * sample_weights)
    normalizer = torch.sum(sample_weights).clamp_min(1e-8)
    return weighted_sum / normalizer


def _combined_loss(
    criterion,
    outputs,
    labels,
    ecg_aux_weight: float,
    clinical_aux_weight: float,
):
    sample_weights = _compute_hard_negative_weights(labels, outputs["fusion_logits"])
    fusion_loss = _weighted_mean(
        criterion(outputs["fusion_logits"], labels, reduction="none"),
        sample_weights,
    )
    ecg_loss = _weighted_mean(
        criterion(outputs["ecg_logits"], labels, reduction="none"),
        sample_weights,
    )
    clinical_loss = _weighted_mean(
        criterion(outputs["clinical_logits"], labels, reduction="none"),
        sample_weights,
    )
    total_loss = fusion_loss + (ecg_aux_weight * ecg_loss) + (clinical_aux_weight * clinical_loss)
    loss_components = {
        "fusion_loss": float(fusion_loss.detach().item()),
        "ecg_loss": float(ecg_loss.detach().item()),
        "clinical_loss": float(clinical_loss.detach().item()),
        "hard_negative_weight_mean": float(sample_weights.detach().mean().item()),
        "hard_negative_weight_max": float(sample_weights.detach().max().item()),
    }
    return total_loss, loss_components


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_grad_norm=1.0,
    threshold=DEFAULT_THRESHOLD,
    amp_enabled=False,
    scaler=None,
    ecg_aux_weight=0.0,
    clinical_aux_weight=0.0,
):
    model.train()
    running_loss = 0.0
    running_fusion_loss = 0.0
    running_ecg_loss = 0.0
    running_clinical_loss = 0.0
    running_hard_negative_weight_mean = 0.0
    running_hard_negative_weight_max = 0.0
    n_samples = 0
    all_probs, all_labels = [], []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            outputs = _forward_branch_outputs(model, ecg, clinical)
            outputs = {key: value.float() for key, value in outputs.items()}
            loss, loss_components = _combined_loss(
                criterion,
                outputs,
                labels,
                ecg_aux_weight=ecg_aux_weight,
                clinical_aux_weight=clinical_aux_weight,
            )

        if torch.isnan(loss):
            continue

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_fusion_loss += loss_components["fusion_loss"] * labels.size(0)
        running_ecg_loss += loss_components["ecg_loss"] * labels.size(0)
        running_clinical_loss += loss_components["clinical_loss"] * labels.size(0)
        running_hard_negative_weight_mean += loss_components["hard_negative_weight_mean"] * labels.size(0)
        running_hard_negative_weight_max += loss_components["hard_negative_weight_max"] * labels.size(0)
        n_samples += labels.size(0)

        probs = torch.sigmoid(outputs["fusion_logits"].detach()).float().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.detach().float().cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    metrics, _ = _compute_metrics(all_labels, all_probs, threshold)
    metrics["fusion_loss"] = running_fusion_loss / max(n_samples, 1)
    metrics["ecg_loss"] = running_ecg_loss / max(n_samples, 1)
    metrics["clinical_loss"] = running_clinical_loss / max(n_samples, 1)
    metrics["hard_negative_weight_mean"] = running_hard_negative_weight_mean / max(n_samples, 1)
    metrics["hard_negative_weight_max"] = running_hard_negative_weight_max / max(n_samples, 1)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold=DEFAULT_THRESHOLD,
    auto_threshold=False,
    return_outputs=False,
    amp_enabled=False,
    ecg_aux_weight=0.0,
    clinical_aux_weight=0.0,
):
    model.eval()
    running_loss = 0.0
    running_fusion_loss = 0.0
    running_ecg_loss = 0.0
    running_clinical_loss = 0.0
    running_hard_negative_weight_mean = 0.0
    running_hard_negative_weight_max = 0.0
    all_probs, all_labels = [], []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with _autocast_context(device, amp_enabled):
            outputs = _forward_branch_outputs(model, ecg, clinical)
            outputs = {key: value.float() for key, value in outputs.items()}
            loss, loss_components = _combined_loss(
                criterion,
                outputs,
                labels,
                ecg_aux_weight=ecg_aux_weight,
                clinical_aux_weight=clinical_aux_weight,
            )
        running_loss += loss.item() * labels.size(0)
        running_fusion_loss += loss_components["fusion_loss"] * labels.size(0)
        running_ecg_loss += loss_components["ecg_loss"] * labels.size(0)
        running_clinical_loss += loss_components["clinical_loss"] * labels.size(0)
        running_hard_negative_weight_mean += loss_components["hard_negative_weight_mean"] * labels.size(0)
        running_hard_negative_weight_max += loss_components["hard_negative_weight_max"] * labels.size(0)

        probs = torch.sigmoid(outputs["fusion_logits"]).float().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.float().cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    if auto_threshold:
        threshold, best_f1 = find_best_threshold(all_labels, all_probs)
        metrics, preds = _compute_metrics(all_labels, all_probs, threshold)
        metrics["best_search_f1"] = best_f1
    else:
        metrics, preds = _compute_metrics(all_labels, all_probs, threshold)

    metrics["fusion_loss"] = running_fusion_loss / max(len(loader.dataset), 1)
    metrics["ecg_loss"] = running_ecg_loss / max(len(loader.dataset), 1)
    metrics["clinical_loss"] = running_clinical_loss / max(len(loader.dataset), 1)
    metrics["hard_negative_weight_mean"] = running_hard_negative_weight_mean / max(len(loader.dataset), 1)
    metrics["hard_negative_weight_max"] = running_hard_negative_weight_max / max(len(loader.dataset), 1)

    if return_outputs:
        outputs = {
            "labels": [int(v) for v in np.asarray(all_labels, dtype=int)],
            "preds": [int(v) for v in preds],
            "probs": [float(v) for v in np.asarray(all_probs, dtype=float)],
        }
        return avg_loss, metrics, outputs

    return avg_loss, metrics
