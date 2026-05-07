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
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_MIN,
    THRESHOLD_SEARCH_STEPS,
)


def find_best_threshold(labels, probs, threshold_min=THRESHOLD_SEARCH_MIN,
                        threshold_max=THRESHOLD_SEARCH_MAX, steps=THRESHOLD_SEARCH_STEPS):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    for threshold in np.linspace(threshold_min, threshold_max, steps):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
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


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm=1.0, threshold=DEFAULT_THRESHOLD,
                    scaler=None, label_smoothing=0.0, scheduler=None):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []
    use_amp = (scaler is not None and device.type == "cuda")

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply label smoothing: 0→smooth, 1→(1-smooth)
        if label_smoothing > 0:
            smooth_labels = labels * (1 - label_smoothing) + (1 - labels) * label_smoothing
        else:
            smooth_labels = labels

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(ecg, clinical)
            loss = criterion(logits, smooth_labels)

        if torch.isnan(loss):
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Step per-batch schedulers (e.g. OneCycleLR) AFTER optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    metrics, _ = _compute_metrics(all_labels, all_probs, threshold)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=DEFAULT_THRESHOLD,
             auto_threshold=False, return_outputs=False):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []
    use_amp = (device.type == "cuda")

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(ecg, clinical)
            loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    if auto_threshold:
        threshold, best_f1 = find_best_threshold(all_labels, all_probs)
        metrics, preds = _compute_metrics(all_labels, all_probs, threshold)
        metrics["best_search_f1"] = best_f1
    else:
        metrics, preds = _compute_metrics(all_labels, all_probs, threshold)

    if return_outputs:
        outputs = {
            "labels": [int(v) for v in np.asarray(all_labels, dtype=int)],
            "preds": [int(v) for v in preds],
            "probs": [float(v) for v in np.asarray(all_probs, dtype=float)],
        }
        return avg_loss, metrics, outputs

    return avg_loss, metrics
