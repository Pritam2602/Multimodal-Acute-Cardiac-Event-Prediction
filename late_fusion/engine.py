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

def _autocast_context(enabled: bool):
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled)

from .config import (
    CLINICAL_AUX_LOSS_WEIGHT,
    DEFAULT_THRESHOLD,
    ECG_AUX_LOSS_WEIGHT,
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


def _compute_total_loss(criterion, outputs, labels):
    final_logits, ecg_logits, clinical_logits = outputs
    final_loss = criterion(final_logits, labels)
    ecg_loss = criterion(ecg_logits, labels)
    clinical_loss = criterion(clinical_logits, labels)
    total_loss = (
        final_loss
        + ECG_AUX_LOSS_WEIGHT * ecg_loss
        + CLINICAL_AUX_LOSS_WEIGHT * clinical_loss
    )
    return total_loss, final_logits


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm=1.0, threshold=DEFAULT_THRESHOLD,
                    use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with _autocast_context(use_amp):
            outputs = model(ecg, clinical)
            loss, final_logits = _compute_total_loss(criterion, outputs, labels)

        if torch.isnan(loss):
            continue

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        probs = torch.sigmoid(final_logits.detach()).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    metrics, _ = _compute_metrics(all_labels, all_probs, threshold)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=DEFAULT_THRESHOLD,
             auto_threshold=False, return_outputs=False, use_amp=False):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for ecg, clinical, labels in loader:
        ecg = ecg.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        with _autocast_context(use_amp):
            outputs = model(ecg, clinical)
            loss, final_logits = _compute_total_loss(criterion, outputs, labels)
        running_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(final_logits).cpu().numpy()
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
