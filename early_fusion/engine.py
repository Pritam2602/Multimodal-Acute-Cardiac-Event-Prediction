import numpy as np
import torch
import torch.nn.functional as F
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
    CLINICAL_FEATURES,
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


def contrastive_margin_loss(embeddings, labels, hard_neg_mask, margin=0.5):
    pos_mask = (labels == 1)
    
    pos_emb = embeddings[pos_mask]
    neg_emb = embeddings[hard_neg_mask]
    
    if len(pos_emb) == 0 or len(neg_emb) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    pos_emb = F.normalize(pos_emb, p=2, dim=1)
    neg_emb = F.normalize(neg_emb, p=2, dim=1)
    
    sim = torch.mm(pos_emb, neg_emb.t())
    threshold = 1.0 - margin
    loss = F.relu(sim - threshold).mean()
    return loss


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm=1.0, threshold=DEFAULT_THRESHOLD,
                    scaler=None, label_smoothing=0.0, scheduler=None,
                    mixup_alpha=0.0, entropy_lambda=0.0, contrastive_lambda=0.0):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []
    use_amp = (scaler is not None and device.type == "cuda")

    for batch in loader:
        if len(batch) == 5:
            ecg, clinical, labels, ecg_base, has_baseline = batch
            ecg_base = ecg_base.to(device, non_blocking=True)
            has_baseline = has_baseline.to(device, non_blocking=True)
        else:
            ecg, clinical, labels = batch
            ecg_base, has_baseline = None, None

        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply label smoothing: 0->smooth, 1->(1-smooth)
        if label_smoothing > 0:
            smooth_labels = labels * (1 - label_smoothing) + (1 - labels) * label_smoothing
        else:
            smooth_labels = labels

        # Mixup augmentation: interpolate pairs of samples
        use_mixup = mixup_alpha > 0.0 and ecg.size(0) > 1
        if use_mixup:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)  # ensure lam >= 0.5 so primary sample dominates
            idx = torch.randperm(ecg.size(0), device=device)
            ecg = lam * ecg + (1 - lam) * ecg[idx]
            clinical = lam * clinical + (1 - lam) * clinical[idx]
            labels_b = smooth_labels[idx]

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            if ecg_base is not None and has_baseline is not None:
                logits = model(ecg, clinical, ecg_base=ecg_base, has_baseline=has_baseline)
            else:
                logits = model(ecg, clinical)
            if use_mixup:
                loss = lam * criterion(logits, smooth_labels) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, smooth_labels)
            
            # Apply Attention Entropy Constraint if requested and supported
            if entropy_lambda > 0.0:
                attn_weights = None
                if hasattr(model, '_clinical_gated') and hasattr(model._clinical_gated, 'last_attn_weights'):
                    attn_weights = model._clinical_gated.last_attn_weights.squeeze(2) # (B, 4, T')
                elif hasattr(model, '_siamese_crossattn') and hasattr(model._siamese_crossattn, 'last_attn_weights'):
                    # siamese_crossattn last_attn_weights shape: (B, 1, 1, T)
                    attn_weights = model._siamese_crossattn.last_attn_weights.squeeze(2) # (B, 1, T)
                
                if attn_weights is not None:
                    # Shannon entropy: -sum(p * log(p + eps))
                    eps = 1e-8
                    # Calculate entropy per head per sample, then mean across batch and heads
                    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()
                    loss = loss + (entropy_lambda * entropy)
                
            # Apply False-Positive Adversarial Separation (Contrastive Margin Loss)
            if contrastive_lambda > 0.0:
                fused_emb = None
                if hasattr(model, '_clinical_gated') and hasattr(model._clinical_gated, 'last_fused_embedding'):
                    fused_emb = model._clinical_gated.last_fused_embedding
                elif hasattr(model, '_siamese_delta') and hasattr(model._siamese_delta, 'last_fused_embedding'):
                    fused_emb = model._siamese_delta.last_fused_embedding
                elif hasattr(model, '_siamese_crossattn') and hasattr(model._siamese_crossattn, 'last_fused_embedding'):
                    fused_emb = model._siamese_crossattn.last_fused_embedding
                
                if fused_emb is not None:
                    # Dynamically find indices for hard negative mining
                    try:
                        idx_trop = CLINICAL_FEATURES.index("Troponin_T_high")
                        idx_ckd = CLINICAL_FEATURES.index("has_ckd")
                        idx_hf = CLINICAL_FEATURES.index("has_hf")
                        idx_sepsis = CLINICAL_FEATURES.index("has_sepsis")
                        
                        comorbid_mask = (clinical[:, idx_trop] > 0.5) | (clinical[:, idx_ckd] > 0.5) | (clinical[:, idx_hf] > 0.5) | (clinical[:, idx_sepsis] > 0.5)
                        hard_neg_mask = (labels == 0) & comorbid_mask
                    except ValueError:
                        # Fallback if features are not exactly as expected
                        hard_neg_mask = (labels == 0)
                    
                    # We use discrete labels (0 vs 1) and the hard negative mask
                    c_loss = contrastive_margin_loss(fused_emb, labels, hard_neg_mask, margin=0.5)
                    loss = loss + (contrastive_lambda * c_loss)

        if torch.isnan(loss):
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer_stepped = (scale_before <= scale_after)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer_stepped = True

        # Step per-batch schedulers (e.g. OneCycleLR) AFTER optimizer.step()
        if scheduler is not None and optimizer_stepped:
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

    for batch in loader:
        if len(batch) == 5:
            ecg, clinical, labels, ecg_base, has_baseline = batch
            ecg_base = ecg_base.to(device, non_blocking=True)
            has_baseline = has_baseline.to(device, non_blocking=True)
        else:
            ecg, clinical, labels = batch
            ecg_base, has_baseline = None, None

        ecg = ecg.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if ecg_base is not None and has_baseline is not None:
                logits = model(ecg, clinical, ecg_base=ecg_base, has_baseline=has_baseline)
            else:
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
