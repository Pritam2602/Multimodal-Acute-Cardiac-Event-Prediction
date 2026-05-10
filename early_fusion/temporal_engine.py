import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, average_precision_score

from .config import DEFAULT_THRESHOLD

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


def temporal_contrastive_loss(embeddings, labels, margin=0.5):
    """
    Push apart Chronic (0) from Acute (1) trajectories.
    embeddings: (B, 256) context vector from TemporalAttention
    """
    with torch.autocast(device_type=embeddings.device.type if embeddings.device.type != 'cpu' else 'cuda', enabled=False):
        embeddings = embeddings.float()
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_emb = embeddings[pos_mask]
        neg_emb = embeddings[neg_mask]
    
    if len(pos_emb) == 0 or len(neg_emb) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    pos_emb = F.normalize(pos_emb.float(), p=2, dim=1, eps=1e-6)
    neg_emb = F.normalize(neg_emb.float(), p=2, dim=1, eps=1e-6)
    
    sim = torch.mm(pos_emb, neg_emb.t())
    threshold = 1.0 - margin
    loss = F.relu(sim - threshold).mean()
    return loss


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm=1.0, threshold=DEFAULT_THRESHOLD,
                    scaler=None, scheduler=None, contrastive_lambda=0.0, entropy_lambda=0.0):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []
    use_amp = (scaler is not None and device.type == "cuda")

    for batch in loader:
        ecg_seq = batch["ecg_seq"].to(device, non_blocking=True)
        trop_seq = batch["trop_seq"].to(device, non_blocking=True)
        trop_times = batch["trop_time_deltas"].to(device, non_blocking=True)
        ecg_times = batch["ecg_time_deltas"].to(device, non_blocking=True)
        seq_mask = batch["seq_mask"].to(device, non_blocking=True)
        clinical = batch["clinical_static"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = labels.size(0)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(ecg_seq, trop_seq, trop_times, ecg_times, seq_mask, clinical)
            logits = logits.squeeze(1)
            
            # Primary Loss
            loss = criterion(logits, labels)
            
            # Temporal Contrastive Loss
            if contrastive_lambda > 0:
                c_loss = temporal_contrastive_loss(model.last_trajectory_embedding, labels)
                loss = loss + contrastive_lambda * c_loss
                
            # Temporal Confidence Supervision (Entropy Band Regularization)
            if entropy_lambda > 0 and model.last_attn_weights is not None:
                attn = torch.clamp(model.last_attn_weights, min=1e-8)
                temporal_entropy = -torch.sum(attn * torch.log(attn), dim=1).mean()
                
                # Target healthy temporal band: 0.45 - 0.65
                temp_penalty = F.relu(0.45 - temporal_entropy) + F.relu(temporal_entropy - 0.65)
                loss = loss + entropy_lambda * temp_penalty
                
            # Spatial Confidence Supervision (Lead Entropy Band Regularization)
            if entropy_lambda > 0 and hasattr(model, 'ecg_encoder') and hasattr(model.ecg_encoder, 'last_spatial_weights') and model.ecg_encoder.last_spatial_weights is not None:
                spatial_w = model.ecg_encoder.last_spatial_weights # (B*T, 12)
                
                # We should only penalize entropy on valid timesteps
                mask_flat = seq_mask.view(-1)
                valid_spatial_w = spatial_w[mask_flat == 1.0]
                
                if len(valid_spatial_w) > 0:
                    spatial_w_safe = torch.clamp(valid_spatial_w, min=1e-8)
                    spatial_entropy = -torch.sum(spatial_w_safe * torch.log(spatial_w_safe), dim=1).mean()
                    
                    # Target healthy spatial band: 1.8 - 2.3 (clinically coherent multi-lead)
                    spatial_penalty = F.relu(1.8 - spatial_entropy) + F.relu(spatial_entropy - 2.3)
                    loss = loss + (entropy_lambda * 0.5) * spatial_penalty

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            if scheduler is not None and scale_before <= scale_after:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / n_samples
    metrics, _ = _compute_metrics(all_labels, all_probs, threshold)
    metrics["loss"] = epoch_loss
    return metrics


def find_optimal_threshold(labels, probs):
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=None):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []
    
    # Optional: collect attention weights
    all_attn_weights = []
    all_spatial_weights = []

    for batch in loader:
        ecg_seq = batch["ecg_seq"].to(device, non_blocking=True)
        trop_seq = batch["trop_seq"].to(device, non_blocking=True)
        trop_times = batch["trop_time_deltas"].to(device, non_blocking=True)
        ecg_times = batch["ecg_time_deltas"].to(device, non_blocking=True)
        seq_mask = batch["seq_mask"].to(device, non_blocking=True)
        clinical = batch["clinical_static"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = labels.size(0)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(ecg_seq, trop_seq, trop_times, ecg_times, seq_mask, clinical)
            logits = logits.squeeze(1)
            loss = criterion(logits, labels)

        running_loss += loss.item() * batch_size
        n_samples += batch_size

        # Apply probability temperature scaling for better calibration
        temperature = 1.5
        probs = torch.sigmoid(logits / temperature).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
        # Save temporal attention
        if model.last_attn_weights is not None:
            all_attn_weights.extend(model.last_attn_weights.cpu().numpy())
            
        # Save spatial attention
        if hasattr(model, 'ecg_encoder') and hasattr(model.ecg_encoder, 'last_spatial_weights') and model.ecg_encoder.last_spatial_weights is not None:
            spatial_w = model.ecg_encoder.last_spatial_weights.detach().cpu().numpy() # (B*T, 12)
            mask_flat = seq_mask.view(-1).cpu().numpy()
            valid_spatial_w = spatial_w[mask_flat == 1.0]
            all_spatial_weights.extend(valid_spatial_w)

    epoch_loss = running_loss / n_samples
    
    if threshold is None:
        threshold = find_optimal_threshold(all_labels, all_probs)
        
    metrics, preds = _compute_metrics(all_labels, all_probs, threshold)
    metrics["loss"] = epoch_loss

    # Compute Attention Biomarkers
    if len(all_attn_weights) > 0:
        attn_np = np.array(all_attn_weights) # (N, T)
        # Avoid log(0)
        attn_safe = np.clip(attn_np, 1e-8, 1.0)
        
        entropy = -np.sum(attn_safe * np.log(attn_safe), axis=1).mean()
        dominance = np.max(attn_np, axis=1).mean()
        # Sparsity: L2 norm of attention weights. Higher = more sparse/peaky.
        sparsity = np.sum(attn_np ** 2, axis=1).mean()
        
        metrics["attn_entropy"] = float(entropy)
        metrics["attn_dominance"] = float(dominance)
        metrics["attn_sparsity"] = float(sparsity)

    if len(all_spatial_weights) > 0:
        spatial_np = np.array(all_spatial_weights) # (Valid_T, 12)
        spatial_safe = np.clip(spatial_np, 1e-8, 1.0)
        lead_entropy = -np.sum(spatial_safe * np.log(spatial_safe), axis=1).mean()
        metrics["lead_entropy"] = float(lead_entropy)

    return metrics, all_probs, all_labels, preds, all_attn_weights
