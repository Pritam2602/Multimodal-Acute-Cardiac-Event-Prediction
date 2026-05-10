import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, average_precision_score

from early_fusion.config import DEFAULT_THRESHOLD

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
            # Late fusion returns 3 logits!
            final_logit, ecg_logit, clin_logit = model(ecg_seq, trop_seq, trop_times, ecg_times, seq_mask, clinical)
            
            # Primary Fusion Loss
            loss = criterion(final_logit, labels)
            
            # Auxiliary Branch Losses (ensures each branch learns independent strong representations)
            loss += 0.3 * criterion(ecg_logit, labels)
            loss += 0.3 * criterion(clin_logit, labels)
            
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(final_logit).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / n_samples
    
    # We find an optimal threshold on the train set dynamically (could also use static)
    # But usually evaluate() does the dynamic threshold finding. 
    # For train, we just use DEFAULT_THRESHOLD
    metrics, _ = _compute_metrics(all_labels, all_probs, threshold)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_all=True):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []
    
    # Track interpretability biomarkers
    total_attn_entropy = 0.0
    total_attn_dominance = 0.0
    total_attn_sparsity = 0.0
    total_lead_entropy = 0.0
    num_interpret_batches = 0

    for batch in loader:
        ecg_seq = batch["ecg_seq"].to(device, non_blocking=True)
        trop_seq = batch["trop_seq"].to(device, non_blocking=True)
        trop_times = batch["trop_time_deltas"].to(device, non_blocking=True)
        ecg_times = batch["ecg_time_deltas"].to(device, non_blocking=True)
        seq_mask = batch["seq_mask"].to(device, non_blocking=True)
        clinical = batch["clinical_static"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = labels.size(0)

        # Handle Late Fusion logits
        final_logit, ecg_logit, clin_logit = model(ecg_seq, trop_seq, trop_times, ecg_times, seq_mask, clinical)
        
        loss = criterion(final_logit, labels)

        running_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(final_logit).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
        # Calculate evaluation biomarkers
        if model.last_attn_weights is not None:
            attn = torch.clamp(model.last_attn_weights, min=1e-8) # (B, T)
            
            # 1. Temporal Entropy
            entropy = -torch.sum(attn * torch.log(attn), dim=1).mean().item()
            total_attn_entropy += entropy
            
            # 2. Temporal Dominance (Max attention weight)
            dominance = torch.max(attn, dim=1)[0].mean().item()
            total_attn_dominance += dominance
            
            # 3. Temporal Sparsity (Fraction of timesteps with > 10% attention)
            sparsity = (attn > 0.1).float().mean(dim=1).mean().item()
            total_attn_sparsity += sparsity
            
            # 4. Lead Entropy
            if hasattr(model, 'ecg_encoder') and hasattr(model.ecg_encoder, 'last_spatial_weights'):
                spatial_w = model.ecg_encoder.last_spatial_weights
                if spatial_w is not None:
                    mask_flat = seq_mask.view(-1)
                    valid_spatial_w = spatial_w[mask_flat == 1.0]
                    if len(valid_spatial_w) > 0:
                        sw = torch.clamp(valid_spatial_w, min=1e-8)
                        l_entropy = -torch.sum(sw * torch.log(sw), dim=1).mean().item()
                        total_lead_entropy += l_entropy
                        
            num_interpret_batches += 1

    epoch_loss = running_loss / n_samples
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 1) Find the optimal threshold via F1 maximization
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # 2) Compute standard metrics with optimal threshold
    metrics, final_preds = _compute_metrics(all_labels, all_probs, best_thresh)
    metrics["loss"] = epoch_loss
    
    # Add biomarkers to metrics
    if num_interpret_batches > 0:
        metrics["attn_entropy"] = total_attn_entropy / num_interpret_batches
        metrics["attn_dominance"] = total_attn_dominance / num_interpret_batches
        metrics["attn_sparsity"] = total_attn_sparsity / num_interpret_batches
        metrics["lead_entropy"] = total_lead_entropy / num_interpret_batches

    if return_all:
        return metrics, all_probs, all_labels, final_preds, None
    return metrics
