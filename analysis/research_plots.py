import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader

from early_fusion.temporal_dataset import TemporalFusionDataset, load_and_prepare_temporal_data, MAX_SEQ_LEN
from early_fusion.temporal_model import TemporalMultimodalGRU
from late_fusion.temporal_model import TemporalHybridLateFusion
from early_fusion.config import NUM_CLINICAL_FEATURES, ECG_LEADS

# Set Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("bright")
PLOT_DIR = Path(r"D:\MINI_PROJECT\research_plots")
PLOT_DIR.mkdir(exist_ok=True)

def get_predictions(model, loader, device, is_late_fusion=False):
    model.eval()
    all_probs = []
    all_labels = []
    all_temporal_attn = []
    all_spatial_attn = []
    
    with torch.no_grad():
        for batch in loader:
            ecg_seq = batch["ecg_seq"].to(device)
            trop_seq = batch["trop_seq"].to(device)
            t_deltas = batch["trop_time_deltas"].to(device)
            e_deltas = batch["ecg_time_deltas"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            clinical = batch["clinical_static"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(ecg_seq, trop_seq, t_deltas, e_deltas, seq_mask, clinical)
            if isinstance(logits, tuple):
                logits = logits[0]
                
            temporal_attn = model.last_attn_weights
            spatial_attn = model.ecg_encoder.last_spatial_weights
                
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_probs.extend(probs if probs.ndim > 0 else [probs])
            all_labels.extend(labels.cpu().numpy())
            all_temporal_attn.extend(temporal_attn.cpu().numpy())
            all_spatial_attn.extend(spatial_attn.cpu().numpy())
            
    return np.array(all_probs), np.array(all_labels), np.array(all_temporal_attn), np.array(all_spatial_attn)

def plot_performance_curves(early_results, late_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    for name, (probs, labels) in [("Early Fusion", early_results), ("Late Fusion", late_results)]:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    
    # PR Curve
    for name, (probs, labels) in [("Early Fusion", early_results), ("Late Fusion", late_results)]:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax2.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})')
        
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall (PR) Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "1_performance_curves.png", dpi=300)
    plt.close()

def plot_calibration(early_results, late_results):
    plt.figure(figsize=(8, 8))
    for name, (probs, labels) in [("Early Fusion", early_results), ("Late Fusion", late_results)]:
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves (Reliability Diagram)')
    plt.legend()
    plt.savefig(PLOT_DIR / "2_calibration_curves.png", dpi=300)
    plt.close()

def plot_anatomical_attention(early_spatial, late_spatial):
    # Regional Groups: Inferior(II, III, aVF: indices 1, 2, 8), Anterior(V1-V4: 6, 7, 8, 9?), 
    # Actually MIMIC indices: I:0, II:1, III:2, aVR:3, aVL:4, aVF:5, V1:6, V2:7, V3:8, V4:9, V5:10, V6:11
    regions = {
        'Inferior': [1, 2, 5],
        'Lateral': [0, 4, 10, 11],
        'Anterior/Septal': [6, 7, 8, 9]
    }
    
    def get_regional_importance(spatial_attn):
        avg_attn = np.mean(spatial_attn, axis=0) # (12,)
        regional_values = {}
        for reg, idxs in regions.items():
            regional_values[reg] = np.mean(avg_attn[idxs])
        return regional_values

    early_reg = get_regional_importance(early_spatial)
    late_reg = get_regional_importance(late_spatial)
    
    categories = list(regions.keys())
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, [early_reg[c] for c in categories], width, label='Early Fusion')
    ax.bar(x + width/2, [late_reg[c] for c in categories], width, label='Late Fusion')
    
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Anatomical Regional Importance')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    plt.savefig(PLOT_DIR / "3_anatomical_attention.png", dpi=300)
    plt.close()

def plot_temporal_attention(early_temp, late_temp):
    # Show mean attention across timesteps T0, T1, T2
    early_mean = np.mean(early_temp, axis=0)
    late_mean = np.mean(late_temp, axis=0)
    
    timesteps = ['T0 (Admission)', 'T1 (Middle)', 'T2 (Latest)']
    x = np.arange(len(timesteps))
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, early_mean, marker='s', lw=3, label='Early Fusion')
    plt.plot(timesteps, late_mean, marker='o', lw=3, label='Late Fusion')
    
    plt.fill_between(timesteps, early_mean, alpha=0.2)
    plt.fill_between(timesteps, late_mean, alpha=0.2)
    
    plt.ylabel('Mean Temporal Attention Score')
    plt.title('Temporal Importance Profile (GRU Attention)')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(PLOT_DIR / "4_temporal_attention.png", dpi=300)
    plt.close()

def plot_f1_robustness(early_results, late_results):
    thresholds = np.linspace(0.1, 0.9, 81)
    
    plt.figure(figsize=(10, 6))
    for name, (probs, labels) in [("Early Fusion", early_results), ("Late Fusion", late_results)]:
        f1_scores = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in thresholds]
        plt.plot(thresholds, f1_scores, lw=2, label=name)
        
        best_idx = np.argmax(f1_scores)
        plt.scatter(thresholds[best_idx], f1_scores[best_idx], s=100)
        plt.annotate(f'{f1_scores[best_idx]:.3f}', (thresholds[best_idx], f1_scores[best_idx]+0.01))

    plt.xlabel('Probability Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Robustness across Thresholds')
    plt.legend()
    plt.savefig(PLOT_DIR / "5_f1_robustness.png", dpi=300)
    plt.close()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PLOT] Using device: {DEVICE}")
    
    # 1. Load Data
    _, val_loader, _, _, n_clinical = load_and_prepare_temporal_data(batch_size=32, train_workers=0, val_workers=0)
    
    # 2. Load Models
    print("[PLOT] Loading Phase 10 (Early Fusion)...")
    early_path = r"D:\MINI_PROJECT\early_fusion\artifacts\runs\phase10_early_fusion_curated\models\best_model.pth"
    early_model = TemporalMultimodalGRU(n_static_clinical=n_clinical)
    early_model.load_state_dict(torch.load(early_path, map_location=DEVICE, weights_only=True))
    early_model.to(DEVICE)
    
    print("[PLOT] Loading Phase 9 (Late Fusion)...")
    late_path = r"D:\MINI_PROJECT\early_fusion\artifacts\runs\phase9_refined_cohort\models\best_model.pth"
    late_model = TemporalHybridLateFusion(n_static_clinical=n_clinical)
    late_model.load_state_dict(torch.load(late_path, map_location=DEVICE, weights_only=True))
    late_model.to(DEVICE)
    
    # 3. Get Predictions and Attention
    print("[PLOT] Computing predictions for Early Fusion...")
    e_probs, e_labels, e_temp, e_spat = get_predictions(early_model, val_loader, DEVICE, is_late_fusion=False)
    
    print("[PLOT] Computing predictions for Late Fusion...")
    l_probs, l_labels, l_temp, l_spat = get_predictions(late_model, val_loader, DEVICE, is_late_fusion=True)
    
    # 4. Filter for True Positives for attention plots (to see what ischemic signals look like)
    tp_idx_e = np.where((e_labels == 1) & (e_probs > 0.5))[0]
    tp_idx_l = np.where((l_labels == 1) & (l_probs > 0.5))[0]
    
    # 5. Generate Plots
    print("[PLOT] Generating Plot 1: Performance Curves...")
    plot_performance_curves((e_probs, e_labels), (l_probs, l_labels))
    
    print("[PLOT] Generating Plot 2: Calibration...")
    plot_calibration((e_probs, e_labels), (l_probs, l_labels))
    
    print("[PLOT] Generating Plot 3: Anatomical Attention...")
    plot_anatomical_attention(e_spat[tp_idx_e], l_spat[tp_idx_l])
    
    print("[PLOT] Generating Plot 4: Temporal Attention...")
    plot_temporal_attention(e_temp[tp_idx_e], l_temp[tp_idx_l])
    
    print("[PLOT] Generating Plot 5: F1 Robustness...")
    plot_f1_robustness((e_probs, e_labels), (l_probs, l_labels))
    
    print(f"\n[SUCCESS] All 5 research plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
