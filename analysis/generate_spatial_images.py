import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from early_fusion.temporal_dataset import load_and_prepare_temporal_data
from early_fusion.temporal_model import TemporalMultimodalGRU

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')
OUT_DIR = r"D:\MINI_PROJECT\research_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Standard 12-lead layout
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def plot_spatial_attention_12lead(ecg_waveform, attention_weights, filename):
    """
    ecg_waveform: (12, 5000)
    attention_weights: (12,)
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    fig.suptitle('Lead-Aware Anatomical Attention (Phase 10 Early Fusion)', fontsize=16, fontweight='bold')
    
    # Normalize attention for coloring (red intensity)
    attn_min, attn_max = np.min(attention_weights), np.max(attention_weights)
    norm_attn = (attention_weights - attn_min) / (attn_max - attn_min + 1e-8)
    
    for i in range(12):
        row, col = i % 4, i // 4
        ax = axes[row, col]
        
        # Plot ECG
        ax.plot(ecg_waveform[i], color='black', linewidth=0.8)
        
        # Color background based on attention
        alpha = float(norm_attn[i]) * 0.65  # Max 65% opacity
        ax.set_facecolor((1.0, 0.85 * (1 - alpha), 0.85 * (1 - alpha), alpha + 0.1))
        
        ax.set_title(f"Lead {LEAD_NAMES[i]} (Attn: {attention_weights[i]:.3f})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
        # Draw a subtle ECG grid
        ax.set_xticks(np.arange(0, 5000, 100), minor=True)
        ax.set_yticks(np.arange(np.min(ecg_waveform[i]), np.max(ecg_waveform[i]), 0.5), minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=0.1, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, val_loader, _, _, n_clinical = load_and_prepare_temporal_data(batch_size=16, train_workers=0, val_workers=0)
    
    print("Loading Phase 10 Model...")
    model = TemporalMultimodalGRU(n_static_clinical=n_clinical)
    model.load_state_dict(torch.load(r"D:\MINI_PROJECT\early_fusion\artifacts\runs\phase10_early_fusion_curated\models\best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    saved_count = 0
    with torch.no_grad():
        for batch in val_loader:
            ecg_seq = batch["ecg_seq"].to(device)
            trop_seq = batch["trop_seq"].to(device)
            t_deltas = batch["trop_time_deltas"].to(device)
            e_deltas = batch["ecg_time_deltas"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            clinical = batch["clinical_static"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(ecg_seq, trop_seq, t_deltas, e_deltas, seq_mask, clinical)
            if isinstance(logits, tuple): logits = logits[0]
            
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            labels = labels.cpu().numpy()
            spatial_attn = model.ecg_encoder.last_spatial_weights.cpu().numpy()
            
            for i in range(len(labels)):
                # Look for high-confidence true positives
                if labels[i] == 1 and probs[i] > 0.8:
                    mask_i = seq_mask[i].cpu().numpy()
                    last_t = np.where(mask_i == 1)[0][-1]
                    
                    ecg = ecg_seq[i, last_t].cpu().numpy()
                    attn = spatial_attn[i]
                    
                    fname = os.path.join(OUT_DIR, f"6_spatial_ami_phase10_{saved_count+1}.png")
                    plot_spatial_attention_12lead(ecg, attn, fname)
                    print(f"Saved: {fname}")
                    
                    saved_count += 1
                    if saved_count >= 3:
                        print("Generated 3 spatial attention plots successfully!")
                        return

if __name__ == "__main__":
    main()
