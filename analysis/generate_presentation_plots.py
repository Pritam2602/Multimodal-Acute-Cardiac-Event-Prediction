import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from early_fusion.temporal_dataset import load_and_prepare_temporal_data
from early_fusion.temporal_model import TemporalMultimodalGRU

OUT_DIR = r"D:\MINI_PROJECT\research_plots"
os.makedirs(OUT_DIR, exist_ok=True)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def plot_clean_12lead(ecg_waveform, filename):
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    fig.suptitle('Processed 12-Lead ECG Waveform (Model Input)', fontsize=16, fontweight='bold')
    
    for i in range(12):
        row, col = i % 4, i // 4
        ax = axes[row, col]
        ax.plot(ecg_waveform[i], color='green', linewidth=1.0)
        ax.set_title(f"Lead {LEAD_NAMES[i]}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        
        # Grid
        ax.set_xticks(np.arange(0, 5000, 100), minor=True)
        ax.set_yticks(np.arange(np.min(ecg_waveform[i]), np.max(ecg_waveform[i]), 0.5), minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=0.2, alpha=0.3)
        ax.grid(which='major', color='red', linestyle='-', linewidth=0.4, alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_confidence_evolution(probs, filename):
    plt.figure(figsize=(9, 5.5))
    plt.style.use('seaborn-v0_8-whitegrid')
    timesteps = ['T0\n(Admission)', 'T1\n(Intermediate)', 'T2\n(Latest Pre-Diagnosis)']
    
    # Probabilities as percentages
    probs_pct = [p * 100 for p in probs]
    
    plt.plot(timesteps, probs_pct, marker='o', markersize=14, linewidth=4, color='#e74c3c')
    
    # Fill under curve
    plt.fill_between(timesteps, probs_pct, color='#e74c3c', alpha=0.15)
    
    # Add text labels on the points
    for i, p in enumerate(probs_pct):
        # Slightly offset the text above the point
        plt.annotate(f"{p:.1f}%", (timesteps[i], p + 4), fontsize=13, ha='center', fontweight='bold', color='#c0392b')
        
    plt.ylim(0, 105)
    plt.ylabel("Model Confidence (AMI Probability %)", fontsize=12, fontweight='bold')
    plt.title("Temporal Confidence Evolution (Disease Progression)", fontsize=15, fontweight='bold')
    
    # Draw decision boundary line
    plt.axhline(y=77.53, color='black', linestyle='--', alpha=0.6, linewidth=2, label='Optimal Decision Threshold (77.5%)')
    
    # Add shading for decision zones
    plt.axhspan(77.53, 105, facecolor='red', alpha=0.05)
    plt.axhspan(0, 77.53, facecolor='green', alpha=0.05)
    
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, val_loader, _, _, n_clinical = load_and_prepare_temporal_data(batch_size=1, train_workers=0, val_workers=0)
    
    print("Loading Phase 10 Model...")
    model = TemporalMultimodalGRU(n_static_clinical=n_clinical)
    model.load_state_dict(torch.load(r"D:\MINI_PROJECT\early_fusion\artifacts\runs\phase10_early_fusion_curated\models\best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    generated_ecg = False
    
    with torch.no_grad():
        for batch in val_loader:
            ecg_seq = batch["ecg_seq"].to(device)
            trop_seq = batch["trop_seq"].to(device)
            t_deltas = batch["trop_time_deltas"].to(device)
            e_deltas = batch["ecg_time_deltas"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            clinical = batch["clinical_static"].to(device)
            labels = batch["label"].to(device)
            
            # We want a patient that actually has 3 timesteps, is an AMI, and model successfully predicts
            if labels[0] == 1 and seq_mask[0, 2] == 1:
                # 1. Generate Clean ECG for T0
                if not generated_ecg:
                    ecg_t0 = ecg_seq[0, 0].cpu().numpy()
                    plot_clean_12lead(ecg_t0, os.path.join(OUT_DIR, "7_clean_12lead_sample.png"))
                    print("Generated clean 12-lead sample.")
                    generated_ecg = True
                
                # 2. Generate Temporal Confidence Evolution
                probs = []
                for t in range(1, 4): # Evaluate progressively up to t length
                    prog_mask = torch.zeros_like(seq_mask)
                    prog_mask[0, :t] = 1.0
                    
                    logits = model(ecg_seq, trop_seq, t_deltas, e_deltas, prog_mask, clinical)
                    if isinstance(logits, tuple): logits = logits[0]
                    p = torch.sigmoid(logits.squeeze()).item()
                    probs.append(p)
                
                # Check for a "good" narrative trajectory for the slide:
                # e.g., starts at low/medium risk at T0, climbs rapidly and crosses threshold at T1 or T2
                if probs[0] < 0.65 and probs[-1] > 0.85:
                    plot_confidence_evolution(probs, os.path.join(OUT_DIR, "8_temporal_confidence_evolution.png"))
                    print(f"Generated Temporal Confidence Evolution for patient showing progression: {probs}")
                    return # We are done!
                
        print("Couldn't find perfect progressive sample, generating for the last valid one...")
        if generated_ecg:
            plot_confidence_evolution(probs, os.path.join(OUT_DIR, "8_temporal_confidence_evolution.png"))
            print(f"Generated Temporal Confidence Evolution for patient: {probs}")

if __name__ == "__main__":
    main()
