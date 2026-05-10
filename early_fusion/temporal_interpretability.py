import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .config import ARTIFACT_ROOT, SEED
from .temporal_model import TemporalMultimodalGRU
from .temporal_dataset import load_and_prepare_temporal_data

def visualize_trajectory_attention(model, dataset_loader, device, output_dir, n_samples=5):
    """
    Runs the model on a few test samples and plots the temporal attention weights
    alongside the troponin trajectories.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_found = {"ami": 0, "control": 0}
    max_per_class = n_samples // 2
    
    with torch.no_grad():
        for batch in dataset_loader:
            ecg_seq = batch["ecg_seq"].to(device)
            trop_seq = batch["trop_seq"].to(device)
            trop_times = batch["trop_time_deltas"].to(device)
            ecg_times = batch["ecg_time_deltas"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            clinical = batch["clinical_static"].to(device)
            labels = batch["label"].cpu().numpy()
            
            # Forward pass to cache attention weights
            logits = model(ecg_seq, trop_seq, trop_times, ecg_times, seq_mask, clinical)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            attn_weights = model.last_attn_weights.cpu().numpy()
            
            for i in range(len(labels)):
                is_ami = labels[i] == 1
                cls_key = "ami" if is_ami else "control"
                
                if samples_found[cls_key] >= max_per_class:
                    continue
                    
                # We only want patients with at least 2 timesteps for interesting visualization
                mask_i = seq_mask[i].cpu().numpy()
                valid_len = int(mask_i.sum())
                if valid_len < 2:
                    continue
                
                # Extract sequences for plotting
                trop_vals = trop_seq[i].cpu().numpy()[:valid_len]
                trop_t = trop_times[i].cpu().numpy()[:valid_len]
                attn_vals = attn_weights[i][:valid_len]
                
                # Plot
                fig, ax1 = plt.subplots(figsize=(8, 5))
                
                # Bar chart for attention
                color = 'tab:blue'
                ax1.set_xlabel('Time from Admission (Hours)')
                ax1.set_ylabel('Attention Weight', color=color)
                # Ensure x-axis correctly spaces the bars by time
                bars = ax1.bar(trop_t, attn_vals, width=1.0, color=color, alpha=0.6, label='Diagnostic Focus')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim(0, 1.0)
                
                # Line chart for Troponin
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('Troponin T (ng/mL)', color=color)
                ax2.plot(trop_t, trop_vals, color=color, marker='o', linewidth=2, label='Troponin')
                ax2.tick_params(axis='y', labelcolor=color)
                
                title = f"{'Acute MI' if is_ami else 'Control'} Patient\nPred Prob: {probs[i]:.3f}"
                plt.title(title)
                fig.tight_layout()
                
                file_name = f"patient_{cls_key}_{samples_found[cls_key]}.png"
                plt.savefig(output_dir / file_name, dpi=150)
                plt.close()
                
                samples_found[cls_key] += 1
                
                if samples_found["ami"] >= max_per_class and samples_found["control"] >= max_per_class:
                    return

def visualize_spatial_attention(model, dataset_loader, device, output_dir, n_samples=5):
    """
    Runs the model on test samples and plots the spatial lead attention heatmaps
    over the trajectory timesteps.
    """
    if not hasattr(model.ecg_encoder, 'last_spatial_weights'):
        print("[INFO] Model does not support spatial attention visualization.")
        return
        
    model.eval()
    spatial_dir = output_dir / "spatial_attention"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    
    samples_found = {"ami": 0, "control": 0}
    max_per_class = n_samples // 2
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    with torch.no_grad():
        for batch in dataset_loader:
            ecg_seq = batch["ecg_seq"].to(device)
            trop_seq = batch["trop_seq"].to(device)
            trop_times = batch["trop_time_deltas"].to(device)
            ecg_times = batch["ecg_time_deltas"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            clinical = batch["clinical_static"].to(device)
            labels = batch["label"].cpu().numpy()
            
            # Forward pass
            logits = model(ecg_seq, trop_seq, trop_times, ecg_times, seq_mask, clinical)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            if model.ecg_encoder.last_spatial_weights is None:
                continue
                
            spatial_weights = model.ecg_encoder.last_spatial_weights.cpu().numpy() # (B*T, 12)
            B, T = seq_mask.shape
            spatial_weights = spatial_weights.reshape(B, T, 12)
            
            for i in range(len(labels)):
                is_ami = labels[i] == 1
                cls_key = "ami" if is_ami else "control"
                
                if samples_found[cls_key] >= max_per_class:
                    continue
                    
                mask_i = seq_mask[i].cpu().numpy()
                valid_len = int(mask_i.sum())
                if valid_len < 2:
                    continue
                
                lead_attn = spatial_weights[i, :valid_len, :]
                trop_t = trop_times[i].cpu().numpy()[:valid_len]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(lead_attn.T, cmap='viridis', aspect='auto', interpolation='nearest')
                
                ax.set_yticks(np.arange(12))
                ax.set_yticklabels(lead_names)
                ax.set_xticks(np.arange(valid_len))
                ax.set_xticklabels([f"{t:.1f}h" for t in trop_t])
                
                ax.set_title(f"{'Acute MI' if is_ami else 'Control'} Spatial Lead Attention (Prob: {probs[i]:.3f})")
                ax.set_xlabel("Time from Admission")
                ax.set_ylabel("ECG Lead")
                plt.colorbar(im, ax=ax, label="Attention Weight")
                
                file_name = f"spatial_{cls_key}_{samples_found[cls_key]}.png"
                plt.savefig(spatial_dir / file_name, dpi=150)
                plt.close()
                
                samples_found[cls_key] += 1
                if samples_found["ami"] >= max_per_class and samples_found["control"] >= max_per_class:
                    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the training run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ARTIFACT_ROOT / "runs" / args.run_name
    model_path = run_dir / "models" / "best_model.pth"
    output_dir = run_dir / "interpretability"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print("[INFO] Loading dataset...")
    # Setting workers to 0 for interpretability script to avoid multiprocessing issues
    _, val_loader, test_loader, pos_weight, n_clinical = load_and_prepare_temporal_data(
        batch_size=16, train_workers=0, val_workers=0
    )
    
    print("[INFO] Loading model...")
    model = TemporalMultimodalGRU(n_static_clinical=n_clinical)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print(f"[INFO] Generating Interpretability plots to {output_dir}...")
    visualize_trajectory_attention(model, val_loader, device, output_dir, n_samples=10)
    visualize_spatial_attention(model, val_loader, device, output_dir, n_samples=10)
    print("[INFO] Done!")

if __name__ == "__main__":
    main()
