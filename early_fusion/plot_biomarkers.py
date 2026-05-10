import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

from early_fusion.config import ARTIFACT_ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    
    history_file = ARTIFACT_ROOT / "runs" / args.run_name / "metrics" / "history.json"
    output_dir = ARTIFACT_ROOT / "runs" / args.run_name / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not history_file.exists():
        print(f"Error: Could not find history.json at {history_file}")
        return
        
    with open(history_file, "r") as f:
        history = json.load(f)
        
    train_f1 = [epoch_metrics["f1"] for epoch_metrics in history["train"]]
    val_f1 = [epoch_metrics["f1"] for epoch_metrics in history["val"]]
    val_auc = [epoch_metrics["auc"] for epoch_metrics in history["val"]]
    
    val_temp_entropy = [epoch_metrics.get("attn_entropy", 0) for epoch_metrics in history["val"]]
    val_lead_entropy = [epoch_metrics.get("lead_entropy", 0) for epoch_metrics in history["val"]]
    val_dominance = [epoch_metrics.get("attn_dominance", 0) for epoch_metrics in history["val"]]
    
    epochs = range(1, len(val_f1) + 1)
    
    # ── Plot 1: Performance (F1 and AUC) ──
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1, label="Train F1", linestyle="--", marker="o", color="lightblue")
    plt.plot(epochs, val_f1, label="Val F1", marker="o", color="tab:blue")
    plt.plot(epochs, val_auc, label="Val AUC", marker="o", color="tab:green")
    
    plt.title("Model Performance Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.savefig(output_dir / "performance_curve.png", dpi=150)
    plt.close()
    
    # ── Plot 2: Attention Biomarkers ──
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_temp_entropy, label="Temporal Entropy", marker="s", color="tab:purple")
    
    if any(e > 0 for e in val_lead_entropy):
        plt.plot(epochs, val_lead_entropy, label="Spatial Lead Entropy", marker="^", color="tab:orange")
        
    plt.plot(epochs, val_dominance, label="Temporal Dominance", marker="d", color="tab:red")
    
    # Add healthy physiological zones
    plt.axhspan(0.45, 0.65, color='tab:purple', alpha=0.1, label="Healthy Temporal Entropy Zone")
    plt.axhspan(0.60, 0.75, color='tab:red', alpha=0.1, label="Healthy Dominance Zone")
    
    plt.title("Attention Biomarker Trajectories")
    plt.xlabel("Epoch")
    plt.ylabel("Biomarker Value")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.savefig(output_dir / "biomarker_curve.png", dpi=150)
    plt.close()
    
    print(f"Saved plotting results to {output_dir}")

if __name__ == "__main__":
    main()
