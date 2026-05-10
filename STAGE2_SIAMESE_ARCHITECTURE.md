# Stage 2 Blueprint: Siamese Delta ECG Architecture
**Explicitly Modeling Physiological Deviation**

This document outlines the technical design for the Stage 2 Temporal Architecture. Our goal is to shift the network's objective from *"What does this waveform look like?"* to *"How has this waveform deviated from the patient's biological baseline?"*

---

## 1. Core Architectural Concept
Human cardiologists rarely diagnose an NSTEMI from a single ECG unless the ischemia is catastrophically obvious. They compare the current ECG against a prior discharge ECG to filter out chronic Left Bundle Branch Blocks (LBBB), Left Ventricular Hypertrophy (LVH), and chronic T-wave inversions. 

We will mathematically replicate this using a **Siamese 1D-ResNet Network**.

### The Forward Pass:
1. **Dual Inputs:** The network receives two tensors per patient:
   * `ecg_admit` (Shape: `12 x 5000`)
   * `ecg_base` (Shape: `12 x 5000`)
2. **Shared Weights (Siamese):** Both waveforms are passed through the *exact same* 1D-ResNet branch.
   * $E_{admit} = \text{ResNet}(ecg_{admit})$
   * $E_{base} = \text{ResNet}(ecg_{base})$
3. **Delta Extraction:** 
   * $E_{diff} = E_{admit} - E_{base}$  *(Signed difference: direction of change)*
   * $E_{abs} = |E_{admit} - E_{base}|$  *(Absolute difference: magnitude of deviation)*
4. **Clinical Embedding:**
   * $C_{emb} = \text{MLP}(Clinical\_Features)$ *(Now includes Troponin Velocity & `hours_since_baseline_ecg`)*
5. **Multimodal Fusion Engine:**
   * The final classifier receives the concatenated tensor: `[E_admit, E_diff, E_abs, C_emb]`
   * *Why keep $E_{admit}$?* If a patient is having a massive acute ST-elevation, we don't want the network to only look at the delta. We want it to see the absolute state *plus* the deviation.

---

## 2. Handling Missing Baseline ECGs
Not every patient in the ED has a prior ECG on file in the hospital system. The architecture must robustly handle zero-shot patients.
* **Learnable Missing Token:** Instead of padding with absolute zeros (which mathematically skews the delta), we will initialize a trainable `nn.Parameter` called `baseline_missing_embedding` (Shape: `1 x 256`).
* **Masking:** When a patient lacks a prior ECG, $E_{base}$ is replaced by `baseline_missing_embedding`. The network learns to naturally ignore the delta features for these specific cases.

---

## 3. The Cross-Attention Evolution (Future-Proofing)
Currently, our `clinical_gated` model uses the clinical tensor to "focus" the ECG tensor. 
In Stage 2, we will upgrade this to **Delta-Aware Attention**:
* **Query:** Clinical Context (Is Troponin rising?)
* **Key/Value:** The $E_{delta}$ tensor.
* **Mechanism:** "If Troponin velocity is high, strictly focus attention on the specific leads in $E_{delta}$ that have changed the most."

---

## 4. Implementation Steps in the Codebase
To execute this, we will need to modify three distinct areas:

1. **`Dataset_preproceesing/`**: 
   * Write a script that scans `record_list.csv` to find pairs of `(hadm_id, prior_ecg_time)`.
2. **`early_fusion/dataset.py`**:
   * Update `MultimodalCardiacDataset` `__getitem__` to return a tuple of `(ecg_admit, ecg_base, clinical, label)`.
3. **`early_fusion/model.py`**:
   * Build `class SiameseDeltaFusion(nn.Module)`.
   * Implement the shared `self.ecg_encoder` and the delta subtraction logic.

---
*Blueprint formulated to break the 0.80 F1 barrier by explicitly teaching the model temporal morphological deviation.*
