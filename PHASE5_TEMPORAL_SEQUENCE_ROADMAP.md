# Phase 5: True Temporal Trajectory Modeling (The Path to 0.84)

## The Scientific Reality
Through exhaustive experimentation in Phase 4, we have mathematically proven that **static physiological snapshots have limited diagnostic entropy**. 

Despite engineering state-of-the-art spatial extraction (Cross-Attention Siamese) and latent-space stabilization (Contrastive Margin Loss), the model hit a hard ceiling at **~0.715 Validation F1**. 
This is because NSTEMIs and complex ischemia often present with **no morphological change compared to baseline**. The diagnostic signal exists entirely in the *dynamic evolution* of the physiology over the next 6-12 hours.

## The Strategy
To cross the 0.80 F1 threshold, we are officially abandoning the `[Baseline + Admission]` static paradigm and transitioning to **Multi-Timepoint Sequence Modeling**.

### 1. Serial Troponin Sequence Dataset (Tabular Trajectories)
Instead of compressing biomarker progression into static engineered features (`troponin_velocity`), we will feed the actual sequence of measurements with their explicit timing: `[trop_0, trop_1, trop_2]`.
- **Why:** Allows the network to differentiate a flat elevated CKD troponin curve from a sharp dynamic NSTEMI rise/fall.

### 2. Multi-ECG Trajectory Extraction
We will move from `ECG_base -> ECG_admit` to full admission trajectories: `ECG_t0 -> ECG_t1 -> ECG_t2`.
- **Why:** Serial ECGs capture evolving ST shifts and T-wave inversions that are the hallmark of acute ischemia.

### 3. Temporal Multimodal Fusion Engine (GRU/LSTM)
We will fundamentally alter the architecture:
- At each timestep $t$: `ECG_encoder(ECG_t) + Clinical_encoder(Clinical_t) -> Vector_t`
- Then, a Temporal GRU or Temporal Attention layer processes the sequence `[Vector_0, Vector_1, Vector_2]`.
- **Why:** This mirrors the actual clinical reasoning of a cardiologist tracking patient progression.

### 4. Hard-Negative Temporal Contrastive Learning
We will train the Temporal GRU using Contrastive Loss explicitly on the trajectories:
- Push apart a stable sequence (Chronic Injury) from a dynamically evolving sequence (Acute Ischemia).

## Next Immediate Steps
1. Re-run MIMIC-IV data extraction pipelines to collect `(ECG, Troponin, Timestamps)` for $t_0, t_1, t_2$ across the first 24 hours of admission.
2. Build `TemporalDataset` wrapper for PyTorch.
3. Architect the `TemporalFusionGRU` in `engine.py`.
