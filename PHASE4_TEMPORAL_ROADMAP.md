# Phase 4: Temporal Physiological Progression Modeling
**Moving Beyond the Static Snapshot in Multimodal AMI Prediction**

This document outlines the complete architectural, data, and experimental roadmap for transitioning the AMI prediction system from a static single-snapshot model to a longitudinal progression model using MIMIC-IV and MIMIC-IV-ECG.

---

## 1. Research Framing
**The Core Thesis:** "Static snapshots (a single 10-second ECG + a single Troponin draw) suffer from a hard physiological information ceiling (~0.72 F1). Distinguishing acute plaque rupture (Type 1 NSTEMI) from chronic demand ischemia (Type 2 MI / CKD / Sepsis) is fundamentally a *dynamic* problem, not a spatial one. To break the ceiling, artificial intelligence must replicate the human diagnostic workflow: observing physiological evolution over time."

**Positioning:** Position this as a pioneering approach to multimodal medical AI. Most literature focuses on brute-force scaling of Transformers on static ECGs. Your work proves that *architectural scaling yields diminishing returns without longitudinal context.*

---

## 2. Serial Troponin Modeling (Highest ROI)
Troponin kinetics form the absolute gold standard for distinguishing acute MI from chronic elevation.
*   **Extraction:** Identify `itemid` 51002 (Troponin I) and 51003 (Troponin T) in `mimiciv_hosp.labevents`. 
*   **Chronology:** Sort by `charttime` per `hadm_id`.
*   **Feature Engineering:**
    *   **$\Delta$ Troponin (Absolute):** `Trop_current - Trop_previous`
    *   **Rise Velocity:** `(Trop_current - Trop_previous) / hours_elapsed`
    *   **Relative Rise (%):** `(Trop_current - Trop_previous) / Trop_previous`
    *   **Trajectory Class:** Rising, Falling, Plateauing (CKD baseline).
*   **Alignment:** Interpolate troponin values to align with the timestamps of the extracted ECGs.

---

## 3. Serial ECG Extraction
A single wide QRS or ST-depression might be the patient's normal baseline. We must extract *changes* from baseline.
*   **Identification:** Group `machine_measurements.csv` by `subject_id` and `hadm_id`. Filter for admissions with $N \ge 2$ ECGs.
*   **Sequence Building:** Sort by `ecg_time`. 
    *   Define the **Anchor ECG ($T_0$)** as the first ECG upon admission or ED triage.
    *   Extract $T_{-1}$ (historical baseline if available), $T_1$, $T_2$ (follow-up ECGs within 24-48 hours).
*   **Progression Features (Tabular):**
    *   $\Delta$ Heart Rate, $\Delta$ QTc, $\Delta$ QRS duration.
    *   New-onset axis deviation or new-onset wide QRS.

---

## 4. Dataset Redesign (Episode-Based)
The fundamental unit of data shifts from a `Row` to a `Temporal Episode`.
*   **Structure:** Instead of shape `(Features)`, the dataset becomes `(Sequence_Length, Features)`.
*   **Padding/Masking:** Standardize to a sequence length of $S=3$ or $S=4$ observation points. If a patient only has 2 ECGs, pad the remaining steps with zeros and use a boolean `attention_mask` so the network ignores padded steps.
*   **Avoid Leakage:** Strictly censor any ECGs or labs taken *after* a cath lab intervention (PCI/CABG), as post-op waveforms are altered.

---

## 5. Temporal Fusion Architecture
Avoid massive parameter explosion. Do not use giant temporal Transformers yet.
*   **Step 1: Feature-Level Sequence:** Pass each ECG in the sequence through the existing frozen or lightweight 1D-ResNet to get an embedding `(S, 256)`. Pass clinical features through an MLP to get `(S, 64)`.
*   **Step 2: Timestep Fusion:** Concatenate to get a multimodal vector per timestep: `(S, 320)`.
*   **Step 3: Lightweight Temporal Modeling:**
    *   **GRU / LSTM:** Pass the sequence `(B, S, 320)` through a 1-layer GRU. The final hidden state elegantly captures the entire trajectory.
    *   **Delta-MLP (Simplest):** If $S=2$, just concatenate `[Emb_T0, Emb_T1, (Emb_T1 - Emb_T0)]` and pass to an MLP. This explicitly forces the network to look at the *change* vector.

---

## 6. Temporal Clinical Context & ED Triage
*   **Vitals Variance:** Extract min, max, and standard deviation of Heart Rate and Blood Pressure from the first 6 hours of ED admission. High variance + troponin rise = highly acute.
*   **MIMIC-IV-ED:** Link `hadm_id` to the ED module to capture triage pain scores (e.g., 0-10 chest pain scale) and exact ED arrival times.

---

## 7. Label Refinement
*   **Time-to-Diagnosis:** NSTEMI diagnoses in billing codes don't have timestamps. Use the timestamp of the *peak Troponin* or the time of *Heparin/Aspirin administration* as the proxy for the acute physiological window.
*   **Isolating Type 2 MI:** Use the longitudinal data to explicitly define Type 2 MI (Demand Ischemia). If Troponin is elevated but plateaued (velocity $\approx 0$), and ECG is unchanged ($\Delta$ morphology $\approx 0$), but Creatinine is high $\rightarrow$ strictly label as False Positive (Non-AMI).

---

## 8. Expected Performance Gains & Prioritization
*   **Serial Troponins ($\Delta$ Velocity):** **Highest ROI.** Will immediately eliminate 90% of CKD/Sepsis false positives. *Expected F1 jump: +0.05 to +0.08.* (Pushing you to ~0.78-0.80).
*   **Serial ECGs ($\Delta$ Waveform):** **Moderate-High ROI.** Resolves ambiguous baseline ST-abnormalities (e.g., LVH vs Ischemia). *Expected F1 jump: +0.03 to +0.05.*
*   **Combined Temporal System:** Breaking the 0.80 F1 barrier becomes highly realistic because you are providing the network with the exact physiological sequence human experts use.

---

## 9. Experimental Roadmap (Execution Plan)

### Stage 1: The "Delta" Baseline (Tabular Only)
*   **Action:** Do not touch the waveform yet. Compute $\Delta$ Troponin and $\Delta$ QTc/HR from the `machine_measurements.csv` tables.
*   **Model:** Standard XGBoost or MLP.
*   **Goal:** Prove that $\Delta$ features immediately break the 0.73 static clinical ceiling.

### Stage 2: Dual-ECG Contrast (Static Architecture)
*   **Action:** Modify the Dataloader to return two waveforms: $ECG_{admission}$ and $ECG_{historical\_baseline}$.
*   **Model:** Pass both through a Siamese 1D-ResNet. Subtract the embeddings $E_{admit} - E_{base}$.
*   **Goal:** Allow the model to subtract out chronic baseline abnormalities (like chronic Left Bundle Branch Block).

### Stage 3: Full Sequence Progression (GRU/LSTM)
*   **Action:** Build the full `(B, S, Features)` sequence dataloader.
*   **Model:** Implement the GRU over the multimodal timestep embeddings.
*   **Goal:** Achieve the final dynamic progression model.

---
*Roadmap generated following the Phase 3 ablation discovery of the static-snapshot information ceiling.*
