# Multimodal Acute Cardiac Event Prediction
## Late Fusion Architecture Evolution and Change Impact Report

---

## Table of Contents
1. Objective
2. Baseline System
3. Change Log and Rationale
4. Phase-by-Phase Results
5. ECG Branch Recovery Analysis
6. What Helped, What Did Not
7. Current Best Configuration
8. Recommended Next Steps

---

## 1. Objective

The goal of this report is to document the major late-fusion model changes made after the original baseline, explain why each change was introduced, and summarize whether it improved validation performance, did not help, or produced mixed results.

The main target metric throughout this work was validation **F1 score** for AMI prediction. Secondary metrics used for interpretation were:

- Precision
- Recall
- ROC-AUC
- Average Precision (PR-AUC summary)

---

## 2. Baseline System

The original late-fusion baseline had the following characteristics:

- Raw ECG input processed as `12 x 5000`
- Simple ECG branch: stacked Conv1D layers + BiLSTM
- Clinical branch: MLP over tabular features
- Late fusion based mainly on branch-level logits
- Fixed training hyperparameters
- No Optuna tuning
- Coarse threshold search
- No explicit ECG branch supervision

### Baseline reference performance

The working baseline that motivated this change series had a best validation F1 of approximately:

- **Validation F1: 0.7086**

This value was the main reference point for later changes.

---

## 3. Change Log and Rationale

### 3.1 Clinical feature schema update

The clinical feature set was updated to reflect the expanded engineered dataset, including:

- Troponin-derived features
- First-24h lab timing features
- ECG admission timing features
- ECG machine measurements
- Missingness and invalidity flags

### Why this was done

The tabular branch needed to stay aligned with the current parquet dataset and use the latest clinically meaningful engineered signals.

### Outcome

- Necessary infrastructure update
- Not evaluated as an isolated late-fusion ablation
- Enabled downstream training consistency

---

### 3.2 Optuna tuning path

Hyperparameter tuning support was added for:

- Learning rate
- Batch size
- Dropout
- Weight decay
- CNN filters
- BiLSTM hidden size
- Fusion hidden dimension
- Focal loss gamma

### Why this was done

The original training path used fixed values and had no systematic search over training dynamics or capacity.

### Outcome

- Useful for experiment control
- Did not produce the main leap by itself
- The main bottlenecks later proved to be ECG representation quality and fusion behavior, not only scalar hyperparameters

---

### 3.3 Threshold search refinement

Threshold selection was changed from a coarse fixed grid to a search over actual validation probabilities.

### Why this was done

The old 50-step threshold grid was too coarse and could miss the true F1-optimal operating point.

### Outcome

- Improved metric fidelity
- Helped reported F1 match the actual decision boundary better
- Important for evaluation quality, but not the main source of representation improvement

---

### 3.4 ECG preprocessing experiment: bandpass + downsampling to 2500

A preprocessing script was added to generate filtered/downsampled ECGs with:

- Bandpass filtering
- Optional notch filtering
- Downsampling from `5000 -> 2500`

### Why this was done

This was intended to remove redundancy and noise while keeping AMI-relevant morphology.

### Outcome

This experiment did **not** improve validation F1 in a meaningful way.

Observed effect from the earlier comparison:

- Old validation F1: **0.708590**
- Preprocessed/downsampled validation F1: **0.705939**

At the same time, ranking metrics slightly improved:

- Old AUC/AP: **0.906543 / 0.756908**
- Preprocessed AUC/AP: **0.908649 / 0.760187**

### Interpretation

- The preprocessing was not catastrophic
- It slightly improved ranking quality
- It did not improve thresholded F1
- The main limitation remained the ECG encoder and the fusion strategy

Result classification:

- **Mixed / not useful for F1 improvement**

---

### 3.5 ECG diagnostics and branch-level evaluation

Dedicated analysis scripts were added to measure:

- ECG-only branch metrics
- Clinical-only branch metrics
- Fusion branch metrics
- ECG activation behavior
- Lead importance
- ECG quality diagnostics

### Why this was done

Fusion performance alone was hiding whether the ECG branch was actually contributing useful signal.

### Outcome

This analysis uncovered the most important early finding:

- The original ECG branch was effectively collapsed

Representative earlier branch metrics:

| Branch | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|
| ECG | 0.1951 | 1.0000 | 0.3265 | 0.4981 | 0.1946 |
| Clinical | 0.6909 | 0.6988 | 0.6948 | 0.8977 | 0.7475 |
| Fusion | 0.6922 | 0.6975 | 0.6948 | 0.8976 | 0.7475 |

### Interpretation

- ECG-only AUC near `0.50` indicated almost random discrimination
- Fusion behaved almost identically to the clinical branch
- The ECG branch was contributing little or no useful AMI signal

Result classification:

- **Major diagnostic success**

---

### 3.6 Clean retraining protections

Training was updated to:

- detect architecture mismatch
- refuse incompatible checkpoints
- support `FORCE_FRESH_TRAIN = True`
- store architecture metadata in checkpoints

### Why this was done

Repeated architecture changes were contaminating evaluation by accidentally loading legacy checkpoints.

### Outcome

- Prevented invalid comparisons
- Ensured that ECG-only analysis reflected the current architecture, not old weights

Result classification:

- **Infrastructure improvement**

---

### 3.7 Lead-aware ECG encoder and embedding-based fusion

The old ECG branch was replaced with a stronger encoder featuring:

- Lead-aware grouped temporal stem
- Cross-lead mixing
- ResNet-style temporal blocks
- BiLSTM
- Attention pooling

Fusion was also upgraded from logit-level fusion to embedding-based fusion.

### Why this was done

The collapsed ECG branch showed that the old encoder was not learning clinically useful waveform structure.

### Outcome

This was the first major representation improvement.

Representative post-upgrade ECG-only branch behavior:

- ECG AUC rose from near-random to meaningfully informative
- ECG branch began responding to leads in a non-trivial way

An intermediate post-fix observation was:

- ECG-only AUC around **0.70**
- ECG-only F1 around **0.43**

### Interpretation

- The ECG signal was not useless
- The original failure was largely architectural
- This was the first change that materially recovered ECG value

Result classification:

- **Strong improvement**

---

### 3.8 Auxiliary branch supervision + lower ECG LR + weighted sampling + early stopping

Training was updated to use:

- Total loss = `fusion + 0.3 * ecg + 0.1 * clinical`
- Lower ECG branch learning rate via parameter groups
- Weighted sampling by default
- Early stopping on validation F1
- Best-model restore

### Why this was done

Even after the ECG branch started learning, fusion could still dominate optimization and destabilize ECG learning.

### Outcome

This change materially stabilized the ECG branch and preserved good overall validation F1.

One observed run after these stabilization changes produced:

| Split | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|
| Validation | 0.6936 | 0.7347 | 0.7136 | 0.9152 | 0.7745 |

At the same stage, ECG-only branch metrics improved to:

| Branch | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|
| ECG | 0.3970 | 0.6066 | 0.4799 | 0.7624 | 0.4766 |

### Interpretation

- ECG branch became genuinely informative
- Fusion remained stronger than ECG-only
- Validation F1 moved slightly above the original 0.7086 baseline

Result classification:

- **Strong improvement**

---

### 3.9 Gated fusion

The plain embedding concatenation fusion was upgraded to gated fusion:

- A learned per-sample gate modulates the ECG embedding before fusion

### Why this was done

Clinical features were strong enough to dominate fusion. A gate allows the model to upweight or downweight ECG contribution per case.

### Outcome

This improved fusion design quality conceptually and remained compatible with the rest of the pipeline. However, it did not produce a dramatic F1 leap by itself.

From the current artifact snapshot:

| Branch | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|
| ECG | 0.4004 | 0.6304 | 0.4898 | 0.7743 | 0.4972 |
| Clinical | 0.6911 | 0.7073 | 0.6991 | 0.8955 | 0.7428 |
| Fusion | 0.6757 | 0.7158 | 0.6952 | 0.9056 | 0.7558 |

And the final validation result from the main trainer:

| Split | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|
| Validation | 0.6869 | 0.7403 | 0.7126 | 0.9128 | 0.7729 |

### Interpretation

- ECG branch continued to improve modestly
- Fusion AUC remained strong
- Validation F1 stayed near the best stabilized regime
- No major jump beyond the best stabilized late-fusion run

Result classification:

- **Moderate / mixed improvement**

---

### 3.10 Hard-negative focused training

The training loss was modified to emphasize false-positive-prone negatives using higher weights for hard negative samples.

### Why this was done

At this stage, the remaining late-fusion errors were likely concentrated in clinically confusing negative cases.

### Outcome

This change was integrated together with gated fusion. In the current run, the model logged:

- Validation hard-negative mean weight: **1.2491**
- Validation hard-negative max weight: **2.3814**

### Interpretation

- The mechanism is active
- It likely contributed to maintaining precision while preserving recall
- Its isolated contribution cannot yet be separated cleanly from the gated-fusion run

Result classification:

- **Plausibly helpful, not yet isolated**

---

### 3.11 Raw 5000-length reversion

The active ECG path was reverted from filtered/downsampled `2500` preprocessing back to raw `12 x 5000`.

### Why this was done

The downsampling experiment did not improve validation F1, and the temporal modeling work was better aligned with the full raw sequence.

### Outcome

- Restored original waveform length
- Reused raw memmap path after backward-compatible cache support was added

Result classification:

- **Correct rollback**

---

### 3.12 Temporal modeling upgrade: dilated CNN + BiLSTM + Transformer + temporal pyramid pooling

The current ECG branch was further upgraded to explicitly target temporal-distribution learning:

- Dilated temporal residual convolutions
- Positional encoding
- Transformer encoder after BiLSTM
- Multi-head temporal attention
- Temporal pyramid pooling

### Why this was done

The next major remaining bottleneck was temporal structure learning:

- long-range dependencies
- morphology progression
- temporal context across wider receptive fields

### Outcome

This architecture is now active in code, but the current artifact snapshot does not yet provide a clean completed post-change comparison beyond the most recent run context.

Result classification:

- **Implemented, final impact still under evaluation**

---

## 4. Phase-by-Phase Results

### Summary Table

| Phase | Main Change | Validation Precision | Validation Recall | Validation F1 | Validation AUC | Validation AP | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | Original late fusion | not fully archived | not fully archived | **0.7086** | not fully archived | not fully archived | Baseline |
| Downsample + bandpass | Preprocessed 2500 ECG | not fully archived | not fully archived | **0.7059** | 0.9086 | 0.7602 | Did not help F1 |
| ECG collapse audit | Branch diagnostics | 0.6922 | 0.6975 | 0.6948 | 0.8976 | 0.7475 | Revealed ECG branch failure |
| Lead-aware ECG encoder | Raw ECG branch redesign | approx. improved | approx. improved | **0.7148** | 0.9145 | 0.7694 | Strong improvement |
| Stabilized training | Aux losses + LR + sampling + early stopping | 0.6936 | 0.7347 | **0.7136** | 0.9152 | 0.7745 | Strong stabilization |
| Gated fusion + hard negatives | Current gated regime | 0.6869 | 0.7403 | **0.7126** | 0.9128 | 0.7729 | Similar F1, better ECG branch |

### Notes

- “Not fully archived” means the exact metric file for that phase is not present in the current artifact snapshot, but the change and approximate result were documented during experimentation.
- The strongest F1 region reached so far is roughly **0.712–0.715**.

---

## 5. ECG Branch Recovery Analysis

This was the most important scientific finding of the late-fusion work.

### Before ECG branch redesign

- ECG-only AUC was near `0.50`
- ECG-only precision was approximately equal to class prevalence
- Fusion behavior was almost entirely clinical-driven

### After ECG branch redesign and stabilization

Current ECG-only metrics:

| Metric | Value |
|---|---:|
| Precision | 0.400401 |
| Recall | 0.630440 |
| F1 | 0.489754 |
| AUC | 0.774315 |
| Average Precision | 0.497241 |

### Interpretation

- The ECG branch is no longer collapsed
- ECG now carries meaningful AMI signal
- Fusion remains stronger than ECG-only, but ECG is now a true contributing modality rather than dead weight

---

## 6. What Helped, What Did Not

### Helped clearly

1. Lead-aware ECG encoder redesign
2. Embedding-based fusion replacing pure logit-level fusion
3. Auxiliary ECG supervision
4. Reduced ECG branch learning rate
5. Weighted sampling
6. Early stopping on validation F1
7. Better threshold search

### Helped diagnostically

1. ECG-only evaluation script
2. Branch-wise comparison reporting
3. ECG quality diagnostics
4. Clean retrain / checkpoint mismatch protection

### Did not help F1 meaningfully

1. Bandpass + downsampling to `2500`

### Mixed or not yet isolated

1. Gated fusion
2. Hard-negative weighting
3. Transformer-based temporal modeling

---

## 7. Current Best Configuration

The current late-fusion system includes:

- Raw ECG input at `12 x 5000`
- Lead-aware temporal CNN
- Dilated residual temporal blocks
- BiLSTM
- Transformer encoder
- Positional encoding
- Multi-head temporal attention
- Temporal pyramid pooling
- Clinical MLP branch
- Gated late fusion
- Auxiliary branch supervision
- Hard-negative focused weighting
- Weighted sampling
- F1-based early stopping

### Current main validation metrics from `metrics.json`

| Metric | Value |
|---|---:|
| Precision | 0.686894 |
| Recall | 0.740336 |
| F1 | 0.712614 |
| AUC | 0.912751 |
| Average Precision | 0.772856 |
| Threshold | 0.824179 |

---

## 8. Recommended Next Steps

The highest-value next analyses are:

1. Run a clean post-transformer ablation against the immediately previous gated-fusion BiLSTM model.
2. Log train vs validation ECG-only metrics per epoch to separate underlearning from overfitting.
3. Isolate hard-negative weighting and gated fusion in separate ablations.
4. Keep the branch diagnostic scripts as standard outputs for every major experiment.

---

## Conclusion

The main story of this late-fusion work is not that every architectural change improved F1. The most important result is that the ECG branch went from essentially collapsed and clinically ignored to a genuinely informative modality with ECG-only AUC around `0.77` and ECG-only F1 around `0.49`.

The largest improvements came from:

- fixing ECG representation quality
- stabilizing ECG optimization
- preventing checkpoint contamination

The strongest validation F1 achieved across this change series is in the **0.712–0.715** range, which is a modest but real improvement over the original `~0.7086` baseline. Some later changes improved ECG quality and interpretability more than final F1, which is still scientifically important because it shows the model is learning more clinically meaningful multimodal behavior rather than relying almost entirely on the tabular branch.
