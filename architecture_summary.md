# Multimodal AMI Prediction Architecture Summary

This document summarizes the current model design for both fusion strategies in the repository.

## Task Definition

The project predicts binary AMI status from:

- ECG waveform input with shape `(12, 5000)`
- 24 structured clinical features

The main research comparison is the fusion stage:

- early fusion: combine modalities before shared temporal modeling is complete
- late fusion: keep modality branches separate until the prediction head

## Early Fusion

Source: [early_fusion/model.py](/abs/path/d:/MINI_PROJECT/early_fusion/model.py:1)

### Current design

```text
Clinical features (24,)
-> Linear -> ReLU -> Dropout
-> Linear -> ReLU
-> projected clinical channels
-> repeat across ECG time axis

ECG waveform (12, 5000)
-> concatenate with projected clinical channels
-> Conv1D block
-> Conv1D block
-> Conv1D block
-> BiLSTM
-> dense head
-> final AMI logit
```

### Why it is early fusion

Clinical information is injected before the shared Conv1D and BiLSTM stack has finished extracting temporal features. The temporal encoder therefore learns from a fused representation instead of separate modality branches.

### Strengths

- gives the sequence encoder direct access to both modalities
- can learn ECG-clinical interactions early
- keeps the final prediction head simple

### Tradeoffs

- harder to separate modality-specific contributions
- clinical information is forced into a temporal channel representation

## Late Fusion

Source: [late_fusion/model.py](/abs/path/d:/MINI_PROJECT/late_fusion/model.py:1)

### Current design

```text
ECG branch
-> Conv1D stack
-> BiLSTM
-> attention pooling
-> ECG embedding (128)
-> ECG branch logit

Clinical branch
-> MLP
-> clinical embedding (64)
-> clinical branch logit

Fusion head
-> concatenate ECG embedding, clinical embedding, ECG logit, clinical logit
-> MLP fusion head
-> final AMI logit
```

### Why it is late fusion

The ECG branch and clinical branch learn independently for most of the network. Fusion happens only after each branch has already formed its own representation and branch-level score.

### What changed from the older design

The current late-fusion model no longer fuses only two scalar logits. It now fuses:

- ECG embedding
- clinical embedding
- ECG branch logit
- clinical branch logit

That makes the fusion head substantially more expressive while still preserving a clean late-fusion structure.

### Strengths

- preserves modality-specific reasoning longer
- makes the late-fusion comparison fairer against the stronger early-fusion backbone
- gives the final head richer information than a logits-only design

### Tradeoffs

- slightly larger decision head
- still depends on the fusion head to combine modality evidence well

## Side-by-Side Comparison

| Aspect | Early fusion | Late fusion |
|---|---|---|
| Fusion point | Before shared temporal modeling finishes | After branch embeddings and branch logits are formed |
| ECG encoder | Shared fused encoder | ECG-only encoder |
| Clinical encoder | Small projector into temporal channels | Independent MLP branch |
| Joint representation | Learned early | Learned near the output |
| Interpretability | Lower modality separation | Better modality separation |

## Training and Evaluation Setup

Both pipelines support:

- train/validation splitting from the same prepared dataset style
- threshold tuning on validation predictions
- focal loss or BCE
- optional weighted sampling
- resumable single-run training
- cross-validation entrypoints

Entrypoints:

- [early_fusion/train.py](/abs/path/d:/MINI_PROJECT/early_fusion/train.py:1)
- [early_fusion/cross_validate.py](/abs/path/d:/MINI_PROJECT/early_fusion/cross_validate.py:1)
- [late_fusion/train.py](/abs/path/d:/MINI_PROJECT/late_fusion/train.py:1)
- [late_fusion/cross_validate.py](/abs/path/d:/MINI_PROJECT/late_fusion/cross_validate.py:1)

## Fair Comparison Guidance

For a meaningful early-vs-late comparison, keep these aligned:

- same train/test split policy
- same preprocessing inputs
- same loss family when comparing
- same threshold-selection logic
- same evaluation metrics
- similar epoch budget and regularization budget

## Recommended Metrics

Primary metrics used in the codebase:

- F1
- ROC-AUC
- average precision
- accuracy
- precision
- recall

Because AMI detection is imbalanced and clinically recall-sensitive, F1 and average precision are usually more informative than accuracy alone.
