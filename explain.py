# ==========================================
# EXPLAIN — Model Explainability (Shared)
# ==========================================
# Model-agnostic gradient × input attribution.
# Works with ANY PyTorch model that takes
# (ecg, clinical) inputs.
# ==========================================

import numpy as np
import torch

# ── Clinical feature names & normal ranges (dataset-level, shared) ───────────

CLINICAL_FEATURES = [
    "anchor_age", "gender",
    "Heart_Rate", "Respiratory_Rate",
    "Troponin_T", "Creatinine", "Sodium", "Potassium",
    "num_diagnoses", "los",
    "PR_interval", "QRS_duration", "QT_interval", "QTc",
    "P_axis", "QRS_axis", "T_axis", "RR_interval",
    "Troponin_T_missing", "Creatinine_missing",
    "Sodium_missing", "Potassium_missing",
    "Heart_Rate_missing", "Respiratory_Rate_missing",
]

NORMAL_RANGES = {
    "anchor_age":        (18,  90,   "years"),
    "Heart_Rate":        (60,  100,  "bpm"),
    "Respiratory_Rate":  (12,  20,   "breaths/min"),
    "Troponin_T":        (0,   0.04, "ng/mL"),
    "Creatinine":        (0.6, 1.2,  "mg/dL"),
    "Sodium":            (136, 145,  "mEq/L"),
    "Potassium":         (3.5, 5.0,  "mEq/L"),
    "PR_interval":       (120, 200,  "ms"),
    "QRS_duration":      (60,  120,  "ms"),
    "QT_interval":       (350, 450,  "ms"),
    "QTc":               (350, 460,  "ms"),
    "P_axis":            (0,   75,   "°"),
    "QRS_axis":          (-30, 90,   "°"),
    "T_axis":            (15,  75,   "°"),
    "RR_interval":       (600, 1000, "ms"),
}

FRIENDLY_NAMES = {
    "anchor_age":           "Age",
    "gender":               "Gender",
    "Heart_Rate":           "Heart Rate",
    "Respiratory_Rate":     "Respiratory Rate",
    "Troponin_T":           "Troponin T",
    "Creatinine":           "Creatinine",
    "Sodium":               "Sodium",
    "Potassium":            "Potassium",
    "num_diagnoses":        "Number of Diagnoses",
    "los":                  "Length of Stay",
    "PR_interval":          "PR Interval",
    "QRS_duration":         "QRS Duration",
    "QT_interval":          "QT Interval",
    "QTc":                  "Corrected QT (QTc)",
    "P_axis":               "P Axis",
    "QRS_axis":             "QRS Axis",
    "T_axis":               "T Axis",
    "RR_interval":          "RR Interval",
    "Troponin_T_missing":   "Troponin T (missing)",
    "Creatinine_missing":   "Creatinine (missing)",
    "Sodium_missing":       "Sodium (missing)",
    "Potassium_missing":    "Potassium (missing)",
    "Heart_Rate_missing":   "Heart Rate (missing)",
    "Respiratory_Rate_missing": "Respiratory Rate (missing)",
}


def _friendly_name(feat):
    return FRIENDLY_NAMES.get(feat, feat.replace("_", " ").title())


# ==========================================
# GRADIENT × INPUT ATTRIBUTION
# ==========================================

def compute_clinical_attributions(model, ecg_tensor, clinical_tensor, device):
    """
    Compute Gradient × Input attributions for clinical features.

    Works with ANY model that has a forward(ecg, clinical) signature.

    Parameters
    ----------
    model           : nn.Module (any fusion model)
    ecg_tensor      : torch.Tensor of shape (1, 12, 5000)
    clinical_tensor : torch.Tensor of shape (1, N_features)
    device          : torch.device

    Returns
    -------
    attributions : np.ndarray of shape (N_features,) — |gradient × input|
    probability  : float — sigmoid probability of AMI
    """
    model.eval()

    ecg = ecg_tensor.to(device)
    clinical = clinical_tensor.to(device).requires_grad_(True)

    logits = model(ecg, clinical)
    probability = torch.sigmoid(logits).item()

    logits.backward()

    grad = clinical.grad.detach().cpu().numpy().flatten()
    inp  = clinical.detach().cpu().numpy().flatten()
    attributions = np.abs(grad * inp)

    return attributions, probability


# ==========================================
# REASONING GENERATION
# ==========================================

def generate_reasoning(attributions, raw_clinical_values, probability, top_k=5):
    """
    Generate human-readable reasoning from model attributions.

    Parameters
    ----------
    attributions        : np.ndarray — attribution scores per feature
    raw_clinical_values : dict — {feature_name: raw_value}
    probability         : float — predicted AMI probability
    top_k               : int — number of top features to explain

    Returns
    -------
    reasoning : list of dict — ranked explanations
    summary   : str — one-line summary of top reasons
    """
    # Normalize attributions to 0–1
    attr_max = attributions.max()
    normalized = attributions / attr_max if attr_max > 0 else attributions

    # Build scored list
    feature_scores = []
    for i, feat_name in enumerate(CLINICAL_FEATURES):
        if i < len(normalized):
            feature_scores.append({
                "feature": feat_name,
                "attribution": float(normalized[i]),
                "raw_value": float(raw_clinical_values.get(feat_name, 0)),
            })

    feature_scores.sort(key=lambda x: x["attribution"], reverse=True)

    # Generate explanations
    reasoning = []
    for rank, item in enumerate(feature_scores[:top_k], 1):
        feat = item["feature"]
        value = item["raw_value"]
        attr_score = item["attribution"]

        if feat in NORMAL_RANGES:
            lo, hi, unit = NORMAL_RANGES[feat]
            normal_range_str = f"{lo}–{hi}"

            if value < lo:
                deviation = (lo - value) / max(abs(hi - lo), 1e-8)
                status = "CRITICAL - Far below normal" if deviation > 2 else "Abnormal - Below normal"
                is_abnormal = 1
            elif value > hi:
                deviation = (value - hi) / max(abs(hi - lo), 1e-8)
                status = "CRITICAL - Far above normal" if deviation > 2 else "Abnormal - Above normal"
                is_abnormal = 1
            else:
                status = "Normal"
                is_abnormal = 0

            explanation = _build_explanation(feat, value, unit, lo, hi, status)
        else:
            unit, normal_range_str = "", "N/A"
            lo, hi = 0, 1
            status, is_abnormal = "Indicator", 0
            explanation = f"{_friendly_name(feat)} = {value:.2f} — model found this relevant"

        reasoning.append({
            "rank": rank,
            "feature": feat,
            "display_name": _friendly_name(feat),
            "value": round(value, 4),
            "unit": unit,
            "normal_min": lo,
            "normal_max": hi,
            "normal_range": normal_range_str,
            "is_abnormal": is_abnormal,
            "status": status,
            "attribution_score": round(attr_score, 4),
            "explanation": explanation,
        })

    summary = "; ".join(r["explanation"] for r in reasoning[:3])
    return reasoning, summary


def _build_explanation(feat, value, unit, lo, hi, status):
    """Build natural language explanation for a single feature."""
    name = _friendly_name(feat)
    if "CRITICAL" in status:
        qualifier = "critically elevated" if "above" in status.lower() else "critically low"
        return f"{name} is {value:.2g} {unit} (normal: {lo}–{hi}) — {qualifier}"
    elif "Above" in status:
        return f"{name} is {value:.2g} {unit} (normal: {lo}–{hi}) — elevated"
    elif "Below" in status:
        return f"{name} is {value:.2g} {unit} (normal: {lo}–{hi}) — below normal"
    else:
        return f"{name} is {value:.2g} {unit} (normal: {lo}–{hi}) — within normal but model-relevant"
