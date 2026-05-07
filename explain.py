# ==========================================
# EXPLAIN - Model Explainability (Shared)
# ==========================================
# Model-agnostic gradient x input attribution.
# Works with any PyTorch model that takes
# (ecg, clinical) inputs.
# ==========================================

import numpy as np
import torch

from early_fusion.config import CLINICAL_FEATURES, NORMAL_RANGES


FRIENDLY_NAMES = {
    "anchor_age": "Age",
    "gender": "Gender",
    "Heart_Rate": "Heart Rate",
    "Respiratory_Rate": "Respiratory Rate",
    "Troponin_T": "Troponin T",
    "log1p_Troponin_T": "Log Troponin T",
    "Troponin_T_positive": "Troponin T Positive",
    "Troponin_T_high": "Troponin T High",
    "troponin_first_24h": "First Troponin T in 24h",
    "troponin_max_24h": "Max Troponin T in 24h",
    "troponin_count_24h": "Troponin T Count in 24h",
    "hours_admit_to_first_troponin": "Hours to First Troponin",
    "troponin_24h_missing": "Troponin T Missing in 24h",
    "Troponin_T_mild_positive": "Mild Troponin T Positive",
    "Troponin_T_moderate_positive": "Moderate Troponin T Positive",
    "Troponin_T_severe_positive": "Severe Troponin T Positive",
    "age_x_log1p_Troponin_T": "Age x Log Troponin T",
    "Creatinine": "Creatinine",
    "Creatinine_high": "Creatinine High",
    "creatinine_max_24h": "Max Creatinine in 24h",
    "log_troponin_x_creatinine": "Log Troponin x Creatinine",
    "troponin_x_creatinine_high": "Troponin x High Creatinine",
    "Sodium": "Sodium",
    "Potassium": "Potassium",
    "Potassium_low": "Potassium Low",
    "Potassium_high": "Potassium High",
    "potassium_min_24h": "Min Potassium in 24h",
    "hours_admit_to_ecg": "Hours from Admission to ECG",
    "ecg_within_first_6h": "ECG Within First 6h",
    "ecg_within_first_24h": "ECG Within First 24h",
    "ecg_before_admission": "ECG Before Admission",
    "ecg_after_discharge": "ECG After Discharge",
    "ecg_time_missing": "ECG Time Missing",
    "num_diagnoses": "Number of Diagnoses",
    "los": "Length of Stay",
    "PR_interval": "PR Interval",
    "QRS_duration": "QRS Duration",
    "QT_interval": "QT Interval",
    "QTc": "Corrected QT (QTc)",
    "P_axis": "P Axis",
    "QRS_axis": "QRS Axis",
    "T_axis": "T Axis",
    "RR_interval": "RR Interval",
    "HR_from_RR_interval": "Heart Rate from RR",
    "HR_RR_disagreement": "Heart Rate/RR Disagreement",
    "QRS_wide": "Wide QRS",
    "QTc_prolonged": "Prolonged QTc",
    "PR_prolonged": "Prolonged PR",
    "QRS_axis_deviation": "QRS Axis Deviation",
    "T_axis_abnormal": "Abnormal T Axis",
    "ecg_abnormality_count": "ECG Abnormality Count",
    "ecg_any_abnormal": "Any ECG Abnormality",
    "troponin_high_and_ecg_abnormal": "High Troponin with ECG Abnormality",
    "troponin_high_and_t_axis_abnormal": "High Troponin with T Axis Abnormality",
    "troponin_high_and_qrs_wide": "High Troponin with Wide QRS",
    "troponin_high_and_qtc_prolonged": "High Troponin with Prolonged QTc",
    "ecg_abnormal_without_troponin_high": "ECG Abnormality without High Troponin",
    "older_with_ecg_abnormal": "Older Patient with ECG Abnormality",
    "qrs_or_t_axis_abnormal_without_troponin": "QRS/T Axis Abnormality without High Troponin",
    "Troponin_T_missing": "Troponin T Missing",
    "Creatinine_missing": "Creatinine Missing",
    "Sodium_missing": "Sodium Missing",
    "Potassium_missing": "Potassium Missing",
    "Heart_Rate_missing": "Heart Rate Missing",
    "Respiratory_Rate_missing": "Respiratory Rate Missing",
    "PR_interval_invalid": "PR Interval Invalid",
    "QRS_duration_invalid": "QRS Duration Invalid",
    "QT_interval_invalid": "QT Interval Invalid",
    "QTc_invalid": "QTc Invalid",
    "P_axis_invalid": "P Axis Invalid",
    "QRS_axis_invalid": "QRS Axis Invalid",
    "T_axis_invalid": "T Axis Invalid",
    "RR_interval_invalid": "RR Interval Invalid",
}


def _friendly_name(feat):
    return FRIENDLY_NAMES.get(feat, feat.replace("_", " ").title())


# ==========================================
# GRADIENT X INPUT ATTRIBUTION
# ==========================================

def compute_clinical_attributions(model, ecg_tensor, clinical_tensor, device):
    """
    Compute Gradient x Input attributions for clinical features.

    Works with any model that has a forward(ecg, clinical) signature.
    """
    was_training = model.training
    model.eval()

    ecg = ecg_tensor.to(device)
    clinical = clinical_tensor.to(device).detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    # cuDNN RNN kernels cannot run backward while the model is in eval mode.
    # Disable cuDNN only for attribution so dropout stays off and gradients work.
    with torch.backends.cudnn.flags(enabled=False):
        logits = model(ecg, clinical)
        probability = torch.sigmoid(logits).item()
        logits.backward()

    if was_training:
        model.train()

    grad = clinical.grad.detach().cpu().numpy().flatten()
    inp = clinical.detach().cpu().numpy().flatten()
    attributions = np.abs(grad * inp)

    return attributions, probability


# ==========================================
# REASONING GENERATION
# ==========================================

def generate_reasoning(attributions, raw_clinical_values, probability, top_k=5, feature_names=None):
    """
    Generate human-readable reasoning from model attributions.

    Parameters
    ----------
    attributions        : np.ndarray - attribution scores per feature
    raw_clinical_values : dict - {feature_name: raw_value}
    probability         : float - predicted AMI probability
    top_k               : int - number of top features to explain
    feature_names        : list[str] or None - feature order matching attributions

    Returns
    -------
    reasoning : list of dict - ranked explanations
    summary   : str - one-line summary of top reasons
    """
    attr_max = attributions.max()
    normalized = attributions / attr_max if attr_max > 0 else attributions

    feature_scores = []
    active_features = list(feature_names or CLINICAL_FEATURES)
    for i, feat_name in enumerate(active_features):
        if i < len(normalized):
            feature_scores.append({
                "feature": feat_name,
                "attribution": float(normalized[i]),
                "raw_value": float(raw_clinical_values.get(feat_name, 0)),
            })

    feature_scores.sort(key=lambda x: x["attribution"], reverse=True)

    reasoning = []
    for rank, item in enumerate(feature_scores[:top_k], 1):
        feat = item["feature"]
        value = item["raw_value"]
        attr_score = item["attribution"]

        if feat in NORMAL_RANGES:
            lo, hi, unit = NORMAL_RANGES[feat]
            normal_range_str = f"{lo}-{hi}"

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
            explanation = f"{_friendly_name(feat)} = {value:.2f} - model found this relevant"

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
        return f"{name} is {value:.2g} {unit} (normal: {lo}-{hi}) - {qualifier}"
    if "Above" in status:
        return f"{name} is {value:.2g} {unit} (normal: {lo}-{hi}) - elevated"
    if "Below" in status:
        return f"{name} is {value:.2g} {unit} (normal: {lo}-{hi}) - below normal"
    return f"{name} is {value:.2g} {unit} (normal: {lo}-{hi}) - within normal but model-relevant"
