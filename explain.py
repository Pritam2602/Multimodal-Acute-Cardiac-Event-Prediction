# ==========================================
# EXPLAIN - Model Explainability (Shared)
# ==========================================
# Model-agnostic gradient x input attribution.
# Works with any PyTorch model that takes
# (ecg, clinical) inputs.
# ==========================================

import numpy as np
import torch

from Dataset.feature_schema import CLINICAL_FEATURES, NORMAL_RANGES


FRIENDLY_NAMES = {
    "anchor_age": "Age",
    "gender": "Gender",
    "Heart_Rate": "Heart Rate",
    "Respiratory_Rate": "Respiratory Rate",
    "Troponin_T": "Troponin T",
    "log1p_Troponin_T": "Log Troponin T",
    "Troponin_T_positive": "Troponin T Positive",
    "Troponin_T_high": "Troponin T High",
    "age_x_log1p_Troponin_T": "Age x Log Troponin T",
    "Creatinine": "Creatinine",
    "Creatinine_high": "Creatinine High",
    "Sodium": "Sodium",
    "Potassium": "Potassium",
    "Potassium_low": "Potassium Low",
    "Potassium_high": "Potassium High",
    "troponin_first_24h": "First 24h Troponin",
    "troponin_max_24h": "Max 24h Troponin",
    "troponin_count_24h": "24h Troponin Count",
    "hours_admit_to_first_troponin": "Hours Admit To First Troponin",
    "troponin_24h_missing": "24h Troponin Missing",
    "creatinine_max_24h": "Max 24h Creatinine",
    "potassium_min_24h": "Min 24h Potassium",
    "hours_admit_to_ecg": "Hours Admit To ECG",
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
    "Troponin_T_missing": "Troponin T (missing)",
    "Creatinine_missing": "Creatinine (missing)",
    "Sodium_missing": "Sodium (missing)",
    "Potassium_missing": "Potassium (missing)",
    "Heart_Rate_missing": "Heart Rate (missing)",
    "Respiratory_Rate_missing": "Respiratory Rate (missing)",
    "PR_interval_invalid": "PR Interval Invalid",
    "QRS_duration_invalid": "QRS Duration Invalid",
    "QT_interval_invalid": "QT Interval Invalid",
    "QTc_invalid": "QTc Invalid",
    "P_axis_invalid": "P Axis Invalid",
    "QRS_axis_invalid": "QRS Axis Invalid",
    "T_axis_invalid": "T Axis Invalid",
    "RR_interval_invalid": "RR Interval Invalid",
    "QRS_wide": "Wide QRS",
    "QTc_prolonged": "QTc Prolonged",
    "PR_prolonged": "PR Prolonged",
    "QRS_axis_deviation": "QRS Axis Deviation",
    "T_axis_abnormal": "T Axis Abnormal",
    "HR_from_RR_interval": "Heart Rate From RR",
    "HR_RR_disagreement": "Heart Rate RR Disagreement",
}


def _friendly_name(feat):
    return FRIENDLY_NAMES.get(feat, feat.replace("_", " ").title())


def compute_clinical_attributions(model, ecg_tensor, clinical_tensor, device):
    """
    Compute gradient x input attributions for clinical features.

    Returns
    -------
    attributions : np.ndarray of shape (N_features,)
    probability  : float
    """
    model.eval()

    ecg = ecg_tensor.to(device)
    clinical = clinical_tensor.to(device).requires_grad_(True)

    logits = model(ecg, clinical)
    probability = torch.sigmoid(logits).item()

    logits.backward()

    grad = clinical.grad.detach().cpu().numpy().flatten()
    inp = clinical.detach().cpu().numpy().flatten()
    attributions = np.abs(grad * inp)

    return attributions, probability


def generate_reasoning(attributions, raw_clinical_values, probability, top_k=5):
    """
    Generate human-readable reasoning from model attributions.
    """
    attr_max = attributions.max()
    normalized = attributions / attr_max if attr_max > 0 else attributions

    feature_scores = []
    for i, feat_name in enumerate(CLINICAL_FEATURES):
        if i < len(normalized):
            feature_scores.append(
                {
                    "feature": feat_name,
                    "attribution": float(normalized[i]),
                    "raw_value": float(raw_clinical_values.get(feat_name, 0)),
                }
            )

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

        reasoning.append(
            {
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
            }
        )

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
