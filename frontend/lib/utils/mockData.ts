export type RiskLevel = "Low" | "Moderate" | "High" | "Critical";

export interface ClinicalTimestep {
  timestep: number;
  label: string;
  time_delta_hrs: number;
  trop_value: number;
  peak_baseline_ratio: number;
  fold_rise: number;
  hr: number;
  sbp: number;
  dbp: number;
  map: number;
  lactate: number;
  creatinine: number;
  ckd_active: boolean;
  sepsis_active: boolean;
}

export interface AttentionWeights {
  leads: Record<string, number>;
  temporal: number[];
  dominant_region: "Inferior" | "Anterior/Septal" | "Lateral" | "Mixed";
}

export interface PredictionData {
  predicted_prob: number;
  threshold: number;
  confidence_evolution: number[];
  dominant_modality: "ECG" | "Troponin" | "Balanced";
  attn_temp_entropy: number;
  attn_spatial_dominance: number;
  ecg_contribution: number;
  trop_contribution: number;
}

export interface Admission {
  hadm_id: string;
  subject_id: string;
  age: number;
  gender: "M" | "F";
  admittime: string;
  ground_truth_ami: boolean;
  max_troponin: number;
  ecg_count: number;
  timesteps: string[];
  risk_level: RiskLevel;
  comorbidities: string[];
  timelines: ClinicalTimestep[];
  prediction: PredictionData;
  attention: AttentionWeights;
  explanation: string;
  explanation_temporal: string;
}

const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

function makeInferiorAttention(): Record<string, number> {
  return {
    I: 0.22, II: 0.91, III: 0.84, aVR: 0.15, aVL: 0.20, aVF: 0.89,
    V1: 0.28, V2: 0.25, V3: 0.31, V4: 0.38, V5: 0.30, V6: 0.27,
  };
}
function makeAnteriorAttention(): Record<string, number> {
  return {
    I: 0.38, II: 0.31, III: 0.24, aVR: 0.18, aVL: 0.42, aVF: 0.28,
    V1: 0.72, V2: 0.89, V3: 0.93, V4: 0.86, V5: 0.67, V6: 0.44,
  };
}
function makeLateralAttention(): Record<string, number> {
  return {
    I: 0.88, II: 0.34, III: 0.21, aVR: 0.19, aVL: 0.82, aVF: 0.26,
    V1: 0.28, V2: 0.31, V3: 0.41, V4: 0.58, V5: 0.87, V6: 0.91,
  };
}
function makeLowAttention(): Record<string, number> {
  return Object.fromEntries(LEADS.map(l => [l, 0.1 + Math.random() * 0.3]));
}

export const MOCK_ADMISSIONS: Admission[] = [
  {
    hadm_id: "HADM-100234",
    subject_id: "P-7742",
    age: 67,
    gender: "M",
    admittime: "2024-01-15 03:42",
    ground_truth_ami: true,
    max_troponin: 0.45,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "High",
    comorbidities: ["Heart Failure", "Hypertension"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.12, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 98, sbp: 148, dbp: 90, map: 109, lactate: 1.8, creatinine: 1.2, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.2, trop_value: 0.28, peak_baseline_ratio: 2.33, fold_rise: 2.33, hr: 104, sbp: 138, dbp: 84, map: 102, lactate: 2.1, creatinine: 1.3, ckd_active: false, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 6.8, trop_value: 0.45, peak_baseline_ratio: 3.75, fold_rise: 3.75, hr: 112, sbp: 128, dbp: 78, map: 95, lactate: 2.4, creatinine: 1.4, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.824, threshold: 0.48, confidence_evolution: [0.24, 0.61, 0.824], dominant_modality: "ECG", attn_temp_entropy: 0.32, attn_spatial_dominance: 0.78, ecg_contribution: 0.68, trop_contribution: 0.32 },
    attention: { leads: makeInferiorAttention(), temporal: [0.18, 0.52, 0.30], dominant_region: "Inferior" },
    explanation: "The model detected evolving inferior-wall ischemic morphology — attention peaked on Leads II and aVF at T₁ — alongside a rapidly rising troponin velocity (+0.165 ng/mL/hr). The absence of active CKD or sepsis flags allowed the model to attribute the troponin rise to acute myocardial injury rather than a metabolic confounder, yielding a final AMI probability of 82.4%.",
    explanation_temporal: "At T₀ the model was uncertain (24%) given borderline troponin. The pivotal T₁ ECG revealed dynamic ST-elevation in the inferior leads, causing the model's confidence to surge to 61%. By T₂, the continued troponin trajectory and morphological evolution locked in the 82.4% prediction.",
  },
  {
    hadm_id: "HADM-109441",
    subject_id: "P-8821",
    age: 58,
    gender: "F",
    admittime: "2024-01-16 11:15",
    ground_truth_ami: false,
    max_troponin: 0.31,
    ecg_count: 2,
    timesteps: ["T₀", "T₁"],
    risk_level: "Moderate",
    comorbidities: ["CKD Stage 3", "Diabetes"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.24, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 82, sbp: 162, dbp: 96, map: 118, lactate: 1.4, creatinine: 2.8, ckd_active: true, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 4.5, trop_value: 0.31, peak_baseline_ratio: 1.29, fold_rise: 1.29, hr: 79, sbp: 158, dbp: 94, map: 115, lactate: 1.6, creatinine: 2.9, ckd_active: true, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.187, threshold: 0.48, confidence_evolution: [0.21, 0.187], dominant_modality: "Troponin", attn_temp_entropy: 0.71, attn_spatial_dominance: 0.22, ecg_contribution: 0.31, trop_contribution: 0.69 },
    attention: { leads: makeLowAttention(), temporal: [0.55, 0.45], dominant_region: "Mixed" },
    explanation: "Despite elevated troponin values (0.24–0.31 ng/mL), the model correctly identified this as a CKD false-positive. The ECG showed no dynamic morphological changes across T₀ and T₁ — the waveforms were stable with no ST deviation or evolving T-wave inversions. The model gated out the troponin signal by weighting the stable ECG morphology heavily, yielding a final probability of 18.7% (below the 0.48 threshold).",
    explanation_temporal: "The model's attention entropy was high (0.71) indicating uncertainty — a hallmark of CKD confounded cases. The troponin rise was slow and plateau-like rather than the acute velocity pattern seen in true AMI. Temporal GRU gating suppressed the troponin channel weight at T₁.",
  },
  {
    hadm_id: "HADM-112987",
    subject_id: "P-3321",
    age: 74,
    gender: "M",
    admittime: "2024-01-17 22:08",
    ground_truth_ami: true,
    max_troponin: 1.24,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "Critical",
    comorbidities: ["Prior MI", "Hypertension"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.18, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 110, sbp: 90, dbp: 60, map: 70, lactate: 3.1, creatinine: 1.5, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 2.5, trop_value: 0.64, peak_baseline_ratio: 3.56, fold_rise: 3.56, hr: 118, sbp: 85, dbp: 55, map: 65, lactate: 3.8, creatinine: 1.7, ckd_active: false, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 5.0, trop_value: 1.24, peak_baseline_ratio: 6.89, fold_rise: 6.89, hr: 124, sbp: 88, dbp: 58, map: 68, lactate: 4.2, creatinine: 1.9, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.961, threshold: 0.48, confidence_evolution: [0.42, 0.87, 0.961], dominant_modality: "ECG", attn_temp_entropy: 0.18, attn_spatial_dominance: 0.92, ecg_contribution: 0.74, trop_contribution: 0.26 },
    attention: { leads: makeAnteriorAttention(), temporal: [0.15, 0.48, 0.37], dominant_region: "Anterior/Septal" },
    explanation: "High-confidence anterior STEMI prediction. The model detected progressive ST-elevation in V1-V4 with early hyperacute T-waves at T₀ evolving to full tombstone morphology by T₂. The troponin velocity (∆ = 0.53 ng/mL/hr) was among the highest 5th percentile of the cohort. Spatial attention was almost entirely dominated by the anterior leads.",
    explanation_temporal: "The temporal GRU rapidly escalated confidence from 42% at T₀ (early ST changes) to 87% at T₁ (established STEMI pattern) to 96.1% at T₂. Attention entropy was extremely low (0.18) reflecting the model's near-certainty.",
  },
  {
    hadm_id: "HADM-118203",
    subject_id: "P-5509",
    age: 45,
    gender: "F",
    admittime: "2024-01-18 08:33",
    ground_truth_ami: false,
    max_troponin: 0.06,
    ecg_count: 2,
    timesteps: ["T₀", "T₁"],
    risk_level: "Low",
    comorbidities: [],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.04, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 76, sbp: 122, dbp: 78, map: 93, lactate: 1.1, creatinine: 0.8, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.0, trop_value: 0.06, peak_baseline_ratio: 1.5, fold_rise: 1.5, hr: 74, sbp: 118, dbp: 76, map: 90, lactate: 1.0, creatinine: 0.8, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.043, threshold: 0.48, confidence_evolution: [0.06, 0.043], dominant_modality: "Troponin", attn_temp_entropy: 0.82, attn_spatial_dominance: 0.14, ecg_contribution: 0.28, trop_contribution: 0.72 },
    attention: { leads: makeLowAttention(), temporal: [0.52, 0.48], dominant_region: "Mixed" },
    explanation: "Very low AMI probability. Troponin values remained below the 99th percentile reference range (0.014 ng/mL) with negligible rise kinetics. The 12-lead ECG showed normal sinus rhythm with no ischemic changes across both timesteps. The model's diffuse, low-magnitude attention pattern reflects the absence of any focal ischemic signal.",
    explanation_temporal: "Confidence actually decreased marginally from T₀ to T₁ as the model's temporal GRU confirmed the absence of troponin velocity. Spatial attention was diffuse with entropy near maximum, indicative of a normal non-ischemic ECG.",
  },
  {
    hadm_id: "HADM-121456",
    subject_id: "P-6614",
    age: 62,
    gender: "M",
    admittime: "2024-01-19 14:20",
    ground_truth_ami: true,
    max_troponin: 0.88,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "High",
    comorbidities: ["Hypertension", "Dyslipidemia"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.08, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 88, sbp: 155, dbp: 92, map: 113, lactate: 1.6, creatinine: 1.1, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.5, trop_value: 0.34, peak_baseline_ratio: 4.25, fold_rise: 4.25, hr: 96, sbp: 142, dbp: 88, map: 106, lactate: 2.0, creatinine: 1.2, ckd_active: false, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 7.2, trop_value: 0.88, peak_baseline_ratio: 11.0, fold_rise: 11.0, hr: 103, sbp: 134, dbp: 82, map: 99, lactate: 2.3, creatinine: 1.3, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.876, threshold: 0.48, confidence_evolution: [0.19, 0.54, 0.876], dominant_modality: "Balanced", attn_temp_entropy: 0.28, attn_spatial_dominance: 0.69, ecg_contribution: 0.51, trop_contribution: 0.49 },
    attention: { leads: makeLateralAttention(), temporal: [0.16, 0.45, 0.39], dominant_region: "Lateral" },
    explanation: "High-confidence lateral wall AMI. The model detected progressive lateral ST-elevation in leads I, aVL, V5, and V6 with reciprocal changes in the inferior leads. A massive 11-fold troponin rise by T₂ provided strong corroborating biochemical evidence. Both modalities contributed approximately equally to the final decision.",
    explanation_temporal: "The T₀ ECG showed subtle lateral ST changes initially dismissed as non-specific. The T₁ ECG showed overt lateral STEMI morphology, driving confidence to 54%. The extreme troponin velocity at T₂ sealed the 87.6% prediction.",
  },
  {
    hadm_id: "HADM-124788",
    subject_id: "P-2287",
    age: 71,
    gender: "F",
    admittime: "2024-01-20 19:55",
    ground_truth_ami: false,
    max_troponin: 0.58,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "Moderate",
    comorbidities: ["CKD Stage 4", "Sepsis", "Anemia"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.42, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 104, sbp: 100, dbp: 62, map: 75, lactate: 3.4, creatinine: 4.1, ckd_active: true, sepsis_active: true },
      { timestep: 1, label: "T₁", time_delta_hrs: 4.0, trop_value: 0.51, peak_baseline_ratio: 1.21, fold_rise: 1.21, hr: 108, sbp: 98, dbp: 60, map: 73, lactate: 3.6, creatinine: 4.3, ckd_active: true, sepsis_active: true },
      { timestep: 2, label: "T₂", time_delta_hrs: 8.0, trop_value: 0.58, peak_baseline_ratio: 1.38, fold_rise: 1.38, hr: 101, sbp: 104, dbp: 64, map: 77, lactate: 3.1, creatinine: 4.4, ckd_active: true, sepsis_active: true },
    ],
    prediction: { predicted_prob: 0.312, threshold: 0.48, confidence_evolution: [0.38, 0.34, 0.312], dominant_modality: "ECG", attn_temp_entropy: 0.66, attn_spatial_dominance: 0.28, ecg_contribution: 0.62, trop_contribution: 0.38 },
    attention: { leads: makeLowAttention(), temporal: [0.38, 0.33, 0.29], dominant_region: "Mixed" },
    explanation: "Despite high absolute troponin values (0.42–0.58 ng/mL), the model correctly classified this as non-AMI. Active CKD Stage 4 (creatinine 4.1 mg/dL) combined with sepsis-induced demand ischemia explains the elevated troponin without primary myocardial infarction. The ECG showed no focal ischemic changes — only rate-related morphology consistent with sinus tachycardia from sepsis. The model down-weighted the troponin channel and relied on the stable ECG morphology.",
    explanation_temporal: "Notably, model confidence actually decreased over time from 38% to 31.2% as the temporal GRU recognized the plateau-like troponin kinetics (low fold-rise of 1.38) inconsistent with AMI velocity patterns.",
  },
  {
    hadm_id: "HADM-128001",
    subject_id: "P-9103",
    age: 55,
    gender: "M",
    admittime: "2024-01-21 07:18",
    ground_truth_ami: true,
    max_troponin: 2.14,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "Critical",
    comorbidities: ["Diabetes", "Smoking"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.22, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 95, sbp: 132, dbp: 80, map: 97, lactate: 2.2, creatinine: 1.0, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.0, trop_value: 0.91, peak_baseline_ratio: 4.14, fold_rise: 4.14, hr: 102, sbp: 124, dbp: 76, map: 92, lactate: 2.6, creatinine: 1.1, ckd_active: false, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 6.0, trop_value: 2.14, peak_baseline_ratio: 9.73, fold_rise: 9.73, hr: 108, sbp: 118, dbp: 72, map: 87, lactate: 2.9, creatinine: 1.2, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.948, threshold: 0.48, confidence_evolution: [0.38, 0.82, 0.948], dominant_modality: "ECG", attn_temp_entropy: 0.21, attn_spatial_dominance: 0.88, ecg_contribution: 0.71, trop_contribution: 0.29 },
    attention: { leads: makeAnteriorAttention(), temporal: [0.14, 0.50, 0.36], dominant_region: "Anterior/Septal" },
    explanation: "Near-certain anterior STEMI diagnosis. Massive 9.7-fold troponin rise combined with diagnostic anterior ST elevation in V1-V4. No confounders present. The model allocated almost all spatial attention to the anterior/septal lead cluster with exceptional confidence (entropy = 0.21).",
    explanation_temporal: "Confidence surged from 38% at T₀ (hyperacute T-waves, borderline troponin) to 82% at T₁ (established STEMI) to 94.8% at T₂ as troponin velocity confirmed massive myocardial necrosis.",
  },
  {
    hadm_id: "HADM-131244",
    subject_id: "P-4478",
    age: 48,
    gender: "F",
    admittime: "2024-01-22 16:42",
    ground_truth_ami: false,
    max_troponin: 0.02,
    ecg_count: 1,
    timesteps: ["T₀"],
    risk_level: "Low",
    comorbidities: ["Anxiety"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.02, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 92, sbp: 118, dbp: 74, map: 89, lactate: 1.2, creatinine: 0.7, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.028, threshold: 0.48, confidence_evolution: [0.028], dominant_modality: "Troponin", attn_temp_entropy: 0.91, attn_spatial_dominance: 0.09, ecg_contribution: 0.22, trop_contribution: 0.78 },
    attention: { leads: makeLowAttention(), temporal: [1.0], dominant_region: "Mixed" },
    explanation: "Very low AMI risk. Single-timestep admission with normal troponin and normal ECG. No focal ischemic changes. High attention entropy (0.91) confirms diffuse, non-specific activation pattern.",
    explanation_temporal: "Single observation — temporal reasoning not applicable. Risk assessed primarily on baseline troponin and ECG morphology.",
  },
  {
    hadm_id: "HADM-135677",
    subject_id: "P-1192",
    age: 69,
    gender: "M",
    admittime: "2024-01-23 02:11",
    ground_truth_ami: true,
    max_troponin: 0.67,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "High",
    comorbidities: ["Hypertension", "Prior PCI"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.11, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 86, sbp: 140, dbp: 84, map: 103, lactate: 1.7, creatinine: 1.3, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 4.0, trop_value: 0.38, peak_baseline_ratio: 3.45, fold_rise: 3.45, hr: 94, sbp: 132, dbp: 80, map: 97, lactate: 2.0, creatinine: 1.4, ckd_active: false, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 8.5, trop_value: 0.67, peak_baseline_ratio: 6.09, fold_rise: 6.09, hr: 100, sbp: 126, dbp: 76, map: 93, lactate: 2.2, creatinine: 1.5, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.791, threshold: 0.48, confidence_evolution: [0.22, 0.58, 0.791], dominant_modality: "ECG", attn_temp_entropy: 0.35, attn_spatial_dominance: 0.74, ecg_contribution: 0.65, trop_contribution: 0.35 },
    attention: { leads: makeInferiorAttention(), temporal: [0.19, 0.49, 0.32], dominant_region: "Inferior" },
    explanation: "Inferior NSTEMI. Progressive inferior ST depression and T-wave inversion in leads II, III, aVF across the three timesteps. 6-fold troponin rise confirms myocardial necrosis. Prior PCI history noted but model correctly attributes the dynamic ECG changes to a new ischemic event rather than baseline scar morphology.",
    explanation_temporal: "The temporal GRU distinguished the new dynamic changes from the patient's known baseline ECG pattern. Confidence rose steadily from 22% (equivocal changes) to 79.1% (established NSTEMI pattern).",
  },
  {
    hadm_id: "HADM-139022",
    subject_id: "P-8834",
    age: 52,
    gender: "F",
    admittime: "2024-01-24 09:30",
    ground_truth_ami: false,
    max_troponin: 0.09,
    ecg_count: 2,
    timesteps: ["T₀", "T₁"],
    risk_level: "Low",
    comorbidities: ["GERD", "Obesity"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.07, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 84, sbp: 128, dbp: 82, map: 97, lactate: 1.3, creatinine: 0.9, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.0, trop_value: 0.09, peak_baseline_ratio: 1.29, fold_rise: 1.29, hr: 82, sbp: 126, dbp: 80, map: 95, lactate: 1.2, creatinine: 0.9, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.092, threshold: 0.48, confidence_evolution: [0.11, 0.092], dominant_modality: "Troponin", attn_temp_entropy: 0.78, attn_spatial_dominance: 0.18, ecg_contribution: 0.31, trop_contribution: 0.69 },
    attention: { leads: makeLowAttention(), temporal: [0.53, 0.47], dominant_region: "Mixed" },
    explanation: "Low AMI risk. Minimal troponin elevation (within the 95th percentile range for this age/sex cohort) with flat kinetics. ECG showed early repolarization pattern — a benign finding. No dynamic changes across the two timesteps.",
    explanation_temporal: "The model's confidence marginally decreased from T₀ to T₁ as the troponin failed to rise significantly. Early repolarization was correctly categorized as benign by the spatial attention model.",
  },
  {
    hadm_id: "HADM-142387",
    subject_id: "P-7731",
    age: 80,
    gender: "M",
    admittime: "2024-01-25 18:05",
    ground_truth_ami: true,
    max_troponin: 3.42,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "Critical",
    comorbidities: ["Heart Failure", "AF", "CKD Stage 2"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.88, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 118, sbp: 96, dbp: 58, map: 71, lactate: 4.1, creatinine: 2.1, ckd_active: true, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 2.0, trop_value: 1.84, peak_baseline_ratio: 2.09, fold_rise: 2.09, hr: 124, sbp: 88, dbp: 54, map: 65, lactate: 4.8, creatinine: 2.3, ckd_active: true, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 4.5, trop_value: 3.42, peak_baseline_ratio: 3.89, fold_rise: 3.89, hr: 130, sbp: 82, dbp: 50, map: 61, lactate: 5.2, creatinine: 2.6, ckd_active: true, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.912, threshold: 0.48, confidence_evolution: [0.61, 0.84, 0.912], dominant_modality: "Balanced", attn_temp_entropy: 0.24, attn_spatial_dominance: 0.81, ecg_contribution: 0.52, trop_contribution: 0.48 },
    attention: { leads: makeAnteriorAttention(), temporal: [0.28, 0.42, 0.30], dominant_region: "Anterior/Septal" },
    explanation: "High-confidence AMI despite CKD Stage 2 confounding. The key discriminator was the massive fold-rise of 3.89× — a pattern atypical for pure CKD-related chronic elevation. The anterior ECG morphology showed progressive Q-wave development and ST changes consistent with extensive AMI. The model weighted both modalities nearly equally.",
    explanation_temporal: "Even at T₀, the model predicted 61% AMI given the combination of cardiogenic shock hemodynamics and ECG morphology. The rapid troponin doubling time distinguished this from CKD's characteristic slow, plateau-like elevation.",
  },
  {
    hadm_id: "HADM-147890",
    subject_id: "P-3309",
    age: 61,
    gender: "F",
    admittime: "2024-01-26 13:44",
    ground_truth_ami: false,
    max_troponin: 0.15,
    ecg_count: 2,
    timesteps: ["T₀", "T₁"],
    risk_level: "Low",
    comorbidities: ["Hypothyroidism", "Hypertension"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.12, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 66, sbp: 136, dbp: 84, map: 101, lactate: 1.5, creatinine: 1.1, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.5, trop_value: 0.15, peak_baseline_ratio: 1.25, fold_rise: 1.25, hr: 64, sbp: 134, dbp: 82, map: 99, lactate: 1.4, creatinine: 1.1, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.134, threshold: 0.48, confidence_evolution: [0.16, 0.134], dominant_modality: "Troponin", attn_temp_entropy: 0.75, attn_spatial_dominance: 0.21, ecg_contribution: 0.34, trop_contribution: 0.66 },
    attention: { leads: makeLowAttention(), temporal: [0.54, 0.46], dominant_region: "Mixed" },
    explanation: "Mildly elevated troponin attributed to demand ischemia from hypothyroid cardiomyopathy. ECG showed non-specific T-wave flattening without focal ischemic changes. The minimal fold-rise (1.25×) and diffuse, low-magnitude attention pattern support a non-AMI classification.",
    explanation_temporal: "Confidence decreased marginally over time, consistent with an improving demand ischemia pattern rather than the rising confidence trajectory seen in true AMI.",
  },
  {
    hadm_id: "HADM-153201",
    subject_id: "P-6621",
    age: 57,
    gender: "M",
    admittime: "2024-01-27 05:22",
    ground_truth_ami: true,
    max_troponin: 0.72,
    ecg_count: 2,
    timesteps: ["T₀", "T₁"],
    risk_level: "High",
    comorbidities: ["Smoking", "Hyperlipidemia"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.14, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 93, sbp: 144, dbp: 88, map: 107, lactate: 1.9, creatinine: 1.0, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 4.0, trop_value: 0.72, peak_baseline_ratio: 5.14, fold_rise: 5.14, hr: 101, sbp: 136, dbp: 84, map: 101, lactate: 2.2, creatinine: 1.1, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.843, threshold: 0.48, confidence_evolution: [0.31, 0.843], dominant_modality: "Balanced", attn_temp_entropy: 0.29, attn_spatial_dominance: 0.72, ecg_contribution: 0.54, trop_contribution: 0.46 },
    attention: { leads: makeLateralAttention(), temporal: [0.38, 0.62], dominant_region: "Lateral" },
    explanation: "Two-timestep lateral NSTEMI. The massive 5.14-fold troponin jump from T₀ to T₁ was the critical diagnostic signal, combined with newly developed lateral ST depression in I, aVL, V5-V6. The temporal GRU accurately captured the inflection point and escalated the confidence sharply.",
    explanation_temporal: "A significant confidence jump occurred between T₀ (31%) and T₁ (84.3%), driven by the rapid troponin rise and confirmatory ECG changes. The model placed 62% of temporal weight on T₁, reflecting the decisive diagnostic value of the second timepoint.",
  },
  {
    hadm_id: "HADM-158432",
    subject_id: "P-4411",
    age: 77,
    gender: "F",
    admittime: "2024-01-28 21:15",
    ground_truth_ami: false,
    max_troponin: 0.44,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "Moderate",
    comorbidities: ["Sepsis", "Lung Cancer"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.30, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 112, sbp: 102, dbp: 64, map: 77, lactate: 3.8, creatinine: 1.8, ckd_active: false, sepsis_active: true },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.5, trop_value: 0.38, peak_baseline_ratio: 1.27, fold_rise: 1.27, hr: 108, sbp: 106, dbp: 66, map: 79, lactate: 3.4, creatinine: 1.9, ckd_active: false, sepsis_active: true },
      { timestep: 2, label: "T₂", time_delta_hrs: 7.0, trop_value: 0.44, peak_baseline_ratio: 1.47, fold_rise: 1.47, hr: 100, sbp: 110, dbp: 68, map: 82, lactate: 3.0, creatinine: 2.0, ckd_active: false, sepsis_active: true },
    ],
    prediction: { predicted_prob: 0.274, threshold: 0.48, confidence_evolution: [0.35, 0.30, 0.274], dominant_modality: "ECG", attn_temp_entropy: 0.63, attn_spatial_dominance: 0.31, ecg_contribution: 0.59, trop_contribution: 0.41 },
    attention: { leads: makeLowAttention(), temporal: [0.38, 0.34, 0.28], dominant_region: "Mixed" },
    explanation: "Sepsis-induced demand ischemia. Despite significant troponin elevation (0.30–0.44 ng/mL), the plateau kinetics and active sepsis flag (HR 112, lactate 3.8) are classic non-ischemic patterns. The ECG showed diffuse ST changes consistent with myocardial strain from sepsis rather than focal plaque rupture. Model confidence decreased over time as the troponin stabilized.",
    explanation_temporal: "Descending confidence trajectory (35% → 27.4%) is characteristic of sepsis-mediated type 2 MI that the model correctly identifies as non-primary. The temporal GRU recognizes the stabilizing troponin kinetics as inconsistent with plaque rupture.",
  },
  {
    hadm_id: "HADM-162877",
    subject_id: "P-9918",
    age: 64,
    gender: "M",
    admittime: "2024-01-29 11:08",
    ground_truth_ami: true,
    max_troponin: 1.58,
    ecg_count: 3,
    timesteps: ["T₀", "T₁", "T₂"],
    risk_level: "Critical",
    comorbidities: ["Diabetes", "Hypertension", "Prior MI"],
    timelines: [
      { timestep: 0, label: "T₀", time_delta_hrs: 0, trop_value: 0.19, peak_baseline_ratio: 1.0, fold_rise: 1.0, hr: 100, sbp: 128, dbp: 80, map: 96, lactate: 2.4, creatinine: 1.4, ckd_active: false, sepsis_active: false },
      { timestep: 1, label: "T₁", time_delta_hrs: 3.0, trop_value: 0.74, peak_baseline_ratio: 3.89, fold_rise: 3.89, hr: 108, sbp: 120, dbp: 74, map: 89, lactate: 2.8, creatinine: 1.5, ckd_active: false, sepsis_active: false },
      { timestep: 2, label: "T₂", time_delta_hrs: 6.0, trop_value: 1.58, peak_baseline_ratio: 8.32, fold_rise: 8.32, hr: 115, sbp: 112, dbp: 70, map: 84, lactate: 3.1, creatinine: 1.6, ckd_active: false, sepsis_active: false },
    ],
    prediction: { predicted_prob: 0.936, threshold: 0.48, confidence_evolution: [0.41, 0.78, 0.936], dominant_modality: "ECG", attn_temp_entropy: 0.22, attn_spatial_dominance: 0.86, ecg_contribution: 0.69, trop_contribution: 0.31 },
    attention: { leads: makeInferiorAttention(), temporal: [0.17, 0.48, 0.35], dominant_region: "Inferior" },
    explanation: "High-confidence inferior STEMI in a patient with prior MI and diabetes. The model correctly differentiated new ischemic changes from the patient's known prior infarct scar. The 8.32-fold troponin rise and progressive inferior ST evolution across three timesteps provided overwhelming diagnostic evidence.",
    explanation_temporal: "Confidence escalated from 41% (subtle changes against scar background) to 93.6% (unambiguous new inferior STEMI). The model's prior MI awareness was encoded through the temporal GRU's learned embedding representations.",
  },
];

export const COHORT_STATS = {
  total_admissions: 40255,
  ami_prevalence: 0.3127,
  model_threshold: 0.48,
  model_f1: 0.7753,
  model_auc: 0.8912,
  temporal_distribution: { one_timestep: 0.18, two_timestep: 0.41, three_timestep: 0.41 },
};

export function getRiskColor(risk: RiskLevel): string {
  switch (risk) {
    case "Critical": return "text-red-400 bg-red-950 border-red-800";
    case "High": return "text-orange-400 bg-orange-950 border-orange-800";
    case "Moderate": return "text-yellow-400 bg-yellow-950 border-yellow-800";
    case "Low": return "text-emerald-400 bg-emerald-950 border-emerald-800";
  }
}

export function getPredictionColor(prob: number): string {
  if (prob >= 0.75) return "text-red-400";
  if (prob >= 0.48) return "text-orange-400";
  if (prob >= 0.25) return "text-yellow-400";
  return "text-emerald-400";
}

export function getLeadRegion(lead: string): "Inferior" | "Anterior/Septal" | "Lateral" {
  if (["II", "III", "aVF"].includes(lead)) return "Inferior";
  if (["V1", "V2", "V3", "V4"].includes(lead)) return "Anterior/Septal";
  return "Lateral";
}
