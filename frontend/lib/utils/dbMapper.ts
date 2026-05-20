import type { Admission, RiskLevel } from "./mockData";

// Maps a raw PostgreSQL row (snake_case, JSONB already parsed) → Admission shape
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function rowToAdmission(row: any): Admission {
  const hadmId = `HADM-${row.hadm_id}`;

  // JSONB columns arrive as parsed objects from pg
  const timelines = Array.isArray(row.timelines) ? row.timelines : [];
  const prediction = row.prediction && typeof row.prediction === "object" ? row.prediction : {};
  const attention = row.attention && typeof row.attention === "object" ? row.attention : {};
  const comorbidities = Array.isArray(row.comorbidities) ? row.comorbidities : [];

  const ecgSeqLen = parseInt(row.ecg_seq_len ?? "1", 10) || 1;

  return {
    hadm_id: hadmId,
    subject_id: row.subject_id || `P${String(row.hadm_id).slice(-5)}`,
    age: parseFloat(row.age ?? row.anchor_age ?? "0"),
    gender: row.gender === "M" ? "M" : "F",
    admittime: row.admittime
      ? new Date(row.admittime).toISOString()
      : new Date().toISOString(),
    ground_truth_ami: Boolean(row.ground_truth_ami),
    max_troponin: parseFloat(row.max_troponin ?? "0"),
    ecg_count: ecgSeqLen,
    timesteps: timelines.map((_: unknown, i: number) => `T${i}`),
    risk_level: (row.risk_level as RiskLevel) ?? "Low",
    comorbidities,
    timelines,
    prediction: {
      predicted_prob: parseFloat(prediction.predicted_prob ?? "0"),
      threshold: parseFloat(prediction.threshold ?? "0.6494"),
      confidence_evolution: Array.isArray(prediction.confidence_evolution)
        ? prediction.confidence_evolution
        : [parseFloat(prediction.predicted_prob ?? "0")],
      dominant_modality: prediction.dominant_modality ?? "Balanced",
      attn_temp_entropy: parseFloat(prediction.attn_temp_entropy ?? "0.5"),
      attn_spatial_dominance: parseFloat(prediction.attn_spatial_dominance ?? "0.5"),
      ecg_contribution: parseFloat(prediction.ecg_contribution ?? "0.5"),
      trop_contribution: parseFloat(prediction.trop_contribution ?? "0.5"),
    },
    attention: {
      leads: attention.leads ?? {},
      temporal: Array.isArray(attention.temporal) ? attention.temporal : [1],
      dominant_region: attention.dominant_region ?? "Mixed",
    },
    explanation: row.explanation ?? "",
    explanation_temporal: row.explanation_temporal ?? "",
  };
}
