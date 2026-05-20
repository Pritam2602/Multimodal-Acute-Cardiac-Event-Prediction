import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { rowToAdmission } from "@/lib/utils/dbMapper";

// GET /api/admissions?search=&risk=&prediction=&limit=100&offset=0
export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const search     = searchParams.get("search") || "";
  const risk       = searchParams.get("risk") || "";
  const prediction = searchParams.get("prediction") || "";
  const limit      = Math.min(parseInt(searchParams.get("limit") || "200", 10), 10000);
  const offset     = parseInt(searchParams.get("offset") || "0", 10);

  try {
    const conditions: string[] = [];
    const params: unknown[] = [];
    let p = 1;

    if (search) {
      conditions.push(`(subject_id ILIKE $${p} OR hadm_id::text ILIKE $${p})`);
      params.push(`%${search}%`);
      p++;
    }
    if (risk) {
      conditions.push(`risk_level = $${p}`);
      params.push(risk);
      p++;
    }
    if (prediction === "AMI") {
      conditions.push(`(prediction->>'predicted_prob')::float >= (prediction->>'threshold')::float`);
    } else if (prediction === "Non-AMI") {
      conditions.push(`(prediction->>'predicted_prob')::float < (prediction->>'threshold')::float`);
    }

    const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

    const rows = await query(
      `SELECT * FROM admissions ${where} ORDER BY created_at DESC LIMIT $${p} OFFSET $${p + 1}`,
      [...params, limit, offset]
    );

    const countRows = await query<{ total: string }>(
      `SELECT COUNT(*) AS total FROM admissions ${where}`,
      params
    );

    return NextResponse.json({
      admissions: rows.map(rowToAdmission),
      total: parseInt(countRows[0]?.total ?? "0", 10),
    });
  } catch (err) {
    console.error("[GET /api/admissions]", err);
    return NextResponse.json({ error: "Database error", detail: String(err) }, { status: 500 });
  }
}

// POST /api/admissions — doctor uploads a new patient
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const {
      hadm_id, subject_id, anchor_age, gender,
      heart_rate, respiratory_rate, troponin_t, creatinine, sodium, potassium,
      num_diagnoses, los, pr_interval, qrs_duration, qt_interval, qtc,
      p_axis, qrs_axis, t_axis, rr_interval,
      has_ckd = false, has_hf = false, has_sepsis = false,
      has_pe = false, has_afib = false, has_diabetes = false,
      ground_truth_ami = false,
      notes = "",
    } = body;

    // Generate a unique hadm_id if not provided (live uploads)
    const newHadmId = hadm_id ?? Date.now();

    // Build heuristic prediction from inputs
    const prob = computeHeuristicProb({
      troponin_t, pr_interval, qrs_duration, qtc, anchor_age,
      has_ckd, has_sepsis,
    });
    const rl = riskLevel(prob);
    const region = dominantRegion(qrs_axis ?? 0, t_axis ?? 0);
    const attnLeads = buildAttentionLeads(region);

    const timeline = [{
      timestep: 0, label: "T0", time_delta_hrs: 0,
      trop_value: troponin_t ?? 0,
      peak_baseline_ratio: 1.0, fold_rise: 1.0,
      hr: heart_rate ?? 75,
      sbp: 120, dbp: 80, map: 93, lactate: 1.2,
      creatinine: creatinine ?? 1.0,
      ckd_active: Boolean(has_ckd),
      sepsis_active: Boolean(has_sepsis),
    }];

    const comorbidities: string[] = [
      has_ckd && "CKD", has_hf && "Heart Failure", has_sepsis && "Sepsis",
      has_pe && "Pulmonary Embolism", has_afib && "Atrial Fibrillation",
      has_diabetes && "Diabetes",
    ].filter(Boolean) as string[];

    const prediction = {
      predicted_prob: Math.round(prob * 10000) / 10000,
      threshold: 0.6494,
      confidence_evolution: [Math.round(prob * 10000) / 10000],
      dominant_modality: "Balanced",
      attn_temp_entropy: 0.5,
      attn_spatial_dominance: Math.max(...Object.values(attnLeads)),
      ecg_contribution: 0.5,
      trop_contribution: 0.5,
    };

    const attention = { leads: attnLeads, temporal: [1.0], dominant_region: region };

    const explanation = `Patient age ${anchor_age ?? "unknown"}, troponin ${troponin_t ?? 0} ng/mL. `
      + `AI confidence: ${(prob * 100).toFixed(1)}% AMI probability. Risk: ${rl}.`;

    await query(
      `INSERT INTO admissions (
        hadm_id, subject_id, anchor_age, gender, admittime,
        heart_rate, respiratory_rate, troponin_t, creatinine, sodium, potassium,
        num_diagnoses, los, pr_interval, qrs_duration, qt_interval, qtc,
        p_axis, qrs_axis, t_axis, rr_interval,
        troponin_peak, max_troponin,
        has_ckd, has_hf, has_sepsis, has_pe, has_afib, has_diabetes,
        ground_truth_ami, risk_level,
        timelines, prediction, attention, comorbidities,
        explanation, explanation_temporal, split
      ) VALUES (
        $1,$2,$3,$4,NOW(),
        $5,$6,$7,$8,$9,$10,
        $11,$12,$13,$14,$15,$16,
        $17,$18,$19,$20,
        $21,$21,
        $22,$23,$24,$25,$26,$27,
        $28,$29,
        $30::jsonb,$31::jsonb,$32::jsonb,$33::jsonb,
        $34,$35,'live'
      )`,
      [
        newHadmId, subject_id ?? `P${String(newHadmId).slice(-5)}`,
        anchor_age ?? 0, gender ?? "M",
        heart_rate ?? 0, respiratory_rate ?? 0, troponin_t ?? 0, creatinine ?? 0, sodium ?? 0, potassium ?? 0,
        num_diagnoses ?? 0, los ?? 0, pr_interval ?? 0, qrs_duration ?? 0, qt_interval ?? 0, qtc ?? 0,
        p_axis ?? 0, qrs_axis ?? 0, t_axis ?? 0, rr_interval ?? 0,
        troponin_t ?? 0,
        Boolean(has_ckd), Boolean(has_hf), Boolean(has_sepsis), Boolean(has_pe), Boolean(has_afib), Boolean(has_diabetes),
        Boolean(ground_truth_ami), rl,
        JSON.stringify(timeline), JSON.stringify(prediction), JSON.stringify(attention), JSON.stringify(comorbidities),
        explanation, explanation,
      ]
    );

    // Log the upload
    await query(
      `INSERT INTO upload_log (hadm_id, uploaded_by, notes) VALUES ($1, 'doctor', $2)`,
      [newHadmId, notes]
    );

    const [newRow] = await query(`SELECT * FROM admissions WHERE hadm_id = $1`, [newHadmId]);
    return NextResponse.json({ admission: rowToAdmission(newRow) }, { status: 201 });
  } catch (err) {
    console.error("[POST /api/admissions]", err);
    return NextResponse.json({ error: "Database error", detail: String(err) }, { status: 500 });
  }
}

// ── Helpers (mirror split_dataset.py logic) ───────────────────────────────────
function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function computeHeuristicProb(r: Record<string, unknown>) {
  let z = -0.6;
  const trop = Number(r.troponin_t ?? 0);
  z += Math.log1p(trop) * 0.9;
  if (Number(r.qtc) > 450) z += 0.18;
  if (Number(r.qrs_duration) > 120) z += 0.28;
  const age = Number(r.anchor_age ?? 55);
  z += (age - 55) / 60 * 0.25;
  if (r.has_ckd) z -= 0.12;
  if (r.has_sepsis) z -= 0.10;
  return sigmoid(z);
}

function riskLevel(prob: number): string {
  if (prob >= 0.75) return "Critical";
  if (prob >= 0.6494) return "High";
  if (prob >= 0.25) return "Moderate";
  return "Low";
}

function dominantRegion(qrsAxis: number, tAxis: number): string {
  if (qrsAxis >= -30 && qrsAxis <= 90 && tAxis < 0) return "Inferior";
  if (qrsAxis < -30 || qrsAxis > 90) return "Anterior/Septal";
  if (tAxis > 90) return "Lateral";
  return "Mixed";
}

function buildAttentionLeads(region: string): Record<string, number> {
  const LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];
  const base: Record<string, number> = {};
  LEADS.forEach(l => { base[l] = parseFloat((0.15 + Math.random() * 0.20).toFixed(3)); });
  const hot = region === "Inferior" ? ["II","III","aVF"]
    : region === "Anterior/Septal" ? ["V1","V2","V3","V4"]
    : region === "Lateral" ? ["I","aVL","V5","V6"] : [];
  hot.forEach(l => { base[l] = parseFloat((0.75 + Math.random() * 0.20).toFixed(3)); });
  return base;
}
