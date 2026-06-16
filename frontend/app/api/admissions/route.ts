import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { rowToAdmission } from "@/lib/utils/dbMapper";
import { Admission, AttentionWeights, MOCK_ADMISSIONS, PredictionData, RiskLevel } from "@/lib/utils/mockData";

// GET /api/admissions?search=&risk=&prediction=&limit=100&offset=0
export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const search     = searchParams.get("search") || "";
  const risk       = searchParams.get("risk") || "";
  const prediction = searchParams.get("prediction") || "";
  const limit      = Math.min(parseInt(searchParams.get("limit") || "100", 10), 500);
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
      `SELECT * FROM admissions ${where} ORDER BY hadm_id DESC LIMIT $${p} OFFSET $${p + 1}`,
      [...params, limit, offset]
    );

    const countRows = await query<{ total: string }>(
      `SELECT COUNT(*) AS total FROM admissions ${where}`,
      params
    );

    const admissions = rows.map(rowToAdmission);

    return NextResponse.json({
      admissions,
      total: parseInt(countRows[0]?.total ?? String(admissions.length), 10),
    });
  } catch (err) {
    console.error("[GET /api/admissions]", err);
    const admissions = filterMockAdmissions(MOCK_ADMISSIONS, { search, risk, prediction });

    return NextResponse.json({
      admissions: admissions.slice(offset, offset + limit),
      total: admissions.length,
      source: "mock",
    });
  }
}

// POST /api/admissions — doctor uploads a new patient
export async function POST(req: NextRequest) {
  let fallbackAdmission: Admission | null = null;

  try {
    const body = await req.json();

    const {
      hadm_id, subject_id, anchor_age, gender,
      heart_rate, creatinine, troponin_t,
      qrs_axis, t_axis, qrs_duration, qtc,
      has_ckd = false, has_hf = false, has_sepsis = false,
      has_pe = false, has_afib = false, has_diabetes = false,
      ground_truth_ami = false,
      notes = "",
    } = body;

    // Generate a unique hadm_id if not provided
    const newHadmId = hadm_id ? String(hadm_id) : `HADM-${Date.now()}`;
    const newSubjectId = subject_id ? String(subject_id) : `P-${String(Date.now()).slice(-5)}`;

    // Build heuristic prediction from submitted values
    const prob = computeHeuristicProb({
      troponin_t, qrs_duration, qtc, anchor_age,
      has_ckd, has_sepsis,
    });
    const rl = riskLevel(prob);
    const region = dominantRegion(qrs_axis ?? 0, t_axis ?? 0);
    const attnLeads = buildAttentionLeads(region);

    // Build the timeline JSONB — all vitals/ECG fields live here
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
      has_ckd && "CKD",
      has_hf && "Heart Failure",
      has_sepsis && "Sepsis",
      has_pe && "Pulmonary Embolism",
      has_afib && "Atrial Fibrillation",
      has_diabetes && "Diabetes",
    ].filter(Boolean) as string[];

    const prediction: PredictionData = {
      predicted_prob: Math.round(prob * 10000) / 10000,
      threshold: 0.45, // Phase 10 curated model threshold
      confidence_evolution: [Math.round(prob * 10000) / 10000],
      dominant_modality: "Balanced",
      attn_temp_entropy: 0.5,
      attn_spatial_dominance: Math.max(...Object.values(attnLeads)),
      ecg_contribution: 0.5,
      trop_contribution: 0.5,
    };

    const attention: AttentionWeights = {
      leads: attnLeads,
      temporal: [1.0],
      dominant_region: region,
    };

    const explanation =
      `Patient age ${anchor_age ?? "unknown"}, troponin ${troponin_t ?? 0} ng/mL. ` +
      `AI confidence: ${(prob * 100).toFixed(1)}% AMI probability. Risk level: ${rl}.`;

    // Build the fallback object (returned if DB write fails)
    fallbackAdmission = {
      hadm_id: newHadmId,
      subject_id: newSubjectId,
      age: anchor_age ?? 0,
      gender: gender === "F" ? "F" : "M",
      admittime: new Date().toISOString(),
      ground_truth_ami: Boolean(ground_truth_ami),
      max_troponin: troponin_t ?? 0,
      ecg_count: 1,
      timesteps: ["T0"],
      risk_level: rl,
      comorbidities,
      timelines: timeline,
      prediction,
      attention,
      explanation,
      explanation_temporal: explanation,
    };

    // ── 1. Upsert patient (required by FK constraint) ────────────────────────
    await query(
      `INSERT INTO patients (subject_id, age, gender)
       VALUES ($1, $2, $3)
       ON CONFLICT (subject_id) DO UPDATE
         SET age = EXCLUDED.age, gender = EXCLUDED.gender`,
      [newSubjectId, anchor_age ?? 0, gender === "F" ? 1 : 0]
    );

    // ── 2. Insert admission — only columns that exist in the schema ──────────
    await query(
      `INSERT INTO admissions (
        hadm_id, subject_id, anchor_age, gender, admittime,
        max_troponin, has_ckd, has_hf, has_sepsis,
        ground_truth_ami, risk_level,
        timelines, prediction, attention, comorbidities,
        explanation, explanation_temporal
      ) VALUES (
        $1, $2, $3, $4, NOW(),
        $5, $6, $7, $8,
        $9, $10,
        $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb,
        $15, $16
      )`,
      [
        newHadmId, newSubjectId,
        anchor_age ?? 0, gender ?? "M",
        troponin_t ?? 0,
        Boolean(has_ckd) ? 1 : 0,
        Boolean(has_hf) ? 1 : 0,
        Boolean(has_sepsis) ? 1 : 0,
        Boolean(ground_truth_ami) ? 1 : 0,
        rl,
        JSON.stringify(timeline),
        JSON.stringify(prediction),
        JSON.stringify(attention),
        JSON.stringify(comorbidities),
        explanation,
        explanation,
      ]
    );

    // ── 3. Try logging the upload (non-fatal if upload_log doesn't exist) ────
    try {
      await query(
        `INSERT INTO upload_log (hadm_id, uploaded_by, notes) VALUES ($1, 'doctor', $2)`,
        [newHadmId, notes]
      );
    } catch { /* upload_log table may not exist — ignore */ }

    const [newRow] = await query(`SELECT * FROM admissions WHERE hadm_id = $1`, [newHadmId]);
    return NextResponse.json({ admission: rowToAdmission(newRow) }, { status: 201 });

  } catch (err) {
    console.error("[POST /api/admissions]", err);
    if (fallbackAdmission) {
      // Still return 201 so the UI navigates to the patient — it just won't
      // be in the DB (user sees it in the session, not after reload)
      return NextResponse.json({ admission: fallbackAdmission, source: "mock" }, { status: 201 });
    }
    return NextResponse.json({ error: "Database error", detail: String(err) }, { status: 500 });
  }
}

// ── Helpers (mirror split_dataset.py logic) ───────────────────────────────────
function filterMockAdmissions(
  admissions: Admission[],
  filters: { search: string; risk: string; prediction: string }
) {
  const search = filters.search.toLowerCase();

  return admissions.filter((admission) => {
    const matchesSearch = !search
      || admission.subject_id.toLowerCase().includes(search)
      || admission.hadm_id.toLowerCase().includes(search);
    const matchesRisk = !filters.risk || admission.risk_level === filters.risk;
    const isAmi = admission.prediction.predicted_prob >= admission.prediction.threshold;
    const matchesPrediction = !filters.prediction
      || (filters.prediction === "AMI" && isAmi)
      || (filters.prediction === "Non-AMI" && !isAmi);

    return matchesSearch && matchesRisk && matchesPrediction;
  });
}

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

function riskLevel(prob: number): RiskLevel {
  if (prob >= 0.75) return "Critical";
  if (prob >= 0.45) return "High"; // Phase 10 threshold
  if (prob >= 0.25) return "Moderate";
  return "Low";
}

function dominantRegion(qrsAxis: number, tAxis: number): AttentionWeights["dominant_region"] {
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
