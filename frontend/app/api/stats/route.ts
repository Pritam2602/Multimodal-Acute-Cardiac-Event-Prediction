import { NextResponse } from "next/server";
import { query } from "@/lib/db";

// GET /api/stats — Live cohort statistics from PostgreSQL
export async function GET() {
  try {
    const [countRow] = await query<{ total: string }>(
      "SELECT COUNT(*) AS total FROM admissions"
    );
    const [amiRow] = await query<{ ami: string }>(
      "SELECT COUNT(*) AS ami FROM admissions WHERE ground_truth_ami = 1"
    );
    const total = parseInt(countRow.total, 10);
    const ami = parseInt(amiRow.ami, 10);

    return NextResponse.json({
      total_admissions: total,
      ami_prevalence: total > 0 ? ami / total : 0,
      // Real model metrics from Phase 10 Early Fusion (best model)
      model_f1: 0.7753,
      model_auc: 0.9411,
      model_threshold: 0.6494,
      best_model: "Phase 10 Early Fusion",
      best_epoch: 6,
      source: "postgres",
    });
  } catch {
    // Fallback: hardcoded stats if DB is down
    return NextResponse.json({
      total_admissions: 40255,
      ami_prevalence: 0.3127,
      model_f1: 0.7753,
      model_auc: 0.9411,
      model_threshold: 0.6494,
      best_model: "Phase 10 Early Fusion",
      best_epoch: 6,
      source: "fallback",
    });
  }
}
