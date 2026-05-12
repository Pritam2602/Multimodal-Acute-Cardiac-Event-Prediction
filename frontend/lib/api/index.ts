// API wrappers — in production these call FastAPI backend
// Currently returns mock data synchronously

import { MOCK_ADMISSIONS, COHORT_STATS, Admission } from "@/lib/utils/mockData";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchAdmissions(page = 1, limit = 20): Promise<{ data: Admission[]; total: number }> {
  if (process.env.NODE_ENV === "development") {
    return { data: MOCK_ADMISSIONS, total: COHORT_STATS.total_admissions };
  }
  const res = await fetch(`${BASE_URL}/api/admissions?page=${page}&limit=${limit}`);
  return res.json();
}

export async function fetchAdmission(hadmId: string): Promise<Admission | undefined> {
  if (process.env.NODE_ENV === "development") {
    return MOCK_ADMISSIONS.find((a) => a.hadm_id === hadmId);
  }
  const res = await fetch(`${BASE_URL}/api/admission/${hadmId}`);
  return res.json();
}

export async function fetchPrediction(hadmId: string) {
  if (process.env.NODE_ENV === "development") {
    return MOCK_ADMISSIONS.find((a) => a.hadm_id === hadmId)?.prediction;
  }
  const res = await fetch(`${BASE_URL}/api/prediction/${hadmId}`);
  return res.json();
}

export async function fetchExplanation(hadmId: string) {
  if (process.env.NODE_ENV === "development") {
    const a = MOCK_ADMISSIONS.find((a) => a.hadm_id === hadmId);
    return { explanation: a?.explanation, explanation_temporal: a?.explanation_temporal, attention: a?.attention };
  }
  const res = await fetch(`${BASE_URL}/api/explain/${hadmId}`);
  return res.json();
}
