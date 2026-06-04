import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db";
import { rowToAdmission } from "@/lib/utils/dbMapper";
import { MOCK_ADMISSIONS } from "@/lib/utils/mockData";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  // Strip the "HADM-" prefix if the frontend passes it
  const hadmId = id.replace(/^HADM-/i, "");

  try {
    const rows = await query(`SELECT * FROM admissions WHERE hadm_id = $1`, [hadmId]);
    if (!rows.length) {
      return NextResponse.json({ error: "Admission not found" }, { status: 404 });
    }
    return NextResponse.json({ admission: rowToAdmission(rows[0]) });
  } catch (err) {
    console.error("[GET /api/admissions/[id]]", err);
    const admission = MOCK_ADMISSIONS.find((item) => item.hadm_id.replace(/^HADM-/i, "") === hadmId);
    if (!admission) {
      return NextResponse.json({ error: "Admission not found" }, { status: 404 });
    }

    return NextResponse.json({ admission, source: "mock" });
  }
}
