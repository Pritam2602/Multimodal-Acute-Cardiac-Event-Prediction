import Link from "next/link";
import { Upload } from "lucide-react";
import Sidebar from "@/components/ui/Sidebar";
import MetricsBar from "@/components/ui/MetricsBar";
import AdmissionsTable from "@/components/AdmissionsTable";

export default function DashboardPage() {
  return (
    <div className="flex h-screen overflow-hidden bg-base">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <MetricsBar />
        <div className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col">
            {/* Section header */}
            <div className="px-6 py-4 border-b border-border-default flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-text-primary">Patient Admissions</h2>
                <p className="text-[11px] text-text-muted mt-0.5">Click any row to open the full clinical workstation</p>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <span className="w-2 h-2 rounded-full bg-danger inline-block" />
                    Critical / High
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <span className="w-2 h-2 rounded-full bg-amber inline-block" />
                    Moderate
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <span className="w-2 h-2 rounded-full bg-safe inline-block" />
                    Low Risk
                  </div>
                </div>
                <Link
                  href="/upload"
                  className="flex items-center gap-1.5 px-3 py-2 text-[11px] font-semibold bg-cyan/10 border border-cyan/40 text-cyan rounded-lg hover:bg-cyan/15 transition-all"
                >
                  <Upload className="w-3.5 h-3.5" />
                  Upload Patient
                </Link>
              </div>
            </div>
            <div className="flex-1 overflow-hidden">
              <AdmissionsTable />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
