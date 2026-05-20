"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, LayoutDashboard, GitCompare, Upload, Cpu, AlertTriangle, LogOut } from "lucide-react";
import { COHORT_STATS } from "@/lib/utils/mockData";
import { useAdmissionStore } from "@/lib/store/useAdmissionStore";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Admissions", icon: LayoutDashboard, desc: "Patient cohort table" },
  { href: "/compare", label: "Case Compare", icon: GitCompare, desc: "Side-by-side analysis" },
  { href: "/upload", label: "Upload Patient", icon: Upload, desc: "New admission entry" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { admissions } = useAdmissionStore();

  const topCritical = [...admissions]
    .sort((a, b) => b.prediction.predicted_prob - a.prediction.predicted_prob)
    .slice(0, 3);

  return (
    <aside
      className="w-64 min-h-screen flex flex-col"
      style={{ background: "var(--sidebar-bg)", boxShadow: "2px 0 12px rgba(0,0,0,0.12)" }}
    >
      {/* Logo */}
      <div className="px-5 py-5" style={{ borderBottom: "1px solid var(--sidebar-border)" }}>
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: "rgba(255,255,255,0.12)" }}
          >
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-tight">AMI Predict</p>
            <p className="text-[10px] leading-tight" style={{ color: "var(--sidebar-muted)" }}>
              Temporal Multimodal AI
            </p>
          </div>
        </div>
      </div>

      {/* Model Status Card */}
      <div
        className="mx-4 mt-4 p-3 rounded-xl"
        style={{ background: "rgba(255,255,255,0.07)", border: "1px solid var(--sidebar-border)" }}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" style={{ color: "var(--sidebar-text)" }} />
            <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--sidebar-muted)" }}>
              Model Status
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse-slow" />
            <span className="text-[10px] text-green-400">Online</span>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <span className="text-[11px]" style={{ color: "var(--sidebar-muted)" }}>F1 Score</span>
            <span className="text-[11px] font-mono text-white">{COHORT_STATS.model_f1.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[11px]" style={{ color: "var(--sidebar-muted)" }}>AUC-ROC</span>
            <span className="text-[11px] font-mono text-white">{COHORT_STATS.model_auc.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[11px]" style={{ color: "var(--sidebar-muted)" }}>Threshold</span>
            <span className="text-[11px] font-mono text-amber">{COHORT_STATS.model_threshold}</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 mt-5">
        <p
          className="text-[9px] font-bold uppercase tracking-widest px-2 mb-3"
          style={{ color: "var(--sidebar-muted)" }}
        >
          Navigation
        </p>
        <div className="space-y-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon, desc }) => {
            const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all group"
                style={
                  active
                    ? { background: "#ffffff" }
                    : { background: "transparent" }
                }
              >
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 transition-all"
                  style={
                    active
                      ? { background: "rgba(22,56,41,0.08)" }
                      : { background: "rgba(255,255,255,0.08)" }
                  }
                >
                  <Icon
                    className="w-3.5 h-3.5"
                    style={active ? { color: "var(--sidebar-bg)" } : { color: "var(--sidebar-text)" }}
                  />
                </div>
                <div>
                  <p
                    className="text-xs font-semibold leading-tight"
                    style={active ? { color: "var(--sidebar-bg)" } : { color: "var(--sidebar-text)" }}
                  >
                    {label}
                  </p>
                  <p
                    className="text-[10px] leading-tight"
                    style={active ? { color: "#4a8a62" } : { color: "var(--sidebar-muted)" }}
                  >
                    {desc}
                  </p>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Alert Section */}
        {topCritical.length > 0 && (
          <div className="mt-6">
            <p
              className="text-[9px] font-bold uppercase tracking-widest px-2 mb-3"
              style={{ color: "var(--sidebar-muted)" }}
            >
              Alerts
            </p>
            <div
              className="px-3 py-2.5 rounded-xl space-y-2"
              style={{ background: "rgba(220,38,38,0.12)", border: "1px solid rgba(220,38,38,0.2)" }}
            >
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                <span className="text-[11px] font-semibold text-red-400">
                  Top {topCritical.length} Critical Cases
                </span>
              </div>
              {topCritical.map((a) => (
                <div key={a.hadm_id} className="flex items-center justify-between">
                  <span className="text-[10px] font-mono" style={{ color: "var(--sidebar-text)" }}>
                    {a.hadm_id}
                  </span>
                  <span className="text-[10px] font-mono font-bold text-red-400">
                    {(a.prediction.predicted_prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Footer */}
      <div className="p-4" style={{ borderTop: "1px solid var(--sidebar-border)" }}>
        <div className="flex items-center gap-2.5 mb-3">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ background: "rgba(255,255,255,0.15)" }}
          >
            <span className="text-[11px] font-bold text-white">DR</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-white leading-tight truncate">Dr. Researcher</p>
            <p className="text-[10px] leading-tight" style={{ color: "var(--sidebar-muted)" }}>Cardiology AI</p>
          </div>
        </div>

        <button
          className="w-full flex items-center justify-center gap-2 py-2 rounded-xl text-[11px] font-semibold transition-all"
          style={{ background: "rgba(255,255,255,0.08)", color: "var(--sidebar-text)", border: "1px solid var(--sidebar-border)" }}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.14)"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.08)"; }}
        >
          <LogOut className="w-3.5 h-3.5" />
          Logout
        </button>

        <p className="text-[9px] mt-3 text-center" style={{ color: "var(--sidebar-muted)" }}>
          MIMIC-IV · N=40,255 · Early Fusion
        </p>
      </div>
    </aside>
  );
}
