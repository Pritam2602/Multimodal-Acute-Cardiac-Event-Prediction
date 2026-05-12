"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, LayoutDashboard, GitCompare, Upload, Cpu, AlertTriangle } from "lucide-react";
import { COHORT_STATS } from "@/lib/utils/mockData";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Admissions", icon: LayoutDashboard, desc: "Patient cohort table" },
  { href: "/compare", label: "Case Compare", icon: GitCompare, desc: "Side-by-side analysis" },
  { href: "/upload", label: "Upload Patient", icon: Upload, desc: "New admission entry" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 min-h-screen flex flex-col bg-surface border-r border-border-default">
      {/* Logo */}
      <div className="p-5 border-b border-border-default">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-9 h-9 rounded-lg bg-cyan/10 border border-cyan/30 flex items-center justify-center">
              <Activity className="w-5 h-5 text-cyan" />
            </div>
            <div className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan animate-pulse-slow" />
          </div>
          <div>
            <p className="text-sm font-semibold text-text-primary leading-tight">AMI Predict</p>
            <p className="text-[10px] text-text-muted leading-tight">Temporal Multimodal AI</p>
          </div>
        </div>
      </div>

      {/* Model Status */}
      <div className="mx-4 mt-4 p-3 rounded-lg bg-elevated border border-border-default">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-cyan" />
            <span className="text-[11px] font-semibold text-text-secondary uppercase tracking-wider">Model Status</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-safe animate-pulse-slow" />
            <span className="text-[10px] text-safe">Online</span>
          </div>
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <span className="text-[11px] text-text-muted">F1 Score</span>
            <span className="text-[11px] font-mono text-cyan">{COHORT_STATS.model_f1.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[11px] text-text-muted">AUC-ROC</span>
            <span className="text-[11px] font-mono text-cyan">{COHORT_STATS.model_auc.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[11px] text-text-muted">Threshold</span>
            <span className="text-[11px] font-mono text-amber">{COHORT_STATS.model_threshold}</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 mt-2">
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest px-2 mb-2">Navigation</p>
        <div className="space-y-0.5">
          {NAV_ITEMS.map(({ href, label, icon: Icon, desc }) => {
            const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all group ${
                  active
                    ? "bg-cyan/10 border-l-2 border-cyan text-cyan"
                    : "text-text-secondary hover:bg-elevated hover:text-text-primary border-l-2 border-transparent"
                }`}
              >
                <Icon className={`w-4 h-4 flex-shrink-0 ${active ? "text-cyan" : "text-text-muted group-hover:text-text-secondary"}`} />
                <div>
                  <p className="text-xs font-medium leading-tight">{label}</p>
                  <p className="text-[10px] text-text-muted leading-tight">{desc}</p>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Alert Section */}
        <div className="mt-4">
          <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest px-2 mb-2">Alerts</p>
          <div className="px-3 py-2.5 rounded-lg bg-danger/5 border border-danger/20">
            <div className="flex items-center gap-2 mb-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-danger" />
              <span className="text-[11px] font-semibold text-danger">3 Critical Cases</span>
            </div>
            <p className="text-[10px] text-text-muted">HADM-112987, HADM-128001, HADM-142387 require immediate review</p>
          </div>
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border-default">
        <div className="flex items-center gap-2 px-2">
          <div className="w-7 h-7 rounded-full bg-elevated border border-border-default flex items-center justify-center">
            <span className="text-[10px] font-bold text-text-secondary">DR</span>
          </div>
          <div>
            <p className="text-xs font-medium text-text-primary">Dr. Researcher</p>
            <p className="text-[10px] text-text-muted">Cardiology AI</p>
          </div>
        </div>
        <p className="text-[9px] text-text-muted mt-3 px-2">MIMIC-IV Cohort · N=40,255</p>
        <p className="text-[9px] text-text-muted px-2">Early Fusion · PyTorch</p>
      </div>
    </aside>
  );
}
