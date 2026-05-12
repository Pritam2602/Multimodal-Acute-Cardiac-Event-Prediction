"use client";
import { Users, Heart, Activity, TrendingUp } from "lucide-react";
import { COHORT_STATS } from "@/lib/utils/mockData";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  icon: React.ReactNode;
  color: "cyan" | "danger" | "amber" | "safe";
}

function MetricCard({ label, value, sub, icon, color }: MetricCardProps) {
  const colors = {
    cyan:   { bg: "bg-cyan/5 border-cyan/20",   text: "text-cyan",   glow: "shadow-cyan-sm" },
    danger: { bg: "bg-danger/5 border-danger/20", text: "text-danger", glow: "shadow-danger" },
    amber:  { bg: "bg-amber/5 border-amber/20",  text: "text-amber",  glow: "" },
    safe:   { bg: "bg-safe/5 border-safe/20",    text: "text-safe",   glow: "shadow-safe" },
  };
  const c = colors[color];

  return (
    <div className={`flex items-center gap-4 px-5 py-3.5 rounded-xl border ${c.bg}`}>
      <div className={`${c.text} opacity-80`}>{icon}</div>
      <div>
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest">{label}</p>
        <p className={`text-xl font-bold font-mono ${c.text} leading-tight`}>{value}</p>
        {sub && <p className="text-[10px] text-text-muted">{sub}</p>}
      </div>
    </div>
  );
}

export default function MetricsBar() {
  return (
    <header className="px-6 py-3 border-b border-border-default bg-surface flex items-center gap-3 flex-wrap">
      <div className="flex items-center gap-2 mr-4">
        <div className="w-1.5 h-5 bg-cyan rounded-full" />
        <div>
          <h1 className="text-sm font-semibold text-text-primary leading-tight">Temporal Multimodal AMI Prediction</h1>
          <p className="text-[10px] text-text-muted">Clinical AI Reasoning Workstation · MIMIC-IV Cohort</p>
        </div>
      </div>

      <div className="flex items-center gap-3 flex-wrap ml-auto">
        <MetricCard
          label="Total Cohort"
          value={COHORT_STATS.total_admissions.toLocaleString()}
          sub="admissions"
          icon={<Users className="w-5 h-5" />}
          color="cyan"
        />
        <MetricCard
          label="AMI Prevalence"
          value={`${(COHORT_STATS.ami_prevalence * 100).toFixed(1)}%`}
          sub="31.27% of cohort"
          icon={<Heart className="w-5 h-5" />}
          color="danger"
        />
        <MetricCard
          label="Confidence Threshold"
          value={`${COHORT_STATS.model_threshold}`}
          sub="Youden's J optimized"
          icon={<Activity className="w-5 h-5" />}
          color="amber"
        />
        <MetricCard
          label="Early Fusion F1"
          value={COHORT_STATS.model_f1.toFixed(4)}
          sub="Best model in cohort"
          icon={<TrendingUp className="w-5 h-5" />}
          color="safe"
        />
      </div>
    </header>
  );
}
