"use client";
import { Users, Heart, Activity, TrendingUp } from "lucide-react";
import { COHORT_STATS } from "@/lib/utils/mockData";
import { useAdmissionStore } from "@/lib/store/useAdmissionStore";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  icon: React.ReactNode;
  color: "green" | "danger" | "amber" | "purple";
}

function MetricCard({ label, value, sub, icon, color }: MetricCardProps) {
  const colors = {
    green:  { bg: "bg-cyan/8 border-cyan/20",    text: "text-cyan",   iconBg: "bg-cyan/10" },
    danger: { bg: "bg-danger/5 border-danger/20", text: "text-danger", iconBg: "bg-danger/10" },
    amber:  { bg: "bg-amber/5 border-amber/20",   text: "text-amber",  iconBg: "bg-amber/10" },
    purple: { bg: "bg-purple/5 border-purple/20", text: "text-purple", iconBg: "bg-purple/10" },
  };
  const c = colors[color];

  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border bg-surface shadow-card`}
      style={{ borderColor: "var(--tw-border-opacity, #d0e8d8)" }}>
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 ${c.iconBg}`}>
        <span className={c.text}>{icon}</span>
      </div>
      <div>
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest leading-tight">{label}</p>
        <p className={`text-lg font-bold font-mono ${c.text} leading-tight`}>{value}</p>
        {sub && <p className="text-[10px] text-text-muted">{sub}</p>}
      </div>
    </div>
  );
}

export default function MetricsBar() {
  const { totalAdmissions, usingLiveData } = useAdmissionStore();
  const displayTotal = usingLiveData ? totalAdmissions : COHORT_STATS.total_admissions;

  return (
    <header className="px-6 py-3 border-b border-border-default bg-surface flex items-center gap-3 flex-wrap">
      <div className="flex items-center gap-2.5 mr-4">
        <div className="w-1.5 h-6 rounded-full bg-cyan" />
        <div>
          <h1 className="text-sm font-bold text-text-primary leading-tight">Temporal Multimodal AMI Prediction</h1>
          <p className="text-[10px] text-text-muted">Clinical AI Reasoning Workstation · MIMIC-IV Cohort</p>
        </div>
      </div>

      <div className="flex items-center gap-2.5 flex-wrap ml-auto">
        <MetricCard
          label="Total Cohort"
          value={displayTotal.toLocaleString()}
          sub="admissions"
          icon={<Users className="w-4.5 h-4.5" />}
          color="green"
        />
        <MetricCard
          label="AMI Prevalence"
          value={`${(COHORT_STATS.ami_prevalence * 100).toFixed(1)}%`}
          sub="31.27% of cohort"
          icon={<Heart className="w-4.5 h-4.5" />}
          color="danger"
        />
        <MetricCard
          label="Threshold"
          value={`${COHORT_STATS.model_threshold}`}
          sub="Youden's J optimized"
          icon={<Activity className="w-4.5 h-4.5" />}
          color="amber"
        />
        <MetricCard
          label="Early Fusion F1"
          value={COHORT_STATS.model_f1.toFixed(4)}
          sub="Best model in cohort"
          icon={<TrendingUp className="w-4.5 h-4.5" />}
          color="purple"
        />
      </div>
    </header>
  );
}
