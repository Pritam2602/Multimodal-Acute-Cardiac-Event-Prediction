"use client";
import { ChevronLeft, User, Clock, Calendar, Activity } from "lucide-react";
import { useRouter } from "next/navigation";
import { Admission } from "@/lib/utils/mockData";
import { RiskBadge, PredBadge, GroundTruthBadge } from "@/components/ui/Badge";

interface PatientHeaderProps {
  admission: Admission;
  activeTimestep: number;
  onTimestepChange: (t: number) => void;
}

export default function PatientHeader({ admission, activeTimestep, onTimestepChange }: PatientHeaderProps) {
  const router = useRouter();

  return (
    <div className="bg-surface border-b border-border-default px-4 py-3">
      <div className="flex items-center gap-4 flex-wrap">
        {/* Back + ID */}
        <button
          onClick={() => router.push("/dashboard")}
          className="flex items-center gap-1.5 text-text-muted hover:text-text-primary transition-colors group"
        >
          <ChevronLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
          <span className="text-xs">Admissions</span>
        </button>

        <div className="w-px h-5 bg-border-default" />

        {/* Admission ID */}
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-5 rounded-full bg-cyan" />
          <div>
            <p className="text-xs font-mono font-bold text-text-primary">{admission.hadm_id}</p>
            <p className="text-[10px] text-text-muted">Clinical Workstation</p>
          </div>
        </div>

        {/* Patient info */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-elevated border border-border-default">
          <User className="w-3.5 h-3.5 text-text-muted" />
          <div>
            <p className="text-[11px] font-mono font-semibold text-text-primary">{admission.subject_id}</p>
            <p className="text-[9px] text-text-muted">{admission.age}y {admission.gender}</p>
          </div>
        </div>

        {/* Admit time */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-elevated border border-border-default">
          <Calendar className="w-3.5 h-3.5 text-text-muted" />
          <div>
            <p className="text-[11px] font-mono text-text-primary">{admission.admittime}</p>
            <p className="text-[9px] text-text-muted">Admission time</p>
          </div>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-2">
          <RiskBadge level={admission.risk_level} />
          <PredBadge prob={admission.prediction.predicted_prob} />
          <GroundTruthBadge isAMI={admission.ground_truth_ami} />
        </div>

        {/* Timestep switcher */}
        <div className="ml-auto flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-[10px] text-text-muted">
            <Activity className="w-3.5 h-3.5" />
            <span>Observation</span>
          </div>
          <div className="flex items-center gap-1">
            {admission.timelines.map((t) => (
              <button
                key={t.timestep}
                onClick={() => onTimestepChange(t.timestep)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                  activeTimestep === t.timestep
                    ? "bg-cyan/10 text-cyan border border-cyan/40 shadow-cyan-sm"
                    : "bg-elevated text-text-muted border border-border-default hover:text-text-primary hover:border-border-bright"
                }`}
              >
                {t.label}
                <span className="block text-[8px] font-normal opacity-70">+{t.time_delta_hrs.toFixed(1)}h</span>
              </button>
            ))}
            {admission.timelines.length < 3 && (
              <button disabled className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-elevated/50 text-text-muted border border-border-subtle opacity-30 cursor-not-allowed">
                T₂<span className="block text-[8px] font-normal">—</span>
              </button>
            )}
          </div>
        </div>

        {/* Confidence evolution mini */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-elevated border border-border-default">
          <Clock className="w-3.5 h-3.5 text-text-muted" />
          <div className="flex items-center gap-1">
            {admission.prediction.confidence_evolution.map((c, i) => (
              <span key={i} className="flex items-center gap-0.5">
                {i > 0 && <span className="text-text-muted text-[10px]">→</span>}
                <span className={`text-[11px] font-mono font-bold ${c >= 0.6494 ? "text-danger" : "text-safe"}`}>
                  {(c * 100).toFixed(0)}%
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
