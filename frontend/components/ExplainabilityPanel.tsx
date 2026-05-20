"use client";
import { MessageSquare, Clock, Cpu } from "lucide-react";
import { Admission } from "@/lib/utils/mockData";

interface ExplainabilityPanelProps {
  admission: Admission;
}

export default function ExplainabilityPanel({ admission }: ExplainabilityPanelProps) {
  const dominant = admission.attention.dominant_region;
  const prob = admission.prediction.predicted_prob;
  const isAMI = prob >= 0.6494;

  return (
    <div className="space-y-4">
      {/* Main narrative */}
      <div className={`p-4 rounded-xl border ${isAMI ? "bg-danger/5 border-danger/20" : "bg-safe/5 border-safe/20"}`}>
        <div className="flex items-start gap-2">
          <MessageSquare className={`w-4 h-4 flex-shrink-0 mt-0.5 ${isAMI ? "text-danger" : "text-safe"}`} />
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-text-muted mb-2">AI Clinical Narrative</p>
            <p className="text-xs text-text-primary leading-relaxed">{admission.explanation}</p>
          </div>
        </div>
      </div>

      {/* Temporal reasoning */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Clock className="w-4 h-4 text-cyan" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Temporal Reasoning</h3>
        </div>
        <p className="text-xs text-text-secondary leading-relaxed">{admission.explanation_temporal}</p>
      </div>

      {/* Key factors */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-purple" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Decision Factors</h3>
        </div>
        <div className="space-y-2">
          {[
            {
              label: "Primary ECG Region",
              value: dominant,
              color: dominant === "Inferior" ? "text-danger" : dominant === "Anterior/Septal" ? "text-cyan" : dominant === "Lateral" ? "text-amber" : "text-text-secondary",
              icon: "📍",
            },
            {
              label: "Troponin Velocity",
              value: admission.timelines.length > 1
                ? `+${((admission.max_troponin - admission.timelines[0].trop_value) / (admission.timelines[admission.timelines.length - 1].time_delta_hrs || 1)).toFixed(3)} ng/mL/hr`
                : "Single timepoint",
              color: "text-text-primary",
              icon: "📈",
            },
            {
              label: "Max Fold Rise",
              value: `${admission.timelines[admission.timelines.length - 1].fold_rise.toFixed(2)}×`,
              color: admission.timelines[admission.timelines.length - 1].fold_rise > 3 ? "text-danger" : "text-amber",
              icon: "🔺",
            },
            {
              label: "Confounder Status",
              value: admission.timelines.some(t => t.ckd_active || t.sepsis_active) ? "CKD/Sepsis Active — gated" : "No confounders",
              color: admission.timelines.some(t => t.ckd_active || t.sepsis_active) ? "text-amber" : "text-safe",
              icon: "⚠️",
            },
            {
              label: "Temporal Entropy",
              value: `${admission.prediction.attn_temp_entropy.toFixed(3)} (${admission.prediction.attn_temp_entropy < 0.35 ? "confident" : admission.prediction.attn_temp_entropy > 0.65 ? "uncertain" : "moderate"})`,
              color: "text-text-secondary",
              icon: "🎯",
            },
          ].map(({ label, value, color, icon }) => (
            <div key={label} className="flex items-center justify-between py-2 border-b border-border-subtle last:border-0">
              <div className="flex items-center gap-2">
                <span className="text-sm">{icon}</span>
                <span className="text-[11px] text-text-muted">{label}</span>
              </div>
              <span className={`text-[11px] font-mono font-semibold ${color}`}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
