"use client";
import { Brain, Clock, Map } from "lucide-react";
import { AttentionWeights } from "@/lib/utils/mockData";
import AnatomicalHeart from "./AnatomicalHeart";

interface AttentionPanelProps {
  attention: AttentionWeights;
  activeTimestep: number;
}

function LeadHeatmapCell({ lead, weight }: { lead: string; weight: number }) {
  let bg: string, text: string;
  if (weight >= 0.80) { bg = "bg-danger/20 border-danger/50"; text = "text-danger"; }
  else if (weight >= 0.60) { bg = "bg-orange-50 border-orange-200"; text = "text-orange-600"; }
  else if (weight >= 0.40) { bg = "bg-amber/10 border-amber/30"; text = "text-amber"; }
  else if (weight >= 0.20) { bg = "bg-cyan/5 border-cyan/20"; text = "text-cyan"; }
  else { bg = "bg-elevated border-border-default"; text = "text-text-muted"; }

  return (
    <div className={`flex flex-col items-center p-2 rounded border ${bg} transition-all`}>
      <span className={`text-[10px] font-mono font-bold ${text}`}>{lead}</span>
      <span className={`text-[9px] font-mono ${text} opacity-80`}>{(weight * 100).toFixed(0)}%</span>
      <div className="w-full h-1 bg-border-subtle rounded-full mt-1 overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${weight * 100}%`, background: text === "text-danger" ? "#dc2626" : text === "text-amber" ? "#d97706" : text === "text-orange-600" ? "#ea580c" : text === "text-cyan" ? "#0284c7" : "#94a3b8" }} />
      </div>
    </div>
  );
}

export default function AttentionPanel({ attention, activeTimestep }: AttentionPanelProps) {
  const LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

  return (
    <div className="space-y-4">
      {/* Anatomical Heart */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Map className="w-4 h-4 text-amber" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Anatomical Attention Map</h3>
          <span className="ml-auto text-[10px] text-text-muted">Active: T{activeTimestep}</span>
        </div>
        <AnatomicalHeart attention={attention} />
      </div>

      {/* Lead-by-lead heatmap */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-4 h-4 text-purple" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Lead Attention Weights</h3>
        </div>
        <div className="grid grid-cols-6 gap-1.5">
          {LEAD_ORDER.map((lead) => (
            <LeadHeatmapCell key={lead} lead={lead} weight={attention.leads[lead] ?? 0} />
          ))}
        </div>
        {/* Gradient legend */}
        <div className="flex items-center gap-2 mt-3">
          <span className="text-[9px] text-text-muted">Low</span>
          <div className="flex-1 h-1.5 rounded-full" style={{
            background: "linear-gradient(to right, #94a3b8, #0284c7, #d97706, #ea580c, #dc2626)"
          }} />
          <span className="text-[9px] text-text-muted">High Attention</span>
        </div>
      </div>

      {/* Temporal attention */}
      {attention.temporal.length > 1 && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4 text-cyan" />
            <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Temporal GRU Gating</h3>
          </div>
          <div className="space-y-2">
            {attention.temporal.map((weight, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className={`text-[10px] font-mono font-bold w-6 ${i === activeTimestep ? "text-cyan" : "text-text-muted"}`}>T{i}</span>
                <div className="flex-1 h-2 bg-border-subtle rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${i === activeTimestep ? "bg-cyan" : "bg-text-muted"}`}
                    style={{
                      width: `${weight * 100}%`,
                      boxShadow: i === activeTimestep ? "0 0 8px rgba(0,212,255,0.6)" : "none"
                    }}
                  />
                </div>
                <span className={`text-[10px] font-mono w-10 text-right ${i === activeTimestep ? "text-cyan" : "text-text-muted"}`}>
                  {(weight * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-text-muted mt-3 italic">
            Temporal weights represent the GRU&apos;s relative contribution weighting across observation timesteps.
          </p>
        </div>
      )}
    </div>
  );
}
