"use client";
import { useState } from "react";
import { GitCompare, ChevronDown } from "lucide-react";
import { MOCK_ADMISSIONS, Admission } from "@/lib/utils/mockData";
import Sidebar from "@/components/ui/Sidebar";
import MetricsBar from "@/components/ui/MetricsBar";
import { RiskBadge, PredBadge, GroundTruthBadge } from "@/components/ui/Badge";
import ECGViewer from "@/components/ecg-viewer/ECGViewer";
import TroponinChart from "@/components/clinical/TroponinChart";
import ConfidenceGauge from "@/components/ui/ConfidenceGauge";

function CaseSelector({ value, onChange, exclude }: { value: string; onChange: (id: string) => void; exclude?: string }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full appearance-none bg-elevated border border-border-default rounded-lg px-3 py-2.5 text-xs text-text-primary focus:outline-none focus:border-cyan/50 pr-8"
      >
        <option value="">Select admission...</option>
        {MOCK_ADMISSIONS
          .filter(a => a.hadm_id !== exclude)
          .map(a => (
            <option key={a.hadm_id} value={a.hadm_id}>
              {a.hadm_id} · {a.subject_id} · {a.risk_level} · {(a.prediction.predicted_prob * 100).toFixed(0)}%
            </option>
          ))}
      </select>
      <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
    </div>
  );
}

interface CaseColumnProps {
  admission: Admission;
  label: string;
}

function CaseColumn({ admission, label }: CaseColumnProps) {
  const [ts, setTs] = useState(0);

  return (
    <div className="flex-1 flex flex-col overflow-hidden border border-border-default rounded-xl bg-surface overflow-y-auto">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border-default bg-elevated/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{label}</span>
          <div className="flex items-center gap-1.5">
            <RiskBadge level={admission.risk_level} />
            <PredBadge prob={admission.prediction.predicted_prob} />
            <GroundTruthBadge isAMI={admission.ground_truth_ami} />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <p className="text-sm font-mono font-bold text-text-primary">{admission.hadm_id}</p>
          <span className="text-text-muted">·</span>
          <p className="text-xs text-text-secondary">{admission.subject_id} · {admission.age}y {admission.gender}</p>
        </div>
        <div className="flex items-center gap-1 mt-2">
          {admission.timelines.map(t => (
            <button
              key={t.timestep}
              onClick={() => setTs(t.timestep)}
              className={`px-2.5 py-1 rounded text-[10px] font-mono font-semibold transition-all ${
                ts === t.timestep
                  ? "bg-cyan/10 text-cyan border border-cyan/40"
                  : "bg-elevated text-text-muted border border-border-default hover:text-text-primary"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Confidence Gauge */}
      <div className="flex items-center justify-center py-4 border-b border-border-default">
        <ConfidenceGauge probability={admission.prediction.predicted_prob} size={140} />
      </div>

      {/* ECG mini viewer */}
      <div className="h-56 border-b border-border-default flex-shrink-0">
        <ECGViewer
          admission={admission}
          activeTimestep={ts}
          attentionWeights={admission.attention.leads}
          onTimestepChange={setTs}
        />
      </div>

      {/* Troponin chart */}
      <div className="p-4 border-b border-border-default">
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">Troponin Trajectory</p>
        <TroponinChart timelines={admission.timelines} activeTimestep={ts} />
      </div>

      {/* Attention dominance */}
      <div className="p-4 border-b border-border-default">
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">Dominant Region</p>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-bold ${
            admission.attention.dominant_region === "Inferior" ? "text-danger" :
            admission.attention.dominant_region === "Anterior/Septal" ? "text-cyan" :
            admission.attention.dominant_region === "Lateral" ? "text-amber" : "text-text-muted"
          }`}>{admission.attention.dominant_region}</span>
        </div>
        <div className="mt-2 space-y-1">
          {admission.attention.temporal.map((w, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-text-muted w-5">T{i}</span>
              <div className="flex-1 h-1.5 bg-border-subtle rounded-full overflow-hidden">
                <div className="h-full bg-purple rounded-full" style={{ width: `${w * 100}%` }} />
              </div>
              <span className="text-[10px] font-mono text-purple w-8 text-right">{(w * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Explanation */}
      <div className="p-4">
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">AI Explanation</p>
        <p className="text-[11px] text-text-secondary leading-relaxed">{admission.explanation}</p>
      </div>
    </div>
  );
}

export default function ComparePage() {
  const [leftId, setLeftId] = useState("HADM-109441");  // CKD false positive
  const [rightId, setRightId] = useState("HADM-100234"); // True AMI

  const leftCase = MOCK_ADMISSIONS.find(a => a.hadm_id === leftId);
  const rightCase = MOCK_ADMISSIONS.find(a => a.hadm_id === rightId);

  return (
    <div className="flex h-screen overflow-hidden bg-base">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <MetricsBar />

        {/* Compare header */}
        <div className="px-6 py-3 border-b border-border-default bg-surface">
          <div className="flex items-center gap-3">
            <GitCompare className="w-4 h-4 text-cyan" />
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Case Comparison</h2>
              <p className="text-[11px] text-text-muted">Side-by-side analysis of two admissions for educational and research use</p>
            </div>
          </div>

          {/* Selectors */}
          <div className="flex items-center gap-4 mt-3">
            <div className="flex-1">
              <p className="text-[10px] text-text-muted mb-1 uppercase tracking-wider">Case A</p>
              <CaseSelector value={leftId} onChange={setLeftId} exclude={rightId} />
            </div>
            <div className="flex items-center gap-1 mt-5 text-text-muted text-lg font-light">vs</div>
            <div className="flex-1">
              <p className="text-[10px] text-text-muted mb-1 uppercase tracking-wider">Case B</p>
              <CaseSelector value={rightId} onChange={setRightId} exclude={leftId} />
            </div>
          </div>
        </div>

        {/* Columns */}
        <div className="flex-1 flex gap-4 p-4 overflow-hidden">
          {leftCase ? (
            <CaseColumn admission={leftCase} label="Case A" />
          ) : (
            <div className="flex-1 flex items-center justify-center border border-dashed border-border-default rounded-xl text-text-muted text-xs">
              Select a case above
            </div>
          )}
          {rightCase ? (
            <CaseColumn admission={rightCase} label="Case B" />
          ) : (
            <div className="flex-1 flex items-center justify-center border border-dashed border-border-default rounded-xl text-text-muted text-xs">
              Select a case above
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
