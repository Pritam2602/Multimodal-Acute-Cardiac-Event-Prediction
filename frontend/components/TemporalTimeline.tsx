"use client";
import { Admission } from "@/lib/utils/mockData";

interface TemporalTimelineProps {
  admission: Admission;
  activeTimestep: number;
  onTimestepChange: (t: number) => void;
}

function getProbColor(prob: number): string {
  if (prob >= 0.75) return "#f43f5e";
  if (prob >= 0.6494) return "#f97316";
  if (prob >= 0.25) return "#f59e0b";
  return "#10b981";
}

export default function TemporalTimeline({ admission, activeTimestep, onTimestepChange }: TemporalTimelineProps) {
  const { timelines, prediction } = admission;
  const maxTime = timelines[timelines.length - 1].time_delta_hrs;
  const timelineWidth = maxTime > 0 ? maxTime : 1;

  return (
    <div className="card p-4 mx-4 mt-3">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-xs font-semibold text-text-primary uppercase tracking-wider">Temporal Reasoning Timeline</p>
          <p className="text-[10px] text-text-muted">Admission progression · Model confidence evolution</p>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-text-muted">
          <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-cyan" />ECG scan</div>
          <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-danger" />Troponin draw</div>
          <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-purple" />AI confidence</div>
        </div>
      </div>

      <div className="relative">
        {/* Timeline track */}
        <div className="relative h-2 bg-border-subtle rounded-full mb-6">
          {/* Filled progress */}
          <div
            className="absolute top-0 left-0 h-full bg-border-default rounded-full"
            style={{ width: "100%" }}
          />

          {/* Threshold line at ~48% confidence */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-amber/60"
            style={{ left: "48%" }}
            title="AI threshold 0.6494"
          />

          {/* Timestep markers */}
          {timelines.map((t) => {
            const leftPct = maxTime > 0 ? (t.time_delta_hrs / timelineWidth) * 100 : (t.timestep / Math.max(timelines.length - 1, 1)) * 100;
            const confidence = prediction.confidence_evolution[t.timestep] ?? 0;
            const color = getProbColor(confidence);
            const isActive = t.timestep === activeTimestep;

            return (
              <button
                key={t.timestep}
                onClick={() => onTimestepChange(t.timestep)}
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 group z-10"
                style={{ left: `${leftPct}%` }}
              >
                {/* Outer ring */}
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all ${
                  isActive ? "scale-125" : "scale-100 hover:scale-110"
                }`}
                  style={{
                    background: `${color}20`,
                    borderColor: color,
                    boxShadow: isActive ? `0 0 12px ${color}80` : "none"
                  }}>
                  <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                </div>

                {/* Label above */}
                <div className="absolute bottom-7 left-1/2 -translate-x-1/2 text-center w-16">
                  <p className="text-[9px] font-mono font-bold" style={{ color }}>{t.label}</p>
                  <p className="text-[8px] text-text-muted">+{t.time_delta_hrs.toFixed(1)}h</p>
                </div>

                {/* Confidence below */}
                <div className="absolute top-7 left-1/2 -translate-x-1/2 text-center w-16 opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="px-1.5 py-1 rounded bg-card border border-border-default">
                    <p className="text-[9px] font-mono font-bold" style={{ color }}>{(confidence * 100).toFixed(0)}%</p>
                    <p className="text-[8px] text-text-muted">AMI prob</p>
                    <p className="text-[8px] text-text-muted">Trop: {t.trop_value.toFixed(3)}</p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Confidence evolution curve (CSS-only) */}
        <div className="flex items-end gap-1 h-16 mt-2">
          {prediction.confidence_evolution.map((conf, i) => {
            const color = getProbColor(conf);
            const label = timelines[i]?.label ?? `T${i}`;
            return (
              <div key={i} className="flex-1 flex flex-col items-center gap-1">
                <div className="w-full relative flex items-end justify-center" style={{ height: "44px" }}>
                  <div
                    className="w-full rounded-t transition-all duration-700 cursor-pointer hover:opacity-80"
                    style={{
                      height: `${conf * 100}%`,
                      background: `${color}30`,
                      border: `1px solid ${color}60`,
                      borderBottom: "none",
                      boxShadow: i === activeTimestep ? `0 0 8px ${color}50` : "none",
                    }}
                    onClick={() => onTimestepChange(i)}
                  />
                  <span className="absolute -top-4 text-[9px] font-mono font-bold" style={{ color }}>
                    {(conf * 100).toFixed(0)}%
                  </span>
                </div>
                <span className="text-[9px] text-text-muted font-mono">{label}</span>
              </div>
            );
          })}
          {prediction.confidence_evolution.length < 3 && (
            <div className="flex-1 flex flex-col items-center gap-1 opacity-20">
              <div className="w-full" style={{ height: "44px" }} />
              <span className="text-[9px] text-text-muted font-mono">T₂</span>
            </div>
          )}
        </div>

        {/* Threshold reference */}
        <div className="flex items-center gap-2 mt-3 pt-2 border-t border-border-subtle">
          <div className="w-3 h-0.5 bg-amber" />
          <span className="text-[9px] text-text-muted">Decision threshold: 0.6494 (Youden&apos;s J optimal)</span>
          <span className="ml-auto text-[9px] text-text-muted">
            Final: <span className={`font-mono font-bold ${prediction.predicted_prob >= 0.6494 ? "text-danger" : "text-safe"}`}>
              {(prediction.predicted_prob * 100).toFixed(1)}%
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}
