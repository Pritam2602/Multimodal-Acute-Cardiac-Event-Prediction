"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Area, AreaChart, Dot,
} from "recharts";
import { ClinicalTimestep } from "@/lib/utils/mockData";

interface TroponinChartProps {
  timelines: ClinicalTimestep[];
  activeTimestep: number;
}

const NORMAL_UPPER = 0.014; // 99th percentile hs-cTnT

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomDot(props: any) {
  const { cx, cy, payload, activeTimestep } = props;
  const isActive = payload.timestep === activeTimestep;
  const isHigh = payload.trop_value > NORMAL_UPPER;

  return (
    <g>
      {isActive && (
        <circle cx={cx} cy={cy} r={14} fill={isHigh ? "rgba(244,63,94,0.1)" : "rgba(16,185,129,0.1)"} />
      )}
      <circle
        cx={cx} cy={cy} r={isActive ? 6 : 4}
        fill={isHigh ? "#f43f5e" : "#10b981"}
        stroke={isActive ? "#ffffff" : "transparent"}
        strokeWidth={isActive ? 2 : 0}
      />
    </g>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-card border border-border-default rounded-lg p-3 text-xs shadow-lg">
      <p className="font-mono font-bold text-cyan mb-1">{d.label}</p>
      <p className="text-text-secondary">
        Troponin: <span className={`font-mono font-bold ${d.trop_value > NORMAL_UPPER ? "text-danger" : "text-safe"}`}>{d.trop_value.toFixed(3)} ng/mL</span>
      </p>
      <p className="text-text-secondary">
        Fold Rise: <span className="font-mono text-amber">{d.fold_rise.toFixed(2)}×</span>
      </p>
      <p className="text-text-secondary">+{d.time_delta_hrs.toFixed(1)}h from admission</p>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ConfidenceLine({ timelines, confidenceEvolution }: { timelines: ClinicalTimestep[], confidenceEvolution?: number[] }) {
  const data = timelines.map((t, i) => ({
    label: t.label,
    confidence: confidenceEvolution?.[i] != null ? confidenceEvolution[i] * 100 : null,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#162845" />
        <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#4a6a94", fontFamily: "JetBrains Mono" }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#4a6a94", fontFamily: "JetBrains Mono" }}
          tickFormatter={(v) => `${v}%`} />
        <ReferenceLine y={48} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1} label={{ value: "Threshold", position: "insideTopRight", fontSize: 9, fill: "#f59e0b" }} />
        <Line
          type="monotone" dataKey="confidence" stroke="#a855f7" strokeWidth={2}
          dot={{ r: 4, fill: "#a855f7", stroke: "#fff", strokeWidth: 1 }}
          strokeDasharray={undefined}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function TroponinChart({ timelines, activeTimestep }: TroponinChartProps) {
  const data = timelines.map(t => ({ ...t }));

  return (
    <div className="space-y-3">
      {/* Troponin trajectory */}
      <div>
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest mb-2">hs-cTnT Trajectory (ng/mL)</p>
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: 0 }}>
              <defs>
                <linearGradient id="tropGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#162845" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: "#4a6a94", fontFamily: "JetBrains Mono" }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#4a6a94", fontFamily: "JetBrains Mono" }}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <ReferenceLine
                y={NORMAL_UPPER}
                stroke="#f59e0b"
                strokeDasharray="4 2"
                strokeWidth={1}
                label={{ value: "ULN 0.014", position: "insideTopRight", fontSize: 9, fill: "#f59e0b" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="trop_value"
                stroke="#f43f5e"
                strokeWidth={2}
                fill="url(#tropGrad)"
                dot={(props) => <CustomDot {...props} activeTimestep={activeTimestep} />}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
