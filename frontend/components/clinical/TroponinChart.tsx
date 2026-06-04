"use client";
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Area, AreaChart,
} from "recharts";
import { ClinicalTimestep } from "@/lib/utils/mockData";

interface TroponinChartProps {
  timelines: ClinicalTimestep[];
  activeTimestep: number;
}

const NORMAL_UPPER = 0.014; // 99th percentile hs-cTnT

type TroponinPoint = ClinicalTimestep;

interface CustomDotProps {
  cx?: number;
  cy?: number;
  payload?: TroponinPoint;
  activeTimestep: number;
}

function CustomDot({ cx, cy, payload, activeTimestep }: CustomDotProps) {
  if (cx == null || cy == null || !payload) return null;

  const isActive = payload.timestep === activeTimestep;
  const isHigh = payload.trop_value > NORMAL_UPPER;

  return (
    <g>
      {isActive && (
        <circle cx={cx} cy={cy} r={14} fill={isHigh ? "rgba(220,38,38,0.08)" : "rgba(5,150,105,0.08)"} />
      )}
      <circle
        cx={cx} cy={cy} r={isActive ? 6 : 4}
        fill={isHigh ? "#dc2626" : "#059669"}
        stroke={isActive ? "#ffffff" : "transparent"}
        strokeWidth={isActive ? 2 : 0}
      />
    </g>
  );
}

interface TooltipPayload {
  payload: TroponinPoint;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
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
                  <stop offset="5%" stopColor="#dc2626" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#dc2626" stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: "#6b9580", fontFamily: "JetBrains Mono" }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#6b9580", fontFamily: "JetBrains Mono" }}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <ReferenceLine
                y={NORMAL_UPPER}
                stroke="#f59e0b"
                strokeDasharray="4 2"
                strokeWidth={1}
                label={{ value: "ULN 0.014", position: "insideTopRight", fontSize: 9, fill: "#d97706" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="trop_value"
                stroke="#dc2626"
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
