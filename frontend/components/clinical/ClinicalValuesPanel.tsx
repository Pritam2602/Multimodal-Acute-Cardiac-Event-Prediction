"use client";
import { Activity, Heart, Droplets, Thermometer, AlertTriangle, TrendingUp } from "lucide-react";
import { Admission } from "@/lib/utils/mockData";
import TroponinChart from "./TroponinChart";

interface ClinicalValuesPanelProps {
  admission: Admission;
  activeTimestep: number;
}

const NORMAL_UPPER = 0.014;

function CellValue({ value, unit, normal }: { value: number | null; unit: string; normal?: [number, number] }) {
  const isAbnormal = normal && value !== null && (value < normal[0] || value > normal[1]);
  return (
    <div className="text-right">
      {value !== null ? (
        <>
          <span className={`font-mono font-bold text-sm ${isAbnormal ? "text-amber" : "text-text-primary"}`}>
            {value.toFixed(value < 10 ? 2 : 0)}
          </span>
          <span className="text-[9px] text-text-muted ml-0.5">{unit}</span>
        </>
      ) : (
        <span className="text-text-muted font-mono text-xs">—</span>
      )}
    </div>
  );
}

function TropCell({ trop, baselineTrop }: { trop: number; baselineTrop: number }) {
  const isAboveULN = trop > NORMAL_UPPER;
  const foldRise = trop / baselineTrop;
  return (
    <div className="text-right">
      <div>
        <span className={`font-mono font-bold text-sm ${isAboveULN ? "text-danger" : "text-safe"}`}>
          {trop.toFixed(3)}
        </span>
        <span className="text-[9px] text-text-muted ml-0.5">ng/mL</span>
      </div>
      {foldRise > 1.05 && (
        <div className="flex items-center justify-end gap-0.5">
          <TrendingUp className="w-2.5 h-2.5 text-danger" />
          <span className="text-[9px] text-danger font-mono">{foldRise.toFixed(2)}×</span>
        </div>
      )}
    </div>
  );
}

export default function ClinicalValuesPanel({ admission, activeTimestep }: ClinicalValuesPanelProps) {
  const { timelines, comorbidities } = admission;
  const baselineTrop = timelines[0].trop_value;

  return (
    <div className="space-y-4 h-full overflow-y-auto pr-1">
      {/* Temporal Table */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-cyan" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Temporal Clinical Values</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border-default">
                <th className="text-left py-2 pr-4 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Parameter</th>
                {timelines.map((t) => (
                  <th key={t.timestep} className={`text-right py-2 px-2 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                    t.timestep === activeTimestep ? "text-cyan" : "text-text-muted"
                  }`}>
                    {t.label}
                    {t.timestep === activeTimestep && <span className="ml-1 text-[8px] text-cyan">●</span>}
                    <div className="text-[9px] font-normal text-text-muted">+{t.time_delta_hrs.toFixed(1)}h</div>
                  </th>
                ))}
                {timelines.length < 3 && (
                  <th className="text-right py-2 px-2 text-[10px] font-semibold text-text-muted uppercase tracking-wider opacity-30">
                    {timelines.length < 3 ? "T₂" : ""}
                    <div className="text-[9px] font-normal">—</div>
                  </th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {/* Troponin */}
              <tr>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-1.5">
                    <Droplets className="w-3 h-3 text-danger" />
                    <span className="text-text-secondary text-[11px]">hs-cTnT</span>
                  </div>
                </td>
                {timelines.map((t) => (
                  <td key={t.timestep} className={`py-2.5 px-2 transition-all ${t.timestep === activeTimestep ? "bg-cyan/5 rounded" : ""}`}>
                    <TropCell trop={t.trop_value} baselineTrop={baselineTrop} />
                  </td>
                ))}
                {timelines.length < 3 && <td className="py-2.5 px-2 text-right"><span className="font-mono text-text-muted text-xs">—</span></td>}
              </tr>

              {/* Heart Rate */}
              <tr>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-1.5">
                    <Heart className="w-3 h-3 text-text-muted" />
                    <span className="text-text-secondary text-[11px]">Heart Rate</span>
                  </div>
                </td>
                {timelines.map((t) => (
                  <td key={t.timestep} className={`py-2.5 px-2 ${t.timestep === activeTimestep ? "bg-cyan/5 rounded" : ""}`}>
                    <CellValue value={t.hr} unit="bpm" normal={[60, 100]} />
                  </td>
                ))}
                {timelines.length < 3 && <td className="py-2.5 px-2 text-right"><span className="font-mono text-text-muted text-xs">—</span></td>}
              </tr>

              {/* SBP */}
              <tr>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-1.5">
                    <Thermometer className="w-3 h-3 text-text-muted" />
                    <span className="text-text-secondary text-[11px]">SBP</span>
                  </div>
                </td>
                {timelines.map((t) => (
                  <td key={t.timestep} className={`py-2.5 px-2 ${t.timestep === activeTimestep ? "bg-cyan/5 rounded" : ""}`}>
                    <CellValue value={t.sbp} unit="mmHg" normal={[90, 140]} />
                  </td>
                ))}
                {timelines.length < 3 && <td className="py-2.5 px-2 text-right"><span className="font-mono text-text-muted text-xs">—</span></td>}
              </tr>

              {/* MAP */}
              <tr>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-1.5">
                    <Thermometer className="w-3 h-3 text-text-muted" />
                    <span className="text-text-secondary text-[11px]">MAP</span>
                  </div>
                </td>
                {timelines.map((t) => (
                  <td key={t.timestep} className={`py-2.5 px-2 ${t.timestep === activeTimestep ? "bg-cyan/5 rounded" : ""}`}>
                    <CellValue value={t.map} unit="mmHg" normal={[70, 100]} />
                  </td>
                ))}
                {timelines.length < 3 && <td className="py-2.5 px-2 text-right"><span className="font-mono text-text-muted text-xs">—</span></td>}
              </tr>

              {/* Creatinine */}
              <tr>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full border border-text-muted flex items-center justify-center">
                      <div className="w-1 h-1 rounded-full bg-text-muted" />
                    </div>
                    <span className="text-text-secondary text-[11px]">Creatinine</span>
                  </div>
                </td>
                {timelines.map((t) => (
                  <td key={t.timestep} className={`py-2.5 px-2 ${t.timestep === activeTimestep ? "bg-cyan/5 rounded" : ""}`}>
                    <CellValue value={t.creatinine} unit="mg/dL" normal={[0.6, 1.2]} />
                  </td>
                ))}
                {timelines.length < 3 && <td className="py-2.5 px-2 text-right"><span className="font-mono text-text-muted text-xs">—</span></td>}
              </tr>

              {/* Lactate */}
              <tr>
                <td className="py-2.5 pr-4">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full border border-text-muted flex items-center justify-center">
                      <div className="w-1 h-1 rounded-full bg-text-muted" />
                    </div>
                    <span className="text-text-secondary text-[11px]">Lactate</span>
                  </div>
                </td>
                {timelines.map((t) => (
                  <td key={t.timestep} className={`py-2.5 px-2 ${t.timestep === activeTimestep ? "bg-cyan/5 rounded" : ""}`}>
                    <CellValue value={t.lactate} unit="mmol/L" normal={[0.5, 2.0]} />
                  </td>
                ))}
                {timelines.length < 3 && <td className="py-2.5 px-2 text-right"><span className="font-mono text-text-muted text-xs">—</span></td>}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Confounder Status */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="w-4 h-4 text-amber" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Confounder Flags</h3>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {/* CKD */}
          {["CKD Stage 3", "CKD Stage 4", "CKD Stage 2"].some(c => comorbidities.includes(c)) || timelines.some(t => t.ckd_active) ? (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber/5 border border-amber/20">
              <div className="w-2 h-2 rounded-full bg-amber animate-pulse-slow" />
              <div>
                <p className="text-[10px] font-bold text-amber">CKD Active</p>
                <p className="text-[9px] text-text-muted">Troponin elevation confounded</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-safe/5 border border-safe/20">
              <div className="w-2 h-2 rounded-full bg-safe" />
              <div>
                <p className="text-[10px] font-bold text-safe">CKD Clear</p>
                <p className="text-[9px] text-text-muted">No renal confounding</p>
              </div>
            </div>
          )}

          {/* Sepsis */}
          {comorbidities.includes("Sepsis") || timelines.some(t => t.sepsis_active) ? (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber/5 border border-amber/20">
              <div className="w-2 h-2 rounded-full bg-amber animate-pulse-slow" />
              <div>
                <p className="text-[10px] font-bold text-amber">Sepsis Active</p>
                <p className="text-[9px] text-text-muted">Type 2 MI possible</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-safe/5 border border-safe/20">
              <div className="w-2 h-2 rounded-full bg-safe" />
              <div>
                <p className="text-[10px] font-bold text-safe">Sepsis Clear</p>
                <p className="text-[9px] text-text-muted">No demand ischemia</p>
              </div>
            </div>
          )}
        </div>

        {/* Comorbidities */}
        {comorbidities.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {comorbidities.map((c) => (
              <span key={c} className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] bg-elevated border border-border-default text-text-secondary">
                {c}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Troponin Chart */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-danger" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Troponin Velocity Curve</h3>
        </div>
        <TroponinChart timelines={timelines} activeTimestep={activeTimestep} />
      </div>
    </div>
  );
}
