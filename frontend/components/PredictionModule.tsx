"use client";
import { useState } from "react";
import { Cpu, Zap, Activity, TrendingUp, BarChart3, CheckCircle2, Loader2 } from "lucide-react";
import { Admission } from "@/lib/utils/mockData";
import { useAdmissionStore } from "@/lib/store/useAdmissionStore";
import ConfidenceGauge from "@/components/ui/ConfidenceGauge";

interface PredictionModuleProps {
  admission: Admission;
}

export default function PredictionModule({ admission }: PredictionModuleProps) {
  const { isRunningAnalysis, analysisComplete, runAnalysis } = useAdmissionStore();
  const [hasRun, setHasRun] = useState(false);

  const pred = admission.prediction;
  const isAMI = pred.predicted_prob >= pred.threshold;

  const handleRun = async () => {
    setHasRun(true);
    await runAnalysis();
  };

  const showResults = hasRun ? analysisComplete : true;

  return (
    <div className="space-y-4">
      {/* Run Analysis Button */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-cyan" />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">AI Inference Engine</h3>
        </div>

        <button
          onClick={handleRun}
          disabled={isRunningAnalysis}
          className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all ${
            isRunningAnalysis
              ? "bg-cyan/5 border border-cyan/20 text-text-muted cursor-not-allowed"
              : "bg-cyan/10 border border-cyan/40 text-cyan hover:bg-cyan/15 hover:border-cyan/60 hover:shadow-cyan"
          }`}
        >
          {isRunningAnalysis ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running Early Fusion Analysis...
            </>
          ) : analysisComplete ? (
            <>
              <CheckCircle2 className="w-4 h-4" />
              Re-run AI Analysis
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Run AI Analysis
            </>
          )}
        </button>

        {/* Model metadata */}
        <div className="flex justify-between mt-3">
          <span className="text-[10px] text-text-muted">Model</span>
          <span className="text-[10px] font-mono text-text-secondary">LeadAwareHybridExtractor (Early Fusion)</span>
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-text-muted">Threshold</span>
          <span className="text-[10px] font-mono text-amber">{pred.threshold} (Youden&apos;s J)</span>
        </div>
      </div>

      {/* Prediction Output */}
      {showResults && (
        <>
          {/* Confidence Gauge */}
          <div className="card p-4 flex flex-col items-center">
            <div className="flex items-center gap-2 mb-3 self-start">
              <Activity className="w-4 h-4 text-purple" />
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Prediction Output</h3>
            </div>

            <ConfidenceGauge probability={pred.predicted_prob} size={180} />

            <div className={`mt-2 flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold ${
              isAMI
                ? "bg-danger/10 border border-danger/40 text-danger"
                : "bg-safe/10 border border-safe/40 text-safe"
            }`}>
              <span className={`w-2 h-2 rounded-full ${isAMI ? "bg-danger animate-pulse-slow" : "bg-safe"}`} />
              {isAMI ? "AMI PREDICTED" : "NON-AMI PREDICTED"}
            </div>
          </div>

          {/* Modality Breakdown */}
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="w-4 h-4 text-amber" />
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Modality Contribution</h3>
            </div>

            <div className="space-y-3">
              {/* ECG */}
              <div>
                <div className="flex justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-cyan" />
                    <span className="text-[11px] text-text-secondary">ECG Morphological Trajectory</span>
                  </div>
                  <span className="text-[11px] font-mono font-bold text-cyan">{(pred.ecg_contribution * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-border-subtle rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan rounded-full transition-all duration-1000"
                    style={{ width: `${pred.ecg_contribution * 100}%` }}
                  />
                </div>
              </div>

              {/* Troponin */}
              <div>
                <div className="flex justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-danger" />
                    <span className="text-[11px] text-text-secondary">Troponin Velocity (Clinical)</span>
                  </div>
                  <span className="text-[11px] font-mono font-bold text-danger">{(pred.trop_contribution * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-border-subtle rounded-full overflow-hidden">
                  <div
                    className="h-full bg-danger rounded-full transition-all duration-1000"
                    style={{ width: `${pred.trop_contribution * 100}%` }}
                  />
                </div>
              </div>

              {/* Dominant modality badge */}
              <div className="flex items-center justify-between pt-1 border-t border-border-subtle">
                <span className="text-[10px] text-text-muted">Dominant Modality</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                  pred.dominant_modality === "ECG"
                    ? "bg-cyan/10 text-cyan border-cyan/30"
                    : pred.dominant_modality === "Troponin"
                    ? "bg-danger/10 text-danger border-danger/30"
                    : "bg-purple/10 text-purple border-purple/30"
                }`}>
                  {pred.dominant_modality}
                </span>
              </div>
            </div>
          </div>

          {/* Attention Metrics */}
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-text-secondary" />
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Attention Diagnostics</h3>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-elevated">
                <div>
                  <p className="text-[11px] text-text-secondary">Temporal Entropy</p>
                  <p className="text-[9px] text-text-muted">Low = concentrated attention (confident)</p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-mono font-bold ${pred.attn_temp_entropy < 0.35 ? "text-safe" : pred.attn_temp_entropy > 0.65 ? "text-danger" : "text-amber"}`}>
                    {pred.attn_temp_entropy.toFixed(3)}
                  </p>
                  <p className="text-[9px] text-text-muted">{pred.attn_temp_entropy < 0.35 ? "Focused" : pred.attn_temp_entropy > 0.65 ? "Uncertain" : "Moderate"}</p>
                </div>
              </div>
              <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-elevated">
                <div>
                  <p className="text-[11px] text-text-secondary">Spatial Dominance</p>
                  <p className="text-[9px] text-text-muted">High = strong regional focus</p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-mono font-bold ${pred.attn_spatial_dominance > 0.65 ? "text-danger" : pred.attn_spatial_dominance < 0.30 ? "text-safe" : "text-amber"}`}>
                    {pred.attn_spatial_dominance.toFixed(3)}
                  </p>
                  <p className="text-[9px] text-text-muted">{pred.attn_spatial_dominance > 0.65 ? "High focal" : pred.attn_spatial_dominance < 0.30 ? "Diffuse" : "Moderate"}</p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
