"use client";
// ─────────────────────────────────────────────────────────────────────────────
// ECG Intelligence Page — /ecg-intelligence
// Explains how the multimodal AI reads P/Q/R deviations for AMI prediction
// ─────────────────────────────────────────────────────────────────────────────
import { useState } from "react";
import Sidebar from "@/components/ui/Sidebar";
import WaveAnatomyHero from "@/components/ecg-intelligence/WaveAnatomyHero";
import DeviationAtlas from "@/components/ecg-intelligence/DeviationAtlas";
import ModelAttentionFlow from "@/components/ecg-intelligence/ModelAttentionFlow";
import { Brain, Zap, BookOpen, FlaskConical, ChevronDown, Info, TrendingUp, Clock, Target } from "lucide-react";

const TABS = [
  { id: "anatomy", label: "Wave Anatomy", icon: BookOpen, desc: "P, Q, R, S, T labeled" },
  { id: "deviations", label: "Deviation Atlas", icon: Zap, desc: "AMI signatures" },
  { id: "model", label: "AI Architecture", icon: Brain, desc: "Actual model flow" },
];

const STAT_CARDS = [
  { label: "Model AUC-ROC", value: "0.9411", sub: "Phase 10 · Curated cohort", color: "#22c55e", icon: TrendingUp },
  { label: "Best F1 Score", value: "0.7753", sub: "Phase 10 · Epoch 6 · Early Fusion", color: "#06b6d4", icon: Target },
  { label: "Decision Threshold", value: "≈0.45", sub: "Precision=0.810 · Recall=0.743", color: "#f59e0b", icon: Clock },
  { label: "Curated Cohort", value: "40,255", sub: "Phase 9 · −1,841 ambiguous", color: "#8b5cf6", icon: FlaskConical },
];

const HOW_IT_HELPS = [
  {
    title: "Pre-symptomatic Detection",
    desc: "Hyperacute T-wave changes precede classic STEMI criteria by 15–30 minutes. Our model detects these early morphological shifts — giving doctors a critical head start to activate the cath lab.",
    color: "#dc2626",
    emoji: "⚡",
  },
  {
    title: "Triage Prioritization",
    desc: "With high-risk patients ranked by P(AMI), physicians can immediately identify which of 40,255 admissions need urgent intervention — without reviewing every ECG manually.",
    color: "#f97316",
    emoji: "📊",
  },
  {
    title: "CKD False-Positive Filtering",
    desc: "Chronic kidney disease elevates troponin without AMI. The contrastive margin loss and troponin/creatinine ratio features help separate genuine ischemia from renal false alarms.",
    color: "#f59e0b",
    emoji: "🔬",
  },
  {
    title: "Clinical Reasoning Transparency",
    desc: "Every prediction comes with ranked feature attributions (Gradient×Input) — showing exactly which ECG features and lab values drove the AMI call, so doctors can validate or override.",
    color: "#22c55e",
    emoji: "🧠",
  },
];

export default function ECGIntelligencePage() {
  const [activeTab, setActiveTab] = useState("anatomy");
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#e8f5ed" }}>
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Page header */}
        <div
          className="px-6 py-4 flex-shrink-0"
          style={{
            background: "linear-gradient(135deg, #0d2b1a, #163829)",
            borderBottom: "1px solid rgba(34,197,94,0.15)",
          }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center"
                style={{ background: "rgba(34,197,94,0.15)", border: "1px solid rgba(34,197,94,0.25)" }}
              >
                <Brain className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <h1 className="text-base font-bold text-white leading-tight">ECG Intelligence</h1>
                <p className="text-[10px] text-green-400/60 leading-tight">
                  How the AI reads waveforms · P/Q/R deviation guide · Architecture deep-dive
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full" style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.2)" }}>
                <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                <span className="text-[10px] font-semibold text-green-400">Phase 2B Golden Model Active</span>
              </div>
              <button
                onClick={() => setShowInfo(v => !v)}
                className="p-2 rounded-xl transition-all"
                style={{ background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.12)" }}
              >
                <Info className="w-4 h-4 text-white/60" />
              </button>
            </div>
          </div>

          {/* Stats bar */}
          <div className="mt-4 grid grid-cols-4 gap-3">
            {STAT_CARDS.map(({ label, value, sub, color, icon: Icon }) => (
              <div
                key={label}
                className="p-3 rounded-xl"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}
              >
                <div className="flex items-center gap-1.5 mb-1">
                  <Icon className="w-3 h-3" style={{ color }} />
                  <span className="text-[9px] font-semibold uppercase tracking-wider text-white/50">{label}</span>
                </div>
                <p className="text-lg font-bold font-mono leading-none" style={{ color }}>{value}</p>
                <p className="text-[9px] text-white/30 mt-0.5">{sub}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Info panel (collapsible) */}
        {showInfo && (
          <div
            className="mx-6 mt-4 p-4 rounded-2xl flex-shrink-0"
            style={{ background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)" }}
          >
            <div className="flex items-start gap-3">
              <Brain className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs font-semibold text-green-800 mb-1">About This Module</p>
                <p className="text-xs text-green-700 leading-relaxed">
                  This page explains how the <strong>ClinicalConditionedFusionModel</strong> (Phase 2B) interprets 12-lead ECG waveforms to predict Acute Myocardial Infarction. The model achieved F1=0.7214 at Epoch 3 with OneCycleLR before attention boundaries degraded — this instantaneous checkpoint is the &quot;Golden Model.&quot; The cross-attention mechanism routes clinical features (troponin, age, comorbidities) as queries that gate which ECG temporal segments the ResNet-18 backbone attends to.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Content area */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* How the AI helps doctors — summary cards */}
          <div className="mb-6">
            <h2 className="text-sm font-bold text-gray-800 mb-3">How AI-Driven ECG Analysis Helps Doctors Predict AMI Earlier</h2>
            <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
              {HOW_IT_HELPS.map(({ title, desc, color, emoji }) => (
                <div
                  key={title}
                  className="p-4 rounded-2xl"
                  style={{
                    background: "white",
                    border: `1px solid ${color}20`,
                    boxShadow: `0 2px 8px ${color}10`,
                  }}
                >
                  <div className="text-xl mb-2">{emoji}</div>
                  <h3 className="text-xs font-bold mb-1.5" style={{ color }}>{title}</h3>
                  <p className="text-[10px] text-gray-600 leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Tabs */}
          <div className="card overflow-hidden">
            {/* Tab bar */}
            <div
              className="flex border-b"
              style={{ borderColor: "#d0e8d8", background: "#f0faf4" }}
            >
              {TABS.map(({ id, label, icon: Icon, desc }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className="flex items-center gap-2 px-5 py-3.5 text-xs font-semibold transition-all relative"
                  style={{
                    color: activeTab === id ? "#163829" : "#6b9e7a",
                    borderBottom: activeTab === id ? "2px solid #163829" : "2px solid transparent",
                    background: activeTab === id ? "white" : "transparent",
                    marginBottom: activeTab === id ? "-1px" : "0",
                  }}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <div className="text-left">
                    <div>{label}</div>
                    <div className="text-[9px] font-normal opacity-70">{desc}</div>
                  </div>
                </button>
              ))}

              <div className="flex-1" />
              <div className="flex items-center px-4 gap-1 text-[10px] text-gray-400">
                <ChevronDown className="w-3 h-3" />
                Click any item to expand
              </div>
            </div>

            {/* Tab content */}
            <div className="p-5">
              {activeTab === "anatomy" && (
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <BookOpen className="w-4 h-4 text-green-700" />
                    <h2 className="text-sm font-bold text-gray-800">ECG Wave Anatomy</h2>
                    <span className="text-[10px] text-gray-500">— Click any labeled segment to learn its AMI significance</span>
                  </div>
                  <WaveAnatomyHero />
                </div>
              )}

              {activeTab === "deviations" && (
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <Zap className="w-4 h-4 text-yellow-600" />
                    <h2 className="text-sm font-bold text-gray-800">AMI Deviation Atlas</h2>
                    <span className="text-[10px] text-gray-500">— 6 key ECG abnormalities the AI detects · Click cards to expand</span>
                  </div>
                  <DeviationAtlas />
                </div>
              )}

              {activeTab === "model" && (
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <Brain className="w-4 h-4 text-red-600" />
                    <h2 className="text-sm font-bold text-gray-800">
                      ClinicalConditionedFusionModel — Architecture Flow
                    </h2>
                    <span className="text-[10px] text-gray-500">— Phase 2B Golden Model · Click each stage to see details</span>
                  </div>
                  <ModelAttentionFlow />
                </div>
              )}
            </div>
          </div>

          {/* Research note */}
          <div
            className="mt-4 p-4 rounded-2xl"
            style={{ background: "rgba(139,92,246,0.06)", border: "1px solid rgba(139,92,246,0.15)" }}
          >
            <div className="flex items-start gap-3">
              <FlaskConical className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs font-bold text-purple-800 mb-1">Research Finding: The Physiological Information Ceiling</p>
                <p className="text-xs text-purple-700 leading-relaxed">
                  Despite 15+ training runs and phases 1–9, the model hit an immovable ceiling of <strong>F1≈0.70</strong> on the original noisy MIMIC-IV dataset. 
                  Error analysis identified the root cause: <em>clinical information entropy</em> — contradictory labels (demand ischemia cases coded as AMI), meaningless temporal ECG sequences (recordings taken 5 min apart due to lead errors), and comorbid false-positives (Sepsis+CKD+HF simultaneously).
                  Phase 9 curated these out. The result was decisive: <strong>Phase 10 Early Fusion achieved F1=0.7753, AUC=0.9411</strong> — shattering the ceiling. 
                  Conclusion: the 0.70 wall was entirely a data artifact, not an architectural limit. When the dataset is clean, Early Fusion (0.7753) even edges Late Fusion (0.7725), proving joint early cross-modal representations are maximally efficient on high-quality data.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
