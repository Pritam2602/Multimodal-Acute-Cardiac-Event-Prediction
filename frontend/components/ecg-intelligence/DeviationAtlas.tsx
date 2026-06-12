"use client";
// ─────────────────────────────────────────────────────────────────────────────
// DeviationAtlas — Grid of AMI-specific ECG deviation cards
// Each card shows: deviation name, mini SVG sketch, model feature, severity
// ─────────────────────────────────────────────────────────────────────────────
import { useState } from "react";

interface Deviation {
  id: string;
  name: string;
  icdLink: string;
  severity: "critical" | "high" | "moderate";
  leads: string;
  modelFeature: string;
  description: string;
  earlyDetection: string;
  svgPath: string; // mini ECG sketch path
  svgViewBox: string;
  color: string;
}

const DEVIATIONS: Deviation[] = [
  {
    id: "st-elevation",
    name: "ST-Segment Elevation",
    icdLink: "STEMI — I21.0–I21.3",
    severity: "critical",
    leads: "V1–V6, II, III, aVF",
    modelFeature: "Cross-attention locks onto the ST plateau region. Clinical query (troponin) gates ECG keys at ST timepoints.",
    description:
      "ST elevation ≥1mm in ≥2 contiguous leads = STEMI. Indicates full-thickness (transmural) myocardial injury — typically from complete coronary occlusion. The injured myocardium cannot repolarize normally, shifting the ST segment above the isoelectric baseline.",
    earlyDetection:
      "Hyperacute T waves precede ST elevation by 15–30 minutes. Our model detects rising T-wave amplitude patterns that predict imminent STEMI even before classic criteria are met.",
    svgPath: "M0,30 L20,30 L25,28 L30,30 L33,8 L36,42 L39,22 L45,15 L60,15 L80,30",
    svgViewBox: "0 0 80 45",
    color: "#dc2626",
  },
  {
    id: "st-depression",
    name: "ST-Segment Depression",
    icdLink: "NSTEMI — I21.4",
    severity: "high",
    leads: "V4–V6, I, aVL",
    modelFeature: "The `QRS_axis_deviation` feature + ST depression pattern → subendocardial ischemia signature.",
    description:
      "ST depression ≥0.5mm in ≥2 contiguous leads indicates subendocardial ischemia (NSTEMI) or posterior STEMI (mirror image in V1-V3). The subendocardium is most vulnerable to ischemia as it is furthest from coronary supply.",
    earlyDetection:
      "Reciprocal ST depression in leads opposite to the infarct zone appears simultaneously with ST elevation elsewhere, allowing triangulation of the culprit vessel territory.",
    svgPath: "M0,15 L20,15 L25,13 L30,15 L33,2 L36,42 L39,20 L45,38 L60,38 L80,15",
    svgViewBox: "0 0 80 45",
    color: "#f97316",
  },
  {
    id: "pathological-q",
    name: "Pathological Q Wave",
    icdLink: "Old MI marker — I25.2",
    severity: "high",
    leads: "V1–V4 (anterior), II/III/aVF (inferior)",
    modelFeature: "ResNet-18 1D backbone — residual blocks 64→128→256→512ch — detects wide negative QRS deflections as key morphological signature.",
    description:
      "Q waves >40ms wide or >25% of R amplitude = pathological. They represent zones of electrically silent necrotic myocardium. The dead tissue cannot depolarize, so the electrical vector points away from it, creating a negative deflection in leads overlying the infarct.",
    earlyDetection:
      "Q waves appear within 1-2 hours of complete coronary occlusion. The model's ResNet-18 backbone is specifically sensitive to the width-to-amplitude ratio of the initial QRS deflection across all 12 leads simultaneously.",
    svgPath: "M0,22 L20,22 L25,20 L30,22 L32,35 L35,5 L38,40 L42,22 L80,22",
    svgViewBox: "0 0 80 45",
    color: "#f59e0b",
  },
  {
    id: "t-wave-inversion",
    name: "T-Wave Inversion",
    icdLink: "Ischemia marker",
    severity: "moderate",
    leads: "V1–V4, I, aVL (anterior), II/III/aVF (inferior)",
    modelFeature: "`T_axis_abnormal` binary flag engineered from ECG machine measurements. Attribution rank 3-5 in NSTEMI predictions.",
    description:
      "Inverted T waves in leads normally showing upright T waves indicate subendocardial ischemia or post-STEMI reperfusion. Wellens syndrome (deep symmetric T inversions in V2-V3) is a pre-infarction pattern indicating proximal LAD stenosis.",
    earlyDetection:
      "Wellens pattern T-wave inversions can precede complete LAD occlusion by hours to days. Early detection of this pattern is one of the highest-value clinical scenarios where our model's ECG branch provides unique value over troponin alone.",
    svgPath: "M0,20 L25,20 L28,18 L30,20 L33,5 L36,45 L39,18 L43,20 L50,20 L58,25 Q68,38 72,35 Q76,28 80,20",
    svgViewBox: "0 0 80 45",
    color: "#8b5cf6",
  },
  {
    id: "qrs-prolongation",
    name: "QRS Prolongation",
    icdLink: "Bundle branch block",
    severity: "moderate",
    leads: "All leads (global prolongation)",
    modelFeature: "`QRS_wide` binary flag (>120ms), `QRS_duration_invalid` flag for extreme values. QRS width changes alter the ResNet feature map timing.",
    description:
      "QRS duration >120ms indicates bundle branch block or aberrant conduction. LBBB (left bundle branch block) both mimics ST elevation AND can mask true STEMI (Sgarbossa criteria needed). RBBB with ST elevation in V1-V3 = Brugada pattern.",
    earlyDetection:
      "New-onset LBBB during chest pain is treated as STEMI-equivalent. The model's `QRS_wide` flag combined with ST features significantly elevates risk probability when new LBBB morphology is detected.",
    svgPath: "M0,22 L15,22 L18,20 L20,22 L22,32 L26,3 L32,45 L38,20 L44,22 L80,22",
    svgViewBox: "0 0 80 45",
    color: "#06b6d4",
  },
  {
    id: "pr-prolongation",
    name: "PR Interval Changes",
    icdLink: "AV conduction block",
    severity: "moderate",
    leads: "Best seen in II, V1",
    modelFeature: "`PR_prolonged` binary flag (>200ms), engineered from ECG machine measurements. Correlates with inferior AMI in model's feature attributions.",
    description:
      "PR interval >200ms = first-degree AV block. In the context of inferior STEMI (right coronary artery occlusion), PR prolongation indicates involvement of the AV nodal artery. Short PR with delta wave = Wolff-Parkinson-White (pre-excitation).",
    earlyDetection:
      "Progressive PR prolongation (Mobitz type I) in inferior leads during AMI signals progressive AV node ischemia, identifying patients at high risk for complete heart block who need immediate pacing consideration.",
    svgPath: "M0,22 L10,22 L14,16 L18,22 L50,22 L53,5 L56,45 L59,20 L65,22 L80,22",
    svgViewBox: "0 0 80 45",
    color: "#a3e635",
  },
];

const SEVERITY_CONFIG = {
  critical: { label: "Critical", color: "#dc2626", bg: "rgba(220,38,38,0.1)", border: "rgba(220,38,38,0.3)" },
  high: { label: "High", color: "#f97316", bg: "rgba(249,115,22,0.1)", border: "rgba(249,115,22,0.3)" },
  moderate: { label: "Moderate", color: "#f59e0b", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.3)" },
};

export default function DeviationAtlas() {
  const [expanded, setExpanded] = useState<string | null>("st-elevation");

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {DEVIATIONS.map((dev) => {
        const sev = SEVERITY_CONFIG[dev.severity];
        const isOpen = expanded === dev.id;

        return (
          <div
            key={dev.id}
            onClick={() => setExpanded(isOpen ? null : dev.id)}
            className="rounded-2xl overflow-hidden cursor-pointer transition-all"
            style={{
              background: isOpen ? "white" : "#fafafa",
              border: `1px solid ${isOpen ? dev.color + "50" : "#e5e7eb"}`,
              boxShadow: isOpen ? `0 4px 20px ${dev.color}15` : "0 1px 3px rgba(0,0,0,0.05)",
              transform: isOpen ? "scale(1.01)" : "scale(1)",
            }}
          >
            {/* Card header */}
            <div className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full"
                      style={{ background: sev.bg, color: sev.color, border: `1px solid ${sev.border}` }}
                    >
                      {sev.label}
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-gray-900">{dev.name}</h3>
                  <p className="text-[10px] text-gray-500 mt-0.5">{dev.icdLink}</p>
                </div>

                {/* Mini ECG sketch */}
                <div
                  className="w-20 h-12 rounded-lg flex-shrink-0 overflow-hidden"
                  style={{
                    background: "#fffefb",
                    backgroundImage: "linear-gradient(rgba(239,68,68,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(239,68,68,0.08) 1px, transparent 1px)",
                    backgroundSize: "10px 10px",
                    border: "1px solid #ffe4e4",
                  }}
                >
                  <svg viewBox={dev.svgViewBox} className="w-full h-full p-1">
                    <path d={dev.svgPath} fill="none" stroke={dev.color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              </div>

              <p className="text-[10px] font-mono text-gray-500">Leads: {dev.leads}</p>
            </div>

            {/* Expanded content */}
            {isOpen && (
              <div className="px-4 pb-4 space-y-3">
                <div
                  className="h-px w-full"
                  style={{ background: `linear-gradient(90deg, ${dev.color}40, transparent)` }}
                />

                <div className="p-3 rounded-xl" style={{ background: "#f9fafb", border: "1px solid #f0f0f0" }}>
                  <p className="text-[9px] font-bold uppercase tracking-widest text-gray-400 mb-1.5">Clinical Significance</p>
                  <p className="text-xs text-gray-700 leading-relaxed">{dev.description}</p>
                </div>

                <div className="p-3 rounded-xl" style={{ background: "rgba(220,38,38,0.04)", border: "1px solid rgba(220,38,38,0.1)" }}>
                  <p className="text-[9px] font-bold uppercase tracking-widest text-red-500 mb-1.5">🧠 Early AMI Detection</p>
                  <p className="text-xs text-gray-700 leading-relaxed">{dev.earlyDetection}</p>
                </div>

                <div className="p-3 rounded-xl" style={{ background: "rgba(22,56,41,0.04)", border: "1px solid rgba(22,56,41,0.1)" }}>
                  <p className="text-[9px] font-bold uppercase tracking-widest text-green-700 mb-1.5">⚙️ Model Feature</p>
                  <p className="text-xs text-gray-700 leading-relaxed font-mono">{dev.modelFeature}</p>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
