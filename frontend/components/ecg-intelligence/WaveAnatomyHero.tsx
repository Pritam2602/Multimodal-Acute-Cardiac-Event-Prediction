"use client";
// ─────────────────────────────────────────────────────────────────────────────
// WaveAnatomyHero — interactive annotated ECG SVG
// Click each labeled segment to expand a clinical explanation card
// ─────────────────────────────────────────────────────────────────────────────
import { useState } from "react";

interface WaveSegment {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  shortDesc: string;
  clinicalMeaning: string;
  amiSignal: string;
  normalRange: string;
}

const SEGMENTS: WaveSegment[] = [
  {
    id: "p-wave",
    label: "P Wave",
    x: 90,
    y: 52,
    color: "#22c55e",
    shortDesc: "Atrial depolarization",
    clinicalMeaning:
      "The P wave reflects electrical activation spreading across both atria before ventricular contraction. It's the first upward deflection in the cardiac cycle.",
    amiSignal:
      "PR interval prolongation (>200ms) may indicate first-degree AV block, common in inferior AMI. A missing or retrograde P wave can indicate junctional rhythm from ischemia.",
    normalRange: "Duration: 80–120ms · Amplitude: <0.25mV",
  },
  {
    id: "pr-interval",
    label: "PR Interval",
    x: 148,
    y: 78,
    color: "#a3e635",
    shortDesc: "AV conduction delay",
    clinicalMeaning:
      "The isoelectric line between P wave and QRS represents conduction through the AV node. The model uses `PR_interval` and `PR_prolonged` flag features.",
    amiSignal:
      "PR prolongation (>200ms) in inferior leads (II, III, aVF) is a hallmark of right coronary artery occlusion. The model's `PR_prolonged` binary flag directly feeds the classifier.",
    normalRange: "120–200ms (0.12–0.20 seconds)",
  },
  {
    id: "q-wave",
    label: "Q Wave",
    x: 196,
    y: 66,
    color: "#f59e0b",
    shortDesc: "Septal depolarization",
    clinicalMeaning:
      "A small negative deflection before the R wave. Pathological Q waves (deep >1/3 R height, wide >40ms) indicate necrotic myocardium — dead tissue that can no longer generate electrical signal.",
    amiSignal:
      "Pathological Q waves are the signature of completed (transmural) MI. The ResNet-18 backbone specifically learned to detect wide Q waves in leads V1-V4 (anterior STEMI) and II, III, aVF (inferior STEMI).",
    normalRange: "Normal: <40ms width, <25% of R amplitude",
  },
  {
    id: "r-wave",
    label: "R Wave",
    x: 234,
    y: 15,
    color: "#ef4444",
    shortDesc: "Ventricular depolarization peak",
    clinicalMeaning:
      "The tallest positive deflection — peak ventricular depolarization. Poor R-wave progression (R waves don't grow from V1 to V4) indicates anterior wall injury.",
    amiSignal:
      "R-wave height loss across V1→V4 leads is a critical AMI marker. The model's 1x1 spatial bottleneck (nn.Conv1d across 12 leads) specifically learns lead-combination patterns that detect R-wave progression failure.",
    normalRange: "V5/V6 peak: 5–25mm · R/S ratio >1 by V4",
  },
  {
    id: "s-wave",
    label: "S Wave",
    x: 268,
    y: 62,
    color: "#f97316",
    shortDesc: "Basal ventricular depolarization",
    clinicalMeaning:
      "The negative deflection after R wave — terminal ventricular depolarization. Prominent S waves in lateral leads with R in V1 suggest right ventricular hypertrophy or posterior MI.",
    amiSignal:
      "Deep S waves in I, aVL combined with ST elevation in right precordial leads (V1-V3) point to posterior STEMI — a pattern the model's multi-head attention detects by correlating lateral and septal lead features.",
    normalRange: "S/R ratio < 1 in left precordial leads (V5, V6)",
  },
  {
    id: "st-segment",
    label: "ST Segment",
    x: 310,
    y: 48,
    color: "#dc2626",
    shortDesc: "Ventricular plateau — KEY AMI marker",
    clinicalMeaning:
      "The isoelectric segment between QRS and T wave. Represents the plateau phase of the ventricular action potential. Elevation (STEMI) or depression (NSTEMI/ischemia) is the primary ECG diagnostic criterion for AMI.",
    amiSignal:
      "ST elevation ≥1mm in ≥2 contiguous leads = STEMI. The model's cross-attention uses clinical features (troponin, age) as QUERY to focus on ECG temporal regions (KEYS/VALUES). ST-segment deviations are the highest-weight attention regions in true AMI cases.",
    normalRange: "Isoelectric ±0.5mm · Elevation >1mm = pathological",
  },
  {
    id: "t-wave",
    label: "T Wave",
    x: 375,
    y: 30,
    color: "#8b5cf6",
    shortDesc: "Ventricular repolarization",
    clinicalMeaning:
      "Ventricular relaxation after contraction. T-wave inversions (flipping negative) in leads facing the ischemic zone indicate subendocardial injury or resolving STEMI.",
    amiSignal:
      "Hyperacute (giant, peaked) T waves are the very FIRST sign of AMI — appearing before ST elevation. T-wave inversion in precordial leads after STEMI indicates reperfusion or anterior ischemia. The `T_axis_abnormal` flag captures this deviation.",
    normalRange: "Upright in I, II, V3-V6 · Inverted normal in aVR, V1",
  },
  {
    id: "qt-interval",
    label: "QTc",
    x: 420,
    y: 75,
    color: "#06b6d4",
    shortDesc: "Ventricular action potential duration",
    clinicalMeaning:
      "Corrected QT interval (Bazett formula). Reflects the total duration of ventricular depolarization + repolarization. The model uses `QTc_prolonged` as an engineered binary feature.",
    amiSignal:
      "QTc prolongation (>450ms men, >470ms women) during AMI indicates electrical instability and risk of ventricular fibrillation. The `QTc_prolonged` flag is among the top-5 feature attributions in high-risk predictions.",
    normalRange: "Men: <450ms · Women: <470ms (Bazett corrected)",
  },
];

export default function WaveAnatomyHero() {
  const [active, setActive] = useState<WaveSegment | null>(SEGMENTS[5]); // ST default

  // ECG path — realistic PQRST morphology
  const ECG_PATH = `
    M 10 75
    L 55 75
    Q 70 74 80 65
    Q 90 56 95 68
    Q 100 78 108 75
    L 168 75
    L 180 70
    L 192 75
    L 200 28
    L 210 115
    L 218 68
    L 228 75
    L 300 75
    L 330 62
    Q 360 38 375 42
    Q 395 48 400 68
    Q 410 78 430 75
    L 490 75
  `;

  return (
    <div className="space-y-4">
      {/* SVG ECG viewer */}
      <div
        className="relative rounded-2xl overflow-hidden"
        style={{
          background: "#fffefb",
          border: "1px solid #e8d5b0",
          backgroundImage: [
            "linear-gradient(rgba(239,68,68,0.10) 1px, transparent 1px)",
            "linear-gradient(90deg, rgba(239,68,68,0.10) 1px, transparent 1px)",
            "linear-gradient(rgba(239,68,68,0.04) 1px, transparent 1px)",
            "linear-gradient(90deg, rgba(239,68,68,0.04) 1px, transparent 1px)",
          ].join(", "),
          backgroundSize: "50px 50px, 50px 50px, 10px 10px, 10px 10px",
        }}
      >
        <svg
          viewBox="0 0 500 130"
          className="w-full"
          style={{ height: "160px" }}
        >
          {/* ECG signal */}
          <path
            d={ECG_PATH}
            fill="none"
            stroke="#dc2626"
            strokeWidth="2.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {/* Segment annotations */}
          {SEGMENTS.map((seg) => (
            <g key={seg.id} onClick={() => setActive(active?.id === seg.id ? null : seg)}>
              {/* Vertical drop line */}
              <line
                x1={seg.x}
                y1={seg.y - 12}
                x2={seg.x}
                y2={seg.y - 22}
                stroke={seg.color}
                strokeWidth={active?.id === seg.id ? 2 : 1}
                strokeDasharray="3,2"
                opacity={active?.id === seg.id ? 1 : 0.6}
              />
              {/* Label bubble */}
              <rect
                x={seg.x - 22}
                y={seg.y - 42}
                width={44}
                height={16}
                rx={8}
                fill={active?.id === seg.id ? seg.color : "rgba(255,255,255,0.9)"}
                stroke={seg.color}
                strokeWidth={1}
                style={{ cursor: "pointer" }}
              />
              <text
                x={seg.x}
                y={seg.y - 31}
                textAnchor="middle"
                fontSize="7"
                fontWeight="700"
                fill={active?.id === seg.id ? "white" : seg.color}
                style={{ cursor: "pointer", fontFamily: "DM Sans, sans-serif" }}
              >
                {seg.label}
              </text>
            </g>
          ))}

          {/* ST segment highlight when active */}
          {active?.id === "st-segment" && (
            <rect x={280} y={68} width={55} height={8} rx={2}
              fill="rgba(220,38,38,0.15)" stroke="rgba(220,38,38,0.4)" strokeWidth={0.5} />
          )}
        </svg>

        <div className="absolute top-2 right-3">
          <span className="text-[9px] font-mono text-red-400/60 uppercase tracking-widest">Lead II · 25mm/s · 10mm/mV</span>
        </div>
      </div>

      {/* Detail card */}
      {active && (
        <div
          className="rounded-2xl p-5 transition-all"
          style={{
            background: `linear-gradient(135deg, ${active.color}10, ${active.color}05)`,
            border: `1px solid ${active.color}35`,
          }}
        >
          <div className="flex items-start justify-between mb-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: active.color }} />
                <h3 className="text-sm font-bold text-gray-900">{active.label}</h3>
                <span
                  className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                  style={{ background: `${active.color}15`, color: active.color }}
                >
                  {active.shortDesc}
                </span>
              </div>
              <p className="text-[10px] font-mono text-gray-500">{active.normalRange}</p>
            </div>
            <button onClick={() => setActive(null)} className="text-gray-400 hover:text-gray-600 text-xs">✕</button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-xl" style={{ background: "rgba(255,255,255,0.7)", border: "1px solid rgba(0,0,0,0.06)" }}>
              <p className="text-[9px] font-bold uppercase tracking-widest text-gray-400 mb-1.5">Clinical Meaning</p>
              <p className="text-xs text-gray-700 leading-relaxed">{active.clinicalMeaning}</p>
            </div>
            <div className="p-3 rounded-xl" style={{ background: "rgba(220,38,38,0.04)", border: "1px solid rgba(220,38,38,0.12)" }}>
              <p className="text-[9px] font-bold uppercase tracking-widest text-red-500 mb-1.5">🧠 AI AMI Signal</p>
              <p className="text-xs text-gray-700 leading-relaxed">{active.amiSignal}</p>
            </div>
          </div>
        </div>
      )}

      {!active && (
        <p className="text-center text-xs text-gray-400 py-2">
          ↑ Click any labeled segment above to see its clinical meaning and how the AI reads it
        </p>
      )}
    </div>
  );
}
