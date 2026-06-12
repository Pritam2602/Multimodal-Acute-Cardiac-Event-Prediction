"use client";
// ─────────────────────────────────────────────────────────────────────────────
// ModelAttentionFlow — Actual Phase 10 architecture
//
// Phase 10 = Phase 7 (ClinicalConditionedFusionModel / Early Fusion) architecture
// trained on Phase 9 curated dataset (N=40,255 after removing 1,841 ambiguous)
// Result: F1=0.7753 · AUC=0.9411 · Precision=0.810 · Recall=0.743 · τ≈0.45
//
// Architecture: ECG → Conv1d bottleneck → Lead Region Grouping → Inter-Region
// Attention → ResNet-18 1D → [K,V]; Clinical 62-feat → MLP → [Q];
// 4-Head CrossAttention → Fusion → Focal Loss MLP → P(AMI)
// ─────────────────────────────────────────────────────────────────────────────
import { useState } from "react";

interface FlowNode {
  id: string;
  title: string;
  subtitle: string;
  detail: string;
  icon: string;
  color: string;
  badge?: string;
  metrics?: string;
}

const FLOW_NODES: FlowNode[] = [
  {
    id: "ecg-input",
    title: "Raw ECG Input",
    subtitle: "Tensor: (B, 12, 5000)",
    detail:
      "12-lead ECG waveform sampled at 500 Hz for 10 seconds = 5,000 time-points per lead. Stored in NumPy memmap for zero-copy batch loading. Preprocessing: (1) Butterworth bandpass 0.5–40 Hz for baseline wander removal, (2) per-lead amplitude normalization to unit variance. Training augmentations: random lead dropout (30% chance), temporal cutout (20% chance, 100–500 samples), sinusoidal baseline wander (25% chance).",
    icon: "📡",
    color: "#22c55e",
    metrics: "500 Hz · 10 sec · 12 leads · N=40,255",
  },
  {
    id: "spatial-bottleneck",
    title: "1×1 Spatial Bottleneck",
    subtitle: "nn.Conv1d(in=12, out=12, kernel_size=1)",
    detail:
      "A pointwise convolution applied across the lead dimension before any temporal processing. Each output 'lead' is a learned linear combination of all 12 input leads. This allows the network to create virtual composite leads tuned for AMI detection — for example, automatically isolating the V1-V4 anterior ischemic pattern or the II/III/aVF inferior grouping — rather than treating each lead independently. The parameters of this layer are the most interpretable in the entire network.",
    icon: "🔀",
    color: "#06b6d4",
    metrics: "nn.Conv1d(12→12, k=1) · 144 trainable params",
  },
  {
    id: "region-grouping",
    title: "Anatomical Lead Region Grouping",
    subtitle: "4 cardiac walls → Intra-Region Pooling",
    badge: "Phase 7",
    detail:
      "ECG leads are grouped by their anatomical projection onto the cardiac wall:\n• Anterior/Septal: V1, V2, V3, V4 (LAD territory)\n• Inferior: II, III, aVF (RCA territory)\n• Lateral: I, aVL, V5, V6 (LCx territory)\n• Septal/Right: aVR, V1 (reciprocal)\n\nWithin each group, lead embeddings are averaged (Intra-Region Pooling) to create 4 compact Region Embeddings. This injects anatomical cardiac topology as a structural prior — the model can now reason about which cardiac wall is under ischemic stress rather than treating 12 independent signals.",
    icon: "🫀",
    color: "#f59e0b",
    metrics: "4 anatomical walls · LAD · RCA · LCx territories",
  },
  {
    id: "inter-region-attn",
    title: "Inter-Region Attention",
    subtitle: "Self-attention over 4 Region Embeddings",
    badge: "Phase 7",
    detail:
      "A secondary self-attention mechanism that evaluates the 4 Region Embeddings against each other. This allows the model to detect spatially contiguous ischemia — a key AMI diagnostic criterion. For example, if both the Anterior and Septal regions show elevated activation, this inter-region correlation strongly suggests LAD occlusion. Random multi-lead noise, by contrast, would activate non-contiguous regions. This layer specifically addresses the false-positive problem where noise in isolated leads was being mis-classified as AMI.",
    icon: "🔍",
    color: "#f59e0b",
    metrics: "Self-attn · 4 regions → contiguous ischemia detection",
  },
  {
    id: "resnet18-backbone",
    title: "ResNet-18 1D Temporal Encoder",
    subtitle: "8 residual blocks · 64→128→256→512 channels",
    detail:
      "Standard ResNet-18 adapted for 1D temporal signals. Architecture: 4 stages × 2 residual blocks each. Channel progression: 64→128→256→512. Each residual block: Conv1d → BN → ReLU → Conv1d → BN → skip connection → ReLU. Stride-2 downsampling at the start of stages 2,3,4 reduces the temporal dimension: 5,000→2,500→1,250→625→313 positions. Total ~10.2M parameters. The skip connections preserve fine-grained morphological features (QRS width, ST-slope angle, T-wave curvature) across depth.\n\nOutput shape: (B, 512, 313) — serves as the KEY and VALUE matrices in the cross-attention.",
    icon: "🏗️",
    color: "#8b5cf6",
    metrics: "~10.2M params · (B,512,313) → K, V matrices",
  },
  {
    id: "clinical-branch",
    title: "Clinical Feature Branch",
    subtitle: "62 features → LayerNorm MLP → 256-dim Query",
    detail:
      "62 engineered clinical features grouped by category:\n\n• Troponin dynamics: troponin_first_24h, troponin_max_24h, troponin_delta_24h, trop_fold_rise*, trop_peak_baseline_ratio* (*Phase 9 additions — encode the RISE trajectory, not just the absolute level)\n• Renal context: Creatinine_max_24h, troponin_creatinine_ratio (separates AMI from CKD false-positives)\n• ECG machine measurements: PR_interval, QRS_duration, QTc (Bazett), P/QRS/T axes, RR_interval\n• Binary abnormality flags: QRS_wide (>120ms), QTc_prolonged, PR_prolonged, QRS_axis_deviation, T_axis_abnormal\n• ECG timing: hours_admit_to_ecg, ecg_within_first_6h, ecg_within_first_24h\n• Vitals: HR_from_RR_interval, HR_RR_disagreement\n• Comorbidity flags: CKD, Heart Failure, Sepsis markers\n\nArchitecture: Linear(62→128) → LayerNorm → ReLU → Linear(128→256) → LayerNorm\nOutput: 256-dim Clinical Embedding — serves as the QUERY matrix in cross-attention.",
    icon: "🧪",
    color: "#a3e635",
    metrics: "62 features · trop_fold_rise (Phase 9) · → 256-dim Q",
  },
  {
    id: "cross-attention",
    title: "4-Head Cross-Attention",
    subtitle: "Q = Clinical (256-dim) · K, V = ECG (512-dim × 313)",
    badge: "Core Innovation",
    detail:
      "The central architectural innovation. The Clinical Embedding acts as the QUERY and the ECG temporal features act as the KEYS and VALUES:\n\nAttention(Q, K, V) = softmax(QKᵀ / √d_k) · V\n\nWith 4 heads (d_k=128 each), the model learns 4 independent clinical-ECG relevance patterns in parallel:\n• Head 1: Troponin trajectory → focuses on ST-segment temporal regions\n• Head 2: Age/Sex/HR → focuses on QRS morphology and rate\n• Head 3: PR/QRS/QTc intervals → focuses on conduction timing windows\n• Head 4: Comorbidity flags (CKD/HF/Sepsis) → modulates global rhythm attention\n\nLayerNorm before attention (Pre-LN) + residual skip connection prevents attention entropy collapse.\n\nAttention Entropy Penalty (λ=0.1): L_entropy = λ × mean(H(attention_weights)). This prevents the model from diffusing attention uniformly across all 313 time positions (which would be equivalent to no attention). It forces the model to localize to specific ECG segments relevant to the clinical context.",
    icon: "🎯",
    color: "#dc2626",
    metrics: "4 heads · d_k=128 · entropy penalty λ=0.1",
  },
  {
    id: "fusion",
    title: "Early Fusion & Concatenation",
    subtitle: "Concat[attn_out ⊕ ecg_pool ⊕ clinical_emb] → 1,280-dim",
    detail:
      "Three streams are concatenated into a single rich representation:\n1. Cross-attention output (512-dim) — clinically-gated ECG context\n2. Global average-pooled ECG features (512-dim) — holistic ECG summary\n3. Clinical embedding (256-dim) — direct clinical feature representation\n\nTotal: 1,280-dim fusion vector.\n\nPhase 10 finding: Early Fusion (F1=0.7753) slightly outperforms Late Fusion (F1=0.7725) on the Phase 9 curated dataset. This confirms that when the dataset is structurally clean (no contradictory labels, no meaningless temporal sequences), joint cross-modal co-adaptation in Early Fusion is more efficient than the protected-gradient strategy of Late Fusion. Late Fusion's cross-modal gating acts as a guard against gradient pollution from noisy data — but becomes unnecessary on clean data.",
    icon: "🔗",
    color: "#f97316",
    metrics: "512 + 512 + 256 = 1,280-dim · Early > Late on clean data",
  },
  {
    id: "classifier",
    title: "MLP Classifier Head",
    subtitle: "1280 → 512 → 128 → 1 · Focal Loss γ=2",
    detail:
      "Three-layer MLP with Dropout(0.3) between layers:\nLinear(1280→512) → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1)\n\nLoss: Focal Loss (γ=2) handles the 11.53% AMI prevalence in the curated cohort. Focal Loss down-weights easy negatives, forcing the model to focus on hard boundary cases.\n\nScheduler: OneCycleLR (per-batch stepping, 10% warmup → peak LR 3e-4 → cosine decay). Converges faster than CosineAnnealingLR, reached best F1 at Epoch 6.\n\nThreshold optimization: searched over [0.30, 0.90] in 200 equal steps on the held-out validation set. Phase 10 optimal threshold: ~0.45 — lower than earlier phases (~0.65–0.70) because the Phase 9 curation cleaned the class boundary, reducing overlap between the AMI and non-AMI probability distributions.\n\nAugmentation during training: label smoothing ε=0.05 (target 0.95/0.05 instead of 1/0).",
    icon: "⚖️",
    color: "#06b6d4",
    metrics: "1280→512→128→1 · Focal γ=2 · OneCycleLR · τ≈0.45",
  },
  {
    id: "output",
    title: "AMI Risk Score — P(AMI)",
    subtitle: "σ(logit) ∈ [0, 1] · Phase 10 Curated Early Fusion",
    badge: "Final Output",
    detail:
      "Final output: σ(logit) = calibrated AMI probability.\n\n📊 Phase 10 Performance (Early Fusion · Phase 9 Curated Dataset):\n• F1 Score: 0.7753 (Epoch 6) — shattered the 0.70 plateau\n• AUC-ROC: 0.9411\n• Precision: 0.810 · Recall: 0.743\n• Decision Threshold: ≈0.45\n• Cohort: 40,255 admissions\n\n🔬 The Scientific Breakthrough:\nPhase 1–9 models were hard-capped at F1≈0.70 across ALL architectures (286K to 10.2M parameters). Phase 9 dataset curation removed 1,841 ambiguous admissions: weak positives (AMI=1 but max troponin <0.04), weak negatives (AMI=0 but troponin >0.5 without confounders), ECG duplicates recorded <30 min apart (technician corrections), and AMI=0 patients with Sepsis+CKD+HF simultaneously. Result: F1 jumped from 0.72 → 0.7753 immediately. The 0.70 ceiling was entirely a data artifact — not architectural capacity.",
    icon: "🫀",
    color: "#dc2626",
    metrics: "F1=0.7753 · AUC=0.9411 · P=0.810 · R=0.743 · τ≈0.45",
  },
];

export default function ModelAttentionFlow() {
  const [activeNode, setActiveNode] = useState<string | null>("cross-attention");

  return (
    <div className="space-y-4">
      {/* Phase label */}
      <div className="flex items-center gap-3 mb-2">
        <div
          className="px-3 py-1.5 rounded-xl text-xs font-bold"
          style={{ background: "rgba(220,38,38,0.1)", color: "#dc2626", border: "1px solid rgba(220,38,38,0.2)" }}
        >
          Phase 10 · Early Fusion on Phase 9 Curated Dataset
        </div>
        <div
          className="px-3 py-1.5 rounded-xl text-xs font-semibold"
          style={{ background: "rgba(34,197,94,0.1)", color: "#22c55e", border: "1px solid rgba(34,197,94,0.2)" }}
        >
          F1 = 0.7753 · AUC = 0.9411
        </div>
        <span className="text-[10px] text-gray-400">Click each stage for architecture details</span>
      </div>

      {/* Flow pipeline */}
      <div className="flex flex-col gap-1.5">
        {FLOW_NODES.map((node, idx) => {
          const isActive = activeNode === node.id;
          const isCore = node.id === "cross-attention";
          const isOutput = node.id === "output";

          return (
            <div key={node.id} className="flex items-stretch gap-3">
              {/* Connector + icon column */}
              <div className="flex flex-col items-center w-9 flex-shrink-0">
                {idx > 0 && (
                  <div
                    className="w-0.5 flex-1 min-h-[8px]"
                    style={{ background: `linear-gradient(180deg, ${FLOW_NODES[idx - 1].color}60, ${node.color}60)` }}
                  />
                )}
                <div
                  className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 z-10 transition-all"
                  style={{
                    background: isActive ? node.color : `${node.color}18`,
                    border: `2px solid ${node.color}${isActive ? "ff" : "50"}`,
                    fontSize: "15px",
                    boxShadow: isActive ? `0 0 12px ${node.color}40` : "none",
                  }}
                >
                  {node.icon}
                </div>
                {idx < FLOW_NODES.length - 1 && (
                  <div
                    className="w-0.5 flex-1 min-h-[8px]"
                    style={{ background: `linear-gradient(180deg, ${node.color}60, ${FLOW_NODES[idx + 1].color}60)` }}
                  />
                )}
              </div>

              {/* Node card */}
              <div
                className="flex-1 rounded-xl overflow-hidden cursor-pointer transition-all mb-1"
                onClick={() => setActiveNode(isActive ? null : node.id)}
                style={{
                  background: isActive
                    ? isCore
                      ? "linear-gradient(135deg, rgba(220,38,38,0.07), rgba(139,92,246,0.04))"
                      : `${node.color}06`
                    : "rgba(255,255,255,0.55)",
                  border: `1px solid ${isActive ? node.color + "45" : "#e5e7eb"}`,
                  boxShadow: isActive
                    ? isCore
                      ? "0 4px 20px rgba(220,38,38,0.12)"
                      : `0 2px 12px ${node.color}12`
                    : "none",
                }}
              >
                {/* Header row */}
                <div className="flex items-center justify-between px-4 py-3">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div>
                      <div className="flex items-center gap-2 flex-wrap">
                        <span
                          className="text-sm font-bold leading-tight"
                          style={{ color: isActive ? node.color : "#111827" }}
                        >
                          {node.title}
                        </span>
                        {node.badge && (
                          <span
                            className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-full flex-shrink-0"
                            style={
                              node.badge === "Core Innovation"
                                ? { background: "rgba(220,38,38,0.12)", color: "#dc2626", border: "1px solid rgba(220,38,38,0.2)" }
                                : node.badge === "Final Output"
                                ? { background: "rgba(34,197,94,0.12)", color: "#16a34a", border: "1px solid rgba(34,197,94,0.2)" }
                                : { background: `${node.color}15`, color: node.color, border: `1px solid ${node.color}30` }
                            }
                          >
                            {node.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-[10px] font-mono text-gray-500 mt-0.5 leading-tight">{node.subtitle}</p>
                    </div>
                  </div>

                  {node.metrics && (
                    <span
                      className="text-[9px] font-mono px-2 py-1 rounded-lg flex-shrink-0 ml-3 text-right leading-relaxed"
                      style={{ background: `${node.color}10`, color: node.color }}
                    >
                      {node.metrics}
                    </span>
                  )}
                </div>

                {/* Expanded detail */}
                {isActive && (
                  <div
                    className="px-4 pb-4"
                  >
                    <div
                      className="p-3 rounded-xl text-xs text-gray-700 leading-relaxed whitespace-pre-line"
                      style={{ background: "rgba(255,255,255,0.85)", border: "1px solid rgba(0,0,0,0.06)" }}
                    >
                      {node.detail}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Architecture formula */}
      <div
        className="rounded-2xl p-4 mt-2"
        style={{ background: "rgba(15,30,20,0.06)", border: "1px solid rgba(22,56,41,0.14)" }}
      >
        <p className="text-[9px] font-bold uppercase tracking-widest text-green-700 mb-2">
          Phase 10 — Complete Forward Pass
        </p>
        <p className="text-[10px] font-mono text-gray-600 leading-loose">
          ECG(B,12,5000) →{" "}
          <span className="text-cyan-600">Conv1d(12→12,k=1)</span> →{" "}
          <span className="text-yellow-600">LeadGroups(4 walls)</span> →{" "}
          <span className="text-yellow-600">IntraRegionPool</span> →{" "}
          <span className="text-yellow-600">InterRegionAttn</span>
          <br />
          → <span className="text-purple-600">ResNet18_1D(64→128→256→512)</span> → ECG_feats(B,512,313) → <strong>[K, V]</strong>
          <br />
          Clinical(B,62) → <span className="text-green-600">MLP(62→128→256)</span> + LayerNorm → Clinical_emb(B,256) → <strong>[Q]</strong>
          <br />
          CrossAttn(<strong className="text-red-600">Q</strong>=Clinical,{" "}
          <strong className="text-red-600">K,V</strong>=ECG, heads=4, λ_entropy=0.1) → Attended(B,512)
          <br />
          Concat[Attended ‖ GlobalPool(ECG_feats) ‖ Clinical_emb] → Fused(B,<strong>1280</strong>)
          <br />
          MLP(1280→512→128→1) + <span className="text-orange-600">FocalLoss(γ=2)</span> + <span className="text-blue-600">OneCycleLR</span> →{" "}
          <strong className="text-red-600">P(AMI)</strong>{" "}
          <span className="text-green-700 font-bold">→ F1=0.7753 · AUC=0.9411 · τ≈0.45</span>
        </p>
      </div>
    </div>
  );
}
