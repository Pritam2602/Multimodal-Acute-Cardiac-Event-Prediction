"use client";
import { AttentionWeights } from "@/lib/utils/mockData";

interface AnatomicalHeartProps {
  attention: AttentionWeights;
}

function regionScore(leads: Record<string, number>, region: string[]): number {
  const vals = region.map(l => leads[l] ?? 0);
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function toOpacity(score: number): number {
  return 0.1 + score * 0.85;
}

function toColor(score: number): string {
  if (score >= 0.75) return "#f43f5e";
  if (score >= 0.55) return "#f97316";
  if (score >= 0.35) return "#f59e0b";
  if (score >= 0.20) return "#00d4ff";
  return "#4a6a94";
}

export default function AnatomicalHeart({ attention }: AnatomicalHeartProps) {
  const inferiorScore = regionScore(attention.leads, ["II", "III", "aVF"]);
  const anteriorScore = regionScore(attention.leads, ["V1", "V2", "V3", "V4"]);
  const lateralScore = regionScore(attention.leads, ["I", "aVL", "V5", "V6"]);

  const infColor = toColor(inferiorScore);
  const antColor = toColor(anteriorScore);
  const latColor = toColor(lateralScore);

  const infOpacity = toOpacity(inferiorScore);
  const antOpacity = toOpacity(anteriorScore);
  const latOpacity = toOpacity(lateralScore);

  return (
    <div className="flex flex-col items-center gap-3">
      {/* SVG Heart Diagram */}
      <div className="relative">
        <svg viewBox="0 0 240 260" className="w-52 h-56" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Glow filters */}
          <defs>
            <filter id="inf-glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="ant-glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="lat-glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* Heart outline (simplified anatomical) */}
          {/* Base heart shape */}
          <path
            d="M120 228 C80 200 30 170 20 130 C10 90 30 55 65 48 C85 44 105 55 120 75 C135 55 155 44 175 48 C210 55 230 90 220 130 C210 170 160 200 120 228 Z"
            fill="#0a1628" stroke="#1e3558" strokeWidth="1.5"
          />

          {/* LV cavity outline */}
          <path
            d="M120 210 C95 190 65 165 58 138 C52 112 62 88 85 82 C100 78 115 88 120 105 C125 88 140 78 155 82 C178 88 188 112 182 138 C175 165 145 190 120 210 Z"
            fill="#060d1a" stroke="#162845" strokeWidth="1"
          />

          {/* INFERIOR WALL (bottom of LV) */}
          <path
            d="M120 210 C100 198 78 182 68 162 C80 168 100 180 120 195 C140 180 160 168 172 162 C162 182 140 198 120 210 Z"
            fill={infColor}
            opacity={infOpacity}
            filter="url(#inf-glow)"
          />

          {/* ANTERIOR WALL (front/septal) */}
          <path
            d="M85 82 C92 78 105 76 120 80 L120 105 C115 90 100 82 85 82 Z"
            fill={antColor}
            opacity={antOpacity}
            filter="url(#ant-glow)"
          />
          <path
            d="M120 80 C120 80 130 78 138 80 C150 84 160 92 162 105 C155 95 140 85 120 80 Z"
            fill={antColor}
            opacity={antOpacity * 0.7}
          />

          {/* LATERAL WALL */}
          <path
            d="M175 48 C200 60 218 88 218 120 C210 150 190 170 170 162 C180 145 185 125 182 105 C179 88 176 68 175 48 Z"
            fill={latColor}
            opacity={latOpacity}
            filter="url(#lat-glow)"
          />
          <path
            d="M22 120 C22 88 40 60 65 48 C64 68 61 88 58 105 C55 125 60 145 70 162 C50 170 30 150 22 120 Z"
            fill={latColor}
            opacity={latOpacity * 0.5}
          />

          {/* LA */}
          <ellipse cx="120" cy="45" rx="28" ry="18" fill="#0a1628" stroke="#1e3558" strokeWidth="1" />
          <text x="120" y="50" textAnchor="middle" fill="#4a6a94" fontSize="8" fontFamily="JetBrains Mono">LA</text>

          {/* RA */}
          <ellipse cx="185" cy="58" rx="20" ry="16" fill="#0a1628" stroke="#1e3558" strokeWidth="1" />
          <text x="185" y="63" textAnchor="middle" fill="#4a6a94" fontSize="8" fontFamily="JetBrains Mono">RA</text>

          {/* Aorta */}
          <path d="M110 35 Q105 15 120 8 Q135 15 130 35" stroke="#2a4d7a" strokeWidth="2.5" fill="none" />

          {/* Septum line */}
          <line x1="120" y1="80" x2="120" y2="195" stroke="#1e3558" strokeWidth="1" strokeDasharray="3 2" />

          {/* Region labels */}
          <text x="120" y="215" textAnchor="middle" fill={infColor} fontSize="7" fontFamily="JetBrains Mono" opacity={0.9}>INF</text>
          <text x="53" y="125" textAnchor="middle" fill={antColor} fontSize="7" fontFamily="JetBrains Mono" opacity={0.9} transform="rotate(-90, 53, 125)">ANT</text>
          <text x="195" y="108" textAnchor="middle" fill={latColor} fontSize="7" fontFamily="JetBrains Mono" opacity={0.9}>LAT</text>
        </svg>

        {/* Dominant region badge */}
        <div className="absolute top-0 right-0">
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded"
            style={{ background: `${toColor(Math.max(inferiorScore, anteriorScore, lateralScore))}20`, color: toColor(Math.max(inferiorScore, anteriorScore, lateralScore)), border: `1px solid ${toColor(Math.max(inferiorScore, anteriorScore, lateralScore))}40` }}>
            {attention.dominant_region}
          </span>
        </div>
      </div>

      {/* Region scores */}
      <div className="w-full space-y-2">
        {[
          { label: "Inferior (II, III, aVF)", score: inferiorScore, color: infColor, leads: ["II", "III", "aVF"] },
          { label: "Anterior/Septal (V1-V4)", score: anteriorScore, color: antColor, leads: ["V1", "V2", "V3", "V4"] },
          { label: "Lateral (I, aVL, V5, V6)", score: lateralScore, color: latColor, leads: ["I", "aVL", "V5", "V6"] },
        ].map(({ label, score, color, leads }) => (
          <div key={label}>
            <div className="flex justify-between mb-1">
              <span className="text-[10px] text-text-muted">{label}</span>
              <span className="text-[10px] font-mono font-bold" style={{ color }}>{(score * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-border-subtle rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${score * 100}%`, background: color, boxShadow: `0 0 6px ${color}80` }}
              />
            </div>
            <div className="flex gap-1 mt-1 flex-wrap">
              {leads.map(l => {
                const w = attention.leads[l] ?? 0;
                return (
                  <span key={l} className="text-[8px] font-mono px-1 py-0.5 rounded border" style={{
                    color: toColor(w), borderColor: `${toColor(w)}40`, background: `${toColor(w)}10`
                  }}>
                    {l}: {(w * 100).toFixed(0)}%
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
