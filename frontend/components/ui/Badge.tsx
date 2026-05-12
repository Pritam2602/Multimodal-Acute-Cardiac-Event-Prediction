import { RiskLevel } from "@/lib/utils/mockData";

interface RiskBadgeProps {
  level: RiskLevel;
  className?: string;
}

export function RiskBadge({ level, className = "" }: RiskBadgeProps) {
  const styles: Record<RiskLevel, string> = {
    Critical: "bg-red-950 text-red-400 border border-red-800",
    High:     "bg-orange-950 text-orange-400 border border-orange-800",
    Moderate: "bg-yellow-950 text-yellow-400 border border-yellow-800",
    Low:      "bg-emerald-950 text-emerald-400 border border-emerald-800",
  };

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-widest ${styles[level]} ${className}`}>
      {level}
    </span>
  );
}

interface PredBadgeProps {
  prob: number;
  threshold?: number;
}

export function PredBadge({ prob, threshold = 0.48 }: PredBadgeProps) {
  const isAMI = prob >= threshold;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-widest ${
      isAMI
        ? "bg-red-950 text-red-400 border border-red-800"
        : "bg-emerald-950 text-emerald-400 border border-emerald-800"
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${isAMI ? "bg-red-400 animate-pulse-slow" : "bg-emerald-400"}`} />
      {isAMI ? "AMI" : "Non-AMI"}
    </span>
  );
}

interface GroundTruthBadgeProps {
  isAMI: boolean;
}

export function GroundTruthBadge({ isAMI }: GroundTruthBadgeProps) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
      isAMI
        ? "bg-red-950/50 text-red-300 border-red-900"
        : "bg-emerald-950/50 text-emerald-300 border-emerald-900"
    }`}>
      GT: {isAMI ? "AMI" : "Non-AMI"}
    </span>
  );
}

interface TimestepBadgeProps {
  count: number;
  labels: string[];
}

export function TimestepBadge({ count, labels }: TimestepBadgeProps) {
  return (
    <div className="flex items-center gap-1">
      {labels.map((label, i) => (
        <span
          key={i}
          className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border ${
            i < count
              ? "bg-cyan/10 text-cyan border-cyan/30"
              : "bg-border-subtle text-text-muted border-border-subtle"
          }`}
        >
          {label}
        </span>
      ))}
    </div>
  );
}
