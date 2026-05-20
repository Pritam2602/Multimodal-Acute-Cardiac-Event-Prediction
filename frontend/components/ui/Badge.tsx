import { RiskLevel } from "@/lib/utils/mockData";

interface RiskBadgeProps {
  level: RiskLevel;
  className?: string;
}

export function RiskBadge({ level, className = "" }: RiskBadgeProps) {
  const styles: Record<RiskLevel, string> = {
    Critical: "bg-red-50 text-red-700 border border-red-200",
    High:     "bg-orange-50 text-orange-700 border border-orange-200",
    Moderate: "bg-amber-50 text-amber-700 border border-amber-200",
    Low:      "bg-emerald-50 text-emerald-700 border border-emerald-200",
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

export function PredBadge({ prob, threshold = 0.6494 }: PredBadgeProps) {
  const isAMI = prob >= threshold;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-widest ${
      isAMI
        ? "bg-red-50 text-red-700 border border-red-200"
        : "bg-emerald-50 text-emerald-700 border border-emerald-200"
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${isAMI ? "bg-red-500 animate-pulse-slow" : "bg-emerald-500"}`} />
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
        ? "bg-red-50 text-red-600 border-red-200"
        : "bg-emerald-50 text-emerald-600 border-emerald-200"
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
