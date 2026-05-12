"use client";
import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Search, Filter, ChevronUp, ChevronDown, ChevronsUpDown, Droplets, Scan, ArrowRight, Database, AlertCircle, RefreshCw } from "lucide-react";
import { useAdmissionStore } from "@/lib/store/useAdmissionStore";
import { Admission, getPredictionColor } from "@/lib/utils/mockData";
import { RiskBadge, PredBadge, GroundTruthBadge, TimestepBadge } from "@/components/ui/Badge";

type SortKey = "hadm_id" | "age" | "max_troponin" | "ecg_count" | "risk_level" | "predicted_prob";
type SortDir = "asc" | "desc";

export default function AdmissionsTable() {
  const router = useRouter();
  const {
    searchQuery, riskFilter, predictionFilter,
    setSearchQuery, setRiskFilter, setPredictionFilter,
    setSelectedAdmission, getFilteredAdmissions,
    fetchAdmissions, isLoadingAdmissions, admissionsError, usingLiveData,
  } = useAdmissionStore();
  const [sortKey, setSortKey] = useState<SortKey>("predicted_prob");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const doFetch = useCallback(() => {
    fetchAdmissions({ search: searchQuery, risk: riskFilter, prediction: predictionFilter });
  }, [fetchAdmissions, searchQuery, riskFilter, predictionFilter]);

  // Fetch on mount to try live DB
  useEffect(() => { doFetch(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const admissions = getFilteredAdmissions();

  const sorted = [...admissions].sort((a, b) => {
    const getVal = (ad: Admission) => {
      if (sortKey === "predicted_prob") return ad.prediction.predicted_prob;
      if (sortKey === "max_troponin") return ad.max_troponin;
      if (sortKey === "age") return ad.age;
      if (sortKey === "ecg_count") return ad.ecg_count;
      if (sortKey === "risk_level") return ["Low", "Moderate", "High", "Critical"].indexOf(ad.risk_level);
      return ad.hadm_id;
    };
    const av = getVal(a), bv = getVal(b);
    if (av < bv) return sortDir === "asc" ? -1 : 1;
    if (av > bv) return sortDir === "asc" ? 1 : -1;
    return 0;
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
  };

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <ChevronsUpDown className="w-3 h-3 text-text-muted" />;
    return sortDir === "asc"
      ? <ChevronUp className="w-3 h-3 text-cyan" />
      : <ChevronDown className="w-3 h-3 text-cyan" />;
  };

  const handleRowClick = (admission: Admission) => {
    setSelectedAdmission(admission);
    router.push(`/admission/${admission.hadm_id}`);
  };

  return (
    <div className="flex flex-col h-full">
      {/* DB status banner */}
      {admissionsError && (
        <div className="flex items-center gap-2 px-4 py-2 bg-amber/5 border-b border-amber/20 text-amber text-[11px]">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>Database not connected — showing demo data. Set up PostgreSQL and update <code className="font-mono">.env.local</code>.</span>
        </div>
      )}
      {usingLiveData && (
        <div className="flex items-center gap-2 px-4 py-2 bg-safe/5 border-b border-safe/20 text-safe text-[11px]">
          <Database className="w-3.5 h-3.5" />
          Live PostgreSQL data
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3 p-4 border-b border-border-default flex-wrap">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" />
          <input
            type="text"
            placeholder="Search admission ID or patient ID..."
            value={searchQuery}
            onChange={e => { setSearchQuery(e.target.value); }}
            onKeyDown={e => e.key === "Enter" && doFetch()}
            className="w-full pl-9 pr-4 py-2 bg-elevated border border-border-default rounded-lg text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-cyan/50 focus:bg-surface"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-text-muted" />
          <select
            value={riskFilter}
            onChange={e => { setRiskFilter(e.target.value); fetchAdmissions({ search: searchQuery, risk: e.target.value, prediction: predictionFilter }); }}
            className="bg-elevated border border-border-default rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-cyan/50"
          >
            <option value="all">All Risk Levels</option>
            <option value="Critical">Critical</option>
            <option value="High">High</option>
            <option value="Moderate">Moderate</option>
            <option value="Low">Low</option>
          </select>

          <select
            value={predictionFilter}
            onChange={e => { setPredictionFilter(e.target.value); fetchAdmissions({ search: searchQuery, risk: riskFilter, prediction: e.target.value }); }}
            className="bg-elevated border border-border-default rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-cyan/50"
          >
            <option value="all">All Predictions</option>
            <option value="AMI">AMI Predicted</option>
            <option value="Non-AMI">Non-AMI Predicted</option>
          </select>

          <button
            onClick={doFetch}
            disabled={isLoadingAdmissions}
            className="flex items-center gap-1.5 px-3 py-2 text-[11px] bg-elevated border border-border-default text-text-muted rounded-lg hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw className={`w-3 h-3 ${isLoadingAdmissions ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        <div className="text-[11px] text-text-muted ml-auto font-mono">
          {sorted.length} of {admissions.length} admissions
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border-default bg-surface sticky top-0">
              {[
                { key: "hadm_id" as SortKey, label: "Admission ID", align: "text-left" },
                { key: "age" as SortKey, label: "Patient", align: "text-left" },
                { key: "max_troponin" as SortKey, label: "Max Troponin", align: "text-right" },
                { key: "ecg_count" as SortKey, label: "ECG Scans", align: "text-center" },
                { key: "risk_level" as SortKey, label: "Risk", align: "text-center" },
                { key: "predicted_prob" as SortKey, label: "AI Prediction", align: "text-right" },
              ].map(({ key, label, align }) => (
                <th
                  key={key}
                  className={`px-4 py-3 font-semibold text-[10px] uppercase tracking-widest text-text-muted cursor-pointer select-none ${align}`}
                  onClick={() => handleSort(key)}
                >
                  <div className={`flex items-center gap-1 ${align.includes("right") ? "justify-end" : align.includes("center") ? "justify-center" : ""}`}>
                    {label}
                    <SortIcon col={key} />
                  </div>
                </th>
              ))}
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest text-text-muted text-center font-semibold">Ground Truth</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-subtle">
            {sorted.map((admission) => {
              const prob = admission.prediction.predicted_prob;
              const probColor = getPredictionColor(prob);

              return (
                <tr
                  key={admission.hadm_id}
                  onClick={() => handleRowClick(admission)}
                  className="table-row-hover transition-colors group cursor-pointer"
                >
                  {/* Admission ID */}
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className={`w-1.5 h-5 rounded-full ${
                        prob >= 0.75 ? "bg-danger" :
                        prob >= 0.48 ? "bg-orange-400" :
                        prob >= 0.25 ? "bg-amber" : "bg-safe"
                      }`} />
                      <div>
                        <p className="font-mono font-semibold text-text-primary text-[11px]">{admission.hadm_id}</p>
                        <p className="text-[10px] text-text-muted">Admitted {admission.admittime.split(" ")[0]}</p>
                      </div>
                    </div>
                  </td>

                  {/* Patient */}
                  <td className="px-4 py-3.5">
                    <div>
                      <p className="font-mono font-semibold text-text-primary text-[11px]">{admission.subject_id}</p>
                      <p className="text-[10px] text-text-muted">{admission.age}y {admission.gender} · {admission.comorbidities.length > 0 ? admission.comorbidities[0] : "No comorbidities"}</p>
                    </div>
                  </td>

                  {/* Max Troponin */}
                  <td className="px-4 py-3.5 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <Droplets className="w-3 h-3 text-text-muted" />
                      <span className={`font-mono font-bold text-[11px] ${admission.max_troponin > 0.5 ? "text-danger" : admission.max_troponin > 0.2 ? "text-amber" : "text-safe"}`}>
                        {admission.max_troponin.toFixed(2)}
                      </span>
                      <span className="text-[9px] text-text-muted">ng/mL</span>
                    </div>
                  </td>

                  {/* ECG Scans */}
                  <td className="px-4 py-3.5">
                    <div className="flex flex-col items-center gap-1">
                      <div className="flex items-center gap-1">
                        <Scan className="w-3 h-3 text-text-muted" />
                        <span className="font-mono text-[11px] text-text-primary">{admission.ecg_count}</span>
                      </div>
                      <TimestepBadge count={admission.ecg_count} labels={["T₀", "T₁", "T₂"]} />
                    </div>
                  </td>

                  {/* Risk */}
                  <td className="px-4 py-3.5 text-center">
                    <RiskBadge level={admission.risk_level} />
                  </td>

                  {/* AI Prediction */}
                  <td className="px-4 py-3.5 text-right">
                    <div className="flex flex-col items-end gap-1">
                      <div className="flex items-center gap-1.5">
                        <span className={`font-mono font-bold text-sm ${probColor}`}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                        <PredBadge prob={prob} />
                      </div>
                      {/* Confidence bar */}
                      <div className="w-20 h-1 bg-border-subtle rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            prob >= 0.75 ? "bg-danger" :
                            prob >= 0.48 ? "bg-orange-400" :
                            prob >= 0.25 ? "bg-amber" : "bg-safe"
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>

                  {/* Ground Truth */}
                  <td className="px-4 py-3.5 text-center">
                    <GroundTruthBadge isAMI={admission.ground_truth_ami} />
                  </td>

                  {/* Arrow */}
                  <td className="px-3 py-3.5">
                    <ArrowRight className="w-4 h-4 text-text-muted opacity-0 group-hover:opacity-100 group-hover:text-cyan transition-all" />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {sorted.length === 0 && (
          <div className="text-center py-16 text-text-muted">
            <Search className="w-8 h-8 mx-auto mb-3 opacity-30" />
            <p className="text-sm">No admissions match your filters</p>
          </div>
        )}
      </div>
    </div>
  );
}
