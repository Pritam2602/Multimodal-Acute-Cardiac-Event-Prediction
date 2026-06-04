"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { MOCK_ADMISSIONS, Admission } from "@/lib/utils/mockData";
import { useAdmissionStore } from "@/lib/store/useAdmissionStore";
import Sidebar from "@/components/ui/Sidebar";
import PatientHeader from "@/components/PatientHeader";
import TemporalTimeline from "@/components/TemporalTimeline";
import ECGViewer from "@/components/ecg-viewer/ECGViewer";
import ClinicalValuesPanel from "@/components/clinical/ClinicalValuesPanel";
import AttentionPanel from "@/components/attention/AttentionPanel";
import PredictionModule from "@/components/PredictionModule";
import ExplainabilityPanel from "@/components/ExplainabilityPanel";

type Tab = "ecg" | "clinical" | "attention" | "explainability";

export default function AdmissionPage() {
  const params = useParams();
  const hadmId = decodeURIComponent(params.id as string);
  const { setSelectedAdmission, activeTimestep, setActiveTimestep } = useAdmissionStore();
  const [admission, setAdmission] = useState<Admission | null>(
    MOCK_ADMISSIONS.find(a => a.hadm_id === hadmId) ?? null
  );
  const [loading, setLoading] = useState(!admission);
  const [activeTab, setActiveTab] = useState<Tab>("ecg");

  useEffect(() => {
    // Try live DB first; fall back to mock data
    const fetchAdmission = async () => {
      try {
        const res = await fetch(`/api/admissions/${encodeURIComponent(hadmId)}`);
        if (res.ok) {
          const data = await res.json();
          setAdmission(data.admission);
        }
      } catch {
        // DB not available; keep mock if found
      } finally {
        setLoading(false);
      }
    };
    fetchAdmission();
  }, [hadmId]);

  useEffect(() => {
    if (admission) setSelectedAdmission(admission);
  }, [admission, setSelectedAdmission]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-base text-text-muted">
        <div className="text-center">
          <p className="text-xs font-mono animate-pulse">Loading admission {hadmId}...</p>
        </div>
      </div>
    );
  }

  if (!admission) {
    return (
      <div className="flex h-screen items-center justify-center bg-base text-text-muted">
        <div className="text-center">
          <p className="text-sm">Admission not found: {hadmId}</p>
          <a href="/dashboard" className="text-xs text-cyan mt-2 block">← Back to dashboard</a>
        </div>
      </div>
    );
  }

  const TABS: { id: Tab; label: string }[] = [
    { id: "ecg", label: "ECG Waveforms" },
    { id: "clinical", label: "Clinical Values" },
    { id: "attention", label: "Attention Maps" },
    { id: "explainability", label: "Explainability" },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-base">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Patient header */}
        <PatientHeader
          admission={admission}
          activeTimestep={activeTimestep}
          onTimestepChange={setActiveTimestep}
        />

        {/* Temporal Timeline */}
        <TemporalTimeline
          admission={admission}
          activeTimestep={activeTimestep}
          onTimestepChange={setActiveTimestep}
        />

        {/* Main content area */}
        <div className="flex-1 flex overflow-hidden min-h-0">
          {/* Left column: Tabbed content (ECG / Clinical / Attention / Explain) */}
          <div className="flex-1 flex flex-col overflow-hidden border-r border-border-default">
            {/* Tabs */}
            <div className="flex items-center border-b border-border-default px-4 bg-surface">
              {TABS.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`px-4 py-3 text-xs font-medium transition-all border-b-2 -mb-px ${
                    activeTab === id
                      ? "text-cyan border-cyan"
                      : "text-text-muted border-transparent hover:text-text-secondary"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="flex-1 overflow-hidden">
              {activeTab === "ecg" && (
                <ECGViewer
                  admission={admission}
                  activeTimestep={activeTimestep}
                  attentionWeights={admission.attention.leads}
                  onTimestepChange={setActiveTimestep}
                />
              )}
              {activeTab === "clinical" && (
                <div className="h-full overflow-y-auto p-4">
                  <ClinicalValuesPanel admission={admission} activeTimestep={activeTimestep} />
                </div>
              )}
              {activeTab === "attention" && (
                <div className="h-full overflow-y-auto p-4">
                  <AttentionPanel attention={admission.attention} activeTimestep={activeTimestep} />
                </div>
              )}
              {activeTab === "explainability" && (
                <div className="h-full overflow-y-auto p-4">
                  <ExplainabilityPanel admission={admission} />
                </div>
              )}
            </div>
          </div>

          {/* Right column: Prediction module (always visible) */}
          <div className="w-80 flex-shrink-0 overflow-y-auto p-4 bg-surface/50">
            <PredictionModule admission={admission} />
          </div>
        </div>
      </main>
    </div>
  );
}
