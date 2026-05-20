"use client";
import { create } from "zustand";
import { Admission, MOCK_ADMISSIONS } from "@/lib/utils/mockData";

interface AdmissionState {
  admissions: Admission[];
  selectedAdmission: Admission | null;
  activeTimestep: number;
  isRunningAnalysis: boolean;
  analysisComplete: boolean;
  searchQuery: string;
  riskFilter: string;
  predictionFilter: string;
  compareIds: [string | null, string | null];
  // DB fetch state
  isLoadingAdmissions: boolean;
  admissionsError: string | null;
  totalAdmissions: number;
  usingLiveData: boolean;

  setSelectedAdmission: (a: Admission | null) => void;
  setActiveTimestep: (t: number) => void;
  runAnalysis: () => Promise<void>;
  setSearchQuery: (q: string) => void;
  setRiskFilter: (r: string) => void;
  setPredictionFilter: (p: string) => void;
  setCompareId: (slot: 0 | 1, id: string | null) => void;
  getFilteredAdmissions: () => Admission[];
  // Fetch from API (PostgreSQL)
  fetchAdmissions: (params?: { search?: string; risk?: string; prediction?: string }) => Promise<void>;
}

export const useAdmissionStore = create<AdmissionState>((set, get) => ({
  admissions: MOCK_ADMISSIONS,
  selectedAdmission: null,
  activeTimestep: 0,
  isRunningAnalysis: false,
  analysisComplete: false,
  searchQuery: "",
  riskFilter: "all",
  predictionFilter: "all",
  compareIds: [null, null],
  isLoadingAdmissions: false,
  admissionsError: null,
  totalAdmissions: MOCK_ADMISSIONS.length,
  usingLiveData: false,

  setSelectedAdmission: (a) => set({ selectedAdmission: a, activeTimestep: 0, analysisComplete: false }),
  setActiveTimestep: (t) => set({ activeTimestep: t }),

  runAnalysis: async () => {
    set({ isRunningAnalysis: true, analysisComplete: false });
    await new Promise((r) => setTimeout(r, 1800));
    set({ isRunningAnalysis: false, analysisComplete: true });
  },

  setSearchQuery: (q) => set({ searchQuery: q }),
  setRiskFilter: (r) => set({ riskFilter: r }),
  setPredictionFilter: (p) => set({ predictionFilter: p }),
  setCompareId: (slot, id) =>
    set((s) => {
      const ids: [string | null, string | null] = [...s.compareIds];
      ids[slot] = id;
      return { compareIds: ids };
    }),

  fetchAdmissions: async (params = {}) => {
    set({ isLoadingAdmissions: true, admissionsError: null });
    try {
      const qs = new URLSearchParams({
        limit: "10000",
        ...(params.search ? { search: params.search } : {}),
        ...(params.risk && params.risk !== "all" ? { risk: params.risk } : {}),
        ...(params.prediction && params.prediction !== "all" ? { prediction: params.prediction } : {}),
      });
      const res = await fetch(`/api/admissions?${qs}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      set({
        admissions: data.admissions,
        totalAdmissions: data.total,
        isLoadingAdmissions: false,
        usingLiveData: true,
      });
    } catch (err) {
      // Fall back to mock data if DB is not connected
      set({
        admissionsError: String(err),
        isLoadingAdmissions: false,
        admissions: MOCK_ADMISSIONS,
        usingLiveData: false,
      });
    }
  },

  getFilteredAdmissions: () => {
    const { admissions, searchQuery, riskFilter, predictionFilter, usingLiveData } = get();
    // When using live data, filtering is done server-side; just return all
    if (usingLiveData) return admissions;
    return admissions.filter((a) => {
      const matchSearch =
        !searchQuery ||
        a.hadm_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        a.subject_id.toLowerCase().includes(searchQuery.toLowerCase());
      const matchRisk = riskFilter === "all" || a.risk_level === riskFilter;
      const matchPred =
        predictionFilter === "all" ||
        (predictionFilter === "ami" && a.prediction.predicted_prob >= 0.6494) ||
        (predictionFilter === "non-ami" && a.prediction.predicted_prob < 0.6494);
      return matchSearch && matchRisk && matchPred;
    });
  },
}));
