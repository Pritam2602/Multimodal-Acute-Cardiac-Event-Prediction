// ─────────────────────────────────────────────────────────────────────────────
// Zustand auth store — persisted to localStorage
// NO "use client" directive here — this is a plain module, not a React component
// skipHydration: true prevents the persist middleware from reading localStorage
// during SSR, which would cause a hook-count mismatch on hydration
// ─────────────────────────────────────────────────────────────────────────────
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { Doctor } from "./doctors";

interface AuthState {
  doctor: Doctor | null;
  isAuthenticated: boolean;
  login: (doctor: Doctor) => void;
  logout: () => void;
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      doctor: null,
      isAuthenticated: false,
      login: (doctor: Doctor) => set({ doctor, isAuthenticated: true }),
      logout: () => set({ doctor: null, isAuthenticated: false }),
    }),
    {
      name: "ami-auth-store",
      storage: createJSONStorage(() => localStorage),
      // CRITICAL: skip automatic rehydration on module load.
      // We manually call rehydrate() inside the client-only AuthGuard
      // after the component mounts. This prevents the server from ever
      // seeing a different auth state than the client's initial render.
      skipHydration: true,
      partialize: (state) => ({
        doctor: state.doctor,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
