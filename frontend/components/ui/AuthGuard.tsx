"use client";
// ─────────────────────────────────────────────────────────────────────────────
// AuthGuard — wraps all protected routes.
//
// WHY THE MOUNTED PATTERN:
// Next.js App Router renders layout.tsx on the server first, then hydrates on
// the client. During SSR, localStorage doesn't exist, so the Zustand persist
// store starts with isAuthenticated=false. Without the mounted guard, the
// server renders "not authenticated" and the client immediately sees
// "authenticated" — React detects different hook execution paths between the
// two renders and throws "rendered more hooks than during previous render."
//
// Solution: render nothing (null) on the server pass. On client mount, we
// trigger rehydrate() to pull auth state from localStorage, then decide
// whether to redirect or show content. This guarantees the server and the
// first client paint have identical output (null), eliminating the mismatch.
// ─────────────────────────────────────────────────────────────────────────────
import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth/useAuth";

const PUBLIC_ROUTES = ["/login"];

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  // --- ALL hooks must be called unconditionally, in the same order, every render ---
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated } = useAuth();
  const [mounted, setMounted] = useState(false);

  // Step 1: On first client mount, trigger Zustand persist rehydration from
  // localStorage (since we set skipHydration: true in the store).
  useEffect(() => {
    useAuth.persist.rehydrate();
    setMounted(true);
  }, []);

  // Step 2: Once mounted (and after rehydration), check auth state.
  useEffect(() => {
    if (!mounted) return;
    const isPublic = PUBLIC_ROUTES.some((r) => pathname.startsWith(r));
    if (!isAuthenticated && !isPublic) {
      router.replace("/login");
    }
  }, [mounted, isAuthenticated, pathname, router]);

  // During SSR and the initial hydration paint: render nothing.
  // Server and client agree: both output null → no hook mismatch.
  if (!mounted) return null;

  // After mount: if not authenticated and not on a public route, show nothing
  // (the redirect effect above will fire in the same tick).
  const isPublic = PUBLIC_ROUTES.some((r) => pathname.startsWith(r));
  if (!isAuthenticated && !isPublic) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "#0d1f14" }}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-green-500/30 border-t-green-400 rounded-full animate-spin" />
          <p className="text-xs text-green-400/60 font-mono tracking-widest">AUTHENTICATING</p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
