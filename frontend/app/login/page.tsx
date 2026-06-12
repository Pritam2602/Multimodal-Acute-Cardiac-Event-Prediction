"use client";
// ─────────────────────────────────────────────────────────────────────────────
// Login Page — AMI Prediction Platform
// Glassmorphism design with animated ECG background
// SHA-256 client-side credential verification
// ─────────────────────────────────────────────────────────────────────────────
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Activity, Heart, Lock, Mail, Eye, EyeOff, AlertCircle, Cpu, Shield } from "lucide-react";
import { verifyCredentials } from "@/lib/auth/doctors";
import { useAuth } from "@/lib/auth/useAuth";

// ── Animated ECG SVG path generator ─────────────────────────────────────────
function generateECGPath(width: number, height: number): string {
  const mid = height / 2;
  const unit = width / 20;
  const segments: string[] = [`M 0 ${mid}`];

  for (let i = 0; i < 4; i++) {
    const x = i * (width / 4);
    segments.push(
      `L ${x + unit * 0.5} ${mid}`,
      `L ${x + unit * 0.8} ${mid - height * 0.08}`,
      `L ${x + unit * 1.0} ${mid}`,
      `L ${x + unit * 1.1} ${mid + height * 0.05}`,
      `L ${x + unit * 1.3} ${mid - height * 0.45}`,
      `L ${x + unit * 1.5} ${mid + height * 0.25}`,
      `L ${x + unit * 1.7} ${mid - height * 0.12}`,
      `L ${x + unit * 1.9} ${mid + height * 0.06}`,
      `L ${x + unit * 2.1} ${mid}`,
      `L ${x + unit * 2.5} ${mid}`
    );
  }
  return segments.join(" ");
}

export default function LoginPage() {
  const router = useRouter();
  const { login, isAuthenticated } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shake, setShake] = useState(false);
  const [ecgOffset, setEcgOffset] = useState(0);
  const animFrameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) router.replace("/dashboard");
  }, [isAuthenticated, router]);

  // Animate ECG scroll
  useEffect(() => {
    const animate = (time: number) => {
      if (time - lastTimeRef.current > 16) {
        setEcgOffset((prev) => (prev + 0.5) % 800);
        lastTimeRef.current = time;
      }
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animFrameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError("Please enter your credentials.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const doctor = await verifyCredentials(email.trim(), password);
      if (doctor) {
        login(doctor);
        router.replace("/dashboard");
      } else {
        setError("Invalid email or password. Please try again.");
        setShake(true);
        setTimeout(() => setShake(false), 600);
      }
    } catch {
      setError("Authentication error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const ecgPath = generateECGPath(800, 100);

  return (
    <div
      className="min-h-screen flex items-center justify-center relative overflow-hidden"
      style={{ background: "linear-gradient(135deg, #091a0f 0%, #0d2b1a 50%, #0a1f12 100%)" }}
    >
      {/* Animated ECG background strips */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[0, 1, 2, 3].map((i) => (
          <svg
            key={i}
            className="absolute w-full"
            style={{
              top: `${15 + i * 22}%`,
              opacity: 0.04 + i * 0.015,
              height: "100px",
            }}
            viewBox="0 0 800 100"
            preserveAspectRatio="none"
          >
            <path
              d={ecgPath}
              fill="none"
              stroke="#22c55e"
              strokeWidth="1.5"
              strokeDasharray="1600"
              strokeDashoffset={-ecgOffset - i * 200}
            />
          </svg>
        ))}
      </div>

      {/* Radial glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: "radial-gradient(ellipse 60% 50% at 50% 50%, rgba(34,197,94,0.06) 0%, transparent 70%)",
        }}
      />

      {/* Grid overlay */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(34,197,94,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.03) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* Main card */}
      <div
        className={`relative w-full max-w-md mx-4 transition-all ${shake ? "animate-shake" : ""}`}
        style={{
          background: "rgba(13, 30, 19, 0.85)",
          border: "1px solid rgba(34,197,94,0.2)",
          borderRadius: "24px",
          backdropFilter: "blur(24px)",
          boxShadow: "0 0 60px rgba(34,197,94,0.08), 0 40px 80px rgba(0,0,0,0.4)",
        }}
      >
        {/* Top accent line */}
        <div
          className="absolute top-0 left-8 right-8 h-px"
          style={{ background: "linear-gradient(90deg, transparent, rgba(34,197,94,0.5), transparent)" }}
        />

        <div className="p-8">
          {/* Logo header */}
          <div className="flex flex-col items-center mb-8">
            <div className="relative mb-4">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{
                  background: "linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,56,41,0.4))",
                  border: "1px solid rgba(34,197,94,0.3)",
                  boxShadow: "0 0 30px rgba(34,197,94,0.15)",
                }}
              >
                <Activity className="w-8 h-8 text-green-400" />
              </div>
              <div
                className="absolute -top-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center"
                style={{ background: "rgba(34,197,94,0.2)", border: "1px solid rgba(34,197,94,0.4)" }}
              >
                <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              </div>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">AMI Predict</h1>
            <p className="text-xs mt-1 font-medium tracking-widest uppercase" style={{ color: "rgba(134,239,172,0.6)" }}>
              Temporal Multimodal Cardiac AI
            </p>

            {/* Institution badge */}
            <div
              className="mt-3 flex items-center gap-1.5 px-3 py-1 rounded-full"
              style={{ background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.15)" }}
            >
              <Shield className="w-3 h-3 text-green-400/70" />
              <span className="text-[10px] font-semibold text-green-400/70 tracking-widest uppercase">
                MIMIC-IV Clinical Research Platform
              </span>
            </div>
          </div>

          {/* Section header */}
          <div className="mb-6">
            <h2 className="text-sm font-semibold text-white/80">Physician Sign In</h2>
            <p className="text-xs mt-0.5" style={{ color: "rgba(134,239,172,0.4)" }}>
              Authorized clinical personnel only
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email field */}
            <div>
              <label className="block text-xs font-semibold mb-1.5" style={{ color: "rgba(134,239,172,0.7)" }}>
                Physician Email
              </label>
              <div className="relative">
                <Mail
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
                  style={{ color: "rgba(134,239,172,0.4)" }}
                />
                <input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="dr.name@cardiac.ai"
                  className="w-full pl-10 pr-4 py-3 text-sm rounded-xl outline-none transition-all"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: `1px solid ${error ? "rgba(239,68,68,0.4)" : "rgba(34,197,94,0.15)"}`,
                    color: "rgba(255,255,255,0.9)",
                    fontFamily: "var(--font-dm-sans)",
                  }}
                  onFocus={(e) => (e.currentTarget.style.border = "1px solid rgba(34,197,94,0.45)")}
                  onBlur={(e) => (e.currentTarget.style.border = `1px solid ${error ? "rgba(239,68,68,0.4)" : "rgba(34,197,94,0.15)"}`)}
                />
              </div>
            </div>

            {/* Password field */}
            <div>
              <label className="block text-xs font-semibold mb-1.5" style={{ color: "rgba(134,239,172,0.7)" }}>
                Password
              </label>
              <div className="relative">
                <Lock
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
                  style={{ color: "rgba(134,239,172,0.4)" }}
                />
                <input
                  id="login-password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••"
                  className="w-full pl-10 pr-10 py-3 text-sm rounded-xl outline-none transition-all"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: `1px solid ${error ? "rgba(239,68,68,0.4)" : "rgba(34,197,94,0.15)"}`,
                    color: "rgba(255,255,255,0.9)",
                    fontFamily: "var(--font-dm-sans)",
                  }}
                  onFocus={(e) => (e.currentTarget.style.border = "1px solid rgba(34,197,94,0.45)")}
                  onBlur={(e) => (e.currentTarget.style.border = `1px solid ${error ? "rgba(239,68,68,0.4)" : "rgba(34,197,94,0.15)"}`)}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5"
                  style={{ color: "rgba(134,239,172,0.4)" }}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Error message */}
            {error && (
              <div
                className="flex items-center gap-2 px-3 py-2.5 rounded-xl"
                style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)" }}
              >
                <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                <p className="text-xs text-red-400">{error}</p>
              </div>
            )}

            {/* Submit */}
            <button
              id="login-submit"
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl text-sm font-bold transition-all relative overflow-hidden group"
              style={{
                background: loading
                  ? "rgba(34,197,94,0.2)"
                  : "linear-gradient(135deg, #166534, #15803d)",
                border: "1px solid rgba(34,197,94,0.4)",
                color: loading ? "rgba(134,239,172,0.5)" : "white",
                boxShadow: loading ? "none" : "0 4px 20px rgba(34,197,94,0.2)",
              }}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-green-400/30 border-t-green-400 rounded-full animate-spin" />
                  Verifying credentials…
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Heart className="w-4 h-4" />
                  Access Clinical Dashboard
                </span>
              )}
            </button>
          </form>

          {/* Demo credentials hint */}
          <div
            className="mt-5 p-3 rounded-xl"
            style={{ background: "rgba(34,197,94,0.04)", border: "1px solid rgba(34,197,94,0.1)" }}
          >
            <div className="flex items-center gap-1.5 mb-2">
              <Cpu className="w-3 h-3 text-green-400/50" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-green-400/50">
                Demo Credentials
              </span>
            </div>
            <div className="space-y-1">
              {[
                { email: "admin@cardiac.ai", pass: "Admin@123", label: "Admin" },
                { email: "dr.sharma@cardiac.ai", pass: "CardioAI@2025", label: "Attending" },
              ].map(({ email: e, pass, label }) => (
                <button
                  key={e}
                  type="button"
                  onClick={() => { setEmail(e); setPassword(pass); setError(null); }}
                  className="w-full text-left px-2.5 py-1.5 rounded-lg text-[10px] font-mono transition-all"
                  style={{
                    color: "rgba(134,239,172,0.5)",
                    background: "transparent",
                    border: "1px solid transparent",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(34,197,94,0.06)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <span className="text-green-400/30 mr-2">[{label}]</span>
                  {e}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom footer */}
        <div
          className="px-8 py-4 text-center"
          style={{ borderTop: "1px solid rgba(34,197,94,0.08)" }}
        >
          <p className="text-[9px] text-green-400/30 font-mono tracking-widest uppercase">
            MIMIC-IV · N=125,757 · SHA-256 Secured · Research Use Only
          </p>
        </div>
      </div>

      {/* Shake animation */}
      <style>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          20% { transform: translateX(-8px); }
          40% { transform: translateX(8px); }
          60% { transform: translateX(-5px); }
          80% { transform: translateX(5px); }
        }
        .animate-shake { animation: shake 0.5s ease-in-out; }
      `}</style>
    </div>
  );
}
