import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        base: "#f8fafc",
        surface: "#ffffff",
        elevated: "#f1f5f9",
        card: "#f8fafc",
        "border-subtle": "#e2e8f0",
        "border-default": "#cbd5e1",
        "border-bright": "#94a3b8",
        "text-primary": "#0f172a",
        "text-secondary": "#334155",
        "text-muted": "#64748b",
        cyan: {
          DEFAULT: "#0284c7",
          dim: "#0369a1",
          glow: "rgba(2, 132, 199, 0.08)",
        },
        amber: {
          DEFAULT: "#d97706",
          glow: "rgba(217, 119, 6, 0.08)",
        },
        danger: {
          DEFAULT: "#dc2626",
          dim: "#b91c1c",
          glow: "rgba(220, 38, 38, 0.08)",
        },
        safe: {
          DEFAULT: "#059669",
          dim: "#047857",
          glow: "rgba(5, 150, 105, 0.08)",
        },
        purple: {
          DEFAULT: "#7c3aed",
          glow: "rgba(124, 58, 237, 0.08)",
        },
        orange: {
          400: "#f97316",
        },
      },
      fontFamily: {
        sans: ["var(--font-dm-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        shimmer: "shimmer 2s linear infinite",
        "fade-in": "fadeIn 0.4s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      boxShadow: {
        cyan: "0 1px 4px rgba(2, 132, 199, 0.15)",
        "cyan-sm": "0 1px 3px rgba(2, 132, 199, 0.12)",
        amber: "0 1px 4px rgba(217, 119, 6, 0.15)",
        danger: "0 1px 4px rgba(220, 38, 38, 0.15)",
        safe: "0 1px 4px rgba(5, 150, 105, 0.12)",
        card: "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
      },
    },
  },
  plugins: [],
};

export default config;
