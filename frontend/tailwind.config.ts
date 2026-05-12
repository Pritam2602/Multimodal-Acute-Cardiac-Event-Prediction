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
        base: "#060d1a",
        surface: "#0a1628",
        elevated: "#0e1d35",
        card: "#111f38",
        "border-subtle": "#162845",
        "border-default": "#1e3558",
        "border-bright": "#2a4d7a",
        "text-primary": "#e2eaf8",
        "text-secondary": "#8aafd4",
        "text-muted": "#4a6a94",
        cyan: {
          DEFAULT: "#00d4ff",
          dim: "#0099bb",
          glow: "rgba(0, 212, 255, 0.15)",
        },
        amber: {
          DEFAULT: "#f59e0b",
          glow: "rgba(245, 158, 11, 0.15)",
        },
        danger: {
          DEFAULT: "#f43f5e",
          dim: "#991b3a",
          glow: "rgba(244, 63, 94, 0.15)",
        },
        safe: {
          DEFAULT: "#10b981",
          dim: "#065f46",
          glow: "rgba(16, 185, 129, 0.15)",
        },
        purple: {
          DEFAULT: "#a855f7",
          glow: "rgba(168, 85, 247, 0.15)",
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
        cyan: "0 0 20px rgba(0, 212, 255, 0.25)",
        "cyan-sm": "0 0 10px rgba(0, 212, 255, 0.2)",
        amber: "0 0 20px rgba(245, 158, 11, 0.25)",
        danger: "0 0 20px rgba(244, 63, 94, 0.25)",
        safe: "0 0 20px rgba(16, 185, 129, 0.2)",
      },
    },
  },
  plugins: [],
};

export default config;
