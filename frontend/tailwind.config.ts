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
        // Main layout backgrounds
        base: "#e8f5ed",       // light mint — page background
        surface: "#ffffff",    // white — cards, panels
        elevated: "#f0faf4",   // very light mint — inputs, chips
        card: "#ffffff",
        "border-subtle": "#d0e8d8",
        "border-default": "#b8d8c4",
        "border-bright": "#7fb898",

        // Text
        "text-primary": "#0d2b1a",
        "text-secondary": "#2d5a3d",
        "text-muted": "#6b9580",

        // Primary accent — medium forest green (buttons, links, active states)
        cyan: {
          DEFAULT: "#1a7a45",
          dim: "#145f35",
          glow: "rgba(26, 122, 69, 0.10)",
        },

        // Semantic
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
          DEFAULT: "#16a34a",
          dim: "#15803d",
          glow: "rgba(22, 163, 74, 0.08)",
        },
        purple: {
          DEFAULT: "#7c3aed",
          glow: "rgba(124, 58, 237, 0.08)",
        },
        orange: {
          400: "#f97316",
        },

        // Sidebar — deep forest green (standalone, never used as bg-base)
        sidebar: {
          DEFAULT: "#163829",
          surface: "#1c4a35",
          active: "#ffffff",
          text: "#c8e8d4",
          muted: "#7aab8a",
          border: "#1f5040",
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
        cyan: "0 2px 8px rgba(26, 122, 69, 0.18)",
        "cyan-sm": "0 1px 4px rgba(26, 122, 69, 0.14)",
        amber: "0 1px 4px rgba(217, 119, 6, 0.15)",
        danger: "0 1px 4px rgba(220, 38, 38, 0.15)",
        safe: "0 1px 4px rgba(22, 163, 74, 0.12)",
        card: "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        sidebar: "2px 0 12px rgba(0,0,0,0.12)",
      },
    },
  },
  plugins: [],
};

export default config;
