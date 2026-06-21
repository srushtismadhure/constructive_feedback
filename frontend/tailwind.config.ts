import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        regolith: "#FF6B00",
        "mars-burnt": "#B94300",
        "solar-amber": "#FFB000",
        "deep-space": "#07090B",
        graphite: "#11161B",
        panel: "#171C22",
        "panel-dark": "#0B0D0F",
        "off-white": "#F2F0EA",
        "muted-text": "#8B918F",
        "soft-gray": "#C6C7C2",
        "success-green": "#4ADE80",
        "warning-yellow": "#FACC15",
        mars: {
          dust: "#c1440e",
          rust: "#8b2500",
          sand: "#d4956a",
          dark: "#1a0a00",
          surface: "#2d1200",
          glow: "#ff6b35",
        },
        blueprint: {
          bg: "#0a1628",
          line: "#1e3a5f",
          accent: "#4a9eff",
          grid: "#0d2040",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "SFMono-Regular", "monospace"],
        sans: ["Inter", "Arial", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "scan": "scan 2s linear infinite",
        "flicker": "flicker 4s ease-in-out infinite",
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        flicker: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.85" },
        },
      },
      backgroundImage: {
        "blueprint-grid":
          "linear-gradient(rgba(30,58,95,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(30,58,95,0.4) 1px, transparent 1px)",
      },
      backgroundSize: {
        "blueprint-grid": "40px 40px",
      },
    },
  },
  plugins: [],
};

export default config;
