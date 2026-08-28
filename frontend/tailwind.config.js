/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#08090c",
          900: "#0d0f14",
          850: "#12141b",
          800: "#181b24",
          700: "#232733",
          600: "#333847",
          500: "#4a5165",
          400: "#6b7186",
          300: "#9298a8",
          200: "#c1c5d0",
          100: "#e7e9ee",
        },
        accent: {
          DEFAULT: "#6d5ef8",
          light: "#9b8fff",
          dark: "#4c3fd6",
        },
        signal: {
          apply: "#3ddc97",
          tailor: "#f5b942",
          build: "#5fb0ff",
          low: "#6b7186",
          danger: "#ff5d6c",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(109,94,248,0.25), 0 8px 30px -8px rgba(109,94,248,0.35)",
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 12px 30px -14px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-fade":
          "linear-gradient(to bottom, transparent, rgba(8,9,12,1)), radial-gradient(ellipse at top, rgba(109,94,248,0.15), transparent 60%)",
      },
      animation: {
        "fade-up": "fadeUp 0.5s ease forwards",
        shimmer: "shimmer 2.4s linear infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};
