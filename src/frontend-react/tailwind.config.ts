import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/app/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#f7f7f5",
        foreground: "#111111",
        muted: "#e8e8e5",
        card: "#ffffff",
        border: "#d5d5d2",
      },
      borderRadius: {
        lg: "0.75rem",
        md: "0.5rem",
        sm: "0.375rem",
      },
      fontFamily: {
        serif: ["Iowan Old Style", "Baskerville", "Times New Roman", "serif"],
        sans: ["Avenir Next", "Helvetica Neue", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        museum: "0 10px 35px rgba(0, 0, 0, 0.1)",
      },
    },
  },
  plugins: [],
};

export default config;
