import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}", "./lib/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: { lab: { bg: "#080a10", panel: "#0d111a", cyan: "#72caff", violet: "#8b7cff" } },
      fontFamily: { mono: ["DM Mono", "monospace"], sans: ["DM Sans", "sans-serif"] },
    },
  },
  plugins: [],
};

export default config;
