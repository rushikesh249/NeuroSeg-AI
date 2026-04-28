/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          deep: "#F8FAFC", // slate-50
          base: "#FFFFFF",
          surface: "rgba(255, 255, 255, 0.7)",
          elevated: "#F1F5F9", // slate-100
          hover: "#E2E8F0", // slate-200
        },
        accent: {
          blue: "#2563EB",
          lightBlue: "#DBEAFE",
          cyan: "#0EA5E9",
          magenta: "#D946EF",
          violet: "#8B5CF6",
          green: "#10B981",
          rose: "#F43F5E",
        },
        tumor: {
          et: "#10B981", // Green
          net: "#0EA5E9", // Cyan
          ed: "#D946EF", // Magenta
        }
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
      boxShadow: {
        'soft': '0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 10px -2px rgba(0, 0, 0, 0.03)',
        'apple': '0 10px 30px -5px rgba(0, 0, 0, 0.08), 0 4px 15px -5px rgba(0, 0, 0, 0.04)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
