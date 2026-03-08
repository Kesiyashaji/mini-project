import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        brand: {
          teal:   '#2dd4bf',
          violet: '#7c3aed',
          navy:   '#0f172a',
        },
      },
      backgroundImage: {
        'glass': 'linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))',
      },
    },
  },
  plugins: [],
}

export default config
