module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'quantum-blue': '#00e0ff',
        'quantum-purple': '#7a00e0',
        'quantum-dark': '#1a1a2e',
        'quantum-light': '#5c00e0',
      },
      fontFamily: {
        'quantum': ['"Roboto"', 'sans-serif'],
      },
      boxShadow: {
        'quantum': '0 4px 10px rgba(0, 224, 255, 0.3)',
        'quantum-glow': '0 0 15px rgba(0, 224, 255, 0.8)', // Efeito de brilho
      },
      keyframes: {
        'pulse-quantum': {
          '0%': {
            transform: 'scale(1)',
            boxShadow: '0 0 15px rgba(0, 224, 255, 0.6)',
          },
          '50%': {
            transform: 'scale(1.1)',
            boxShadow: '0 0 30px rgba(0, 224, 255, 1)',
          },
          '100%': {
            transform: 'scale(1)',
            boxShadow: '0 0 15px rgba(0, 224, 255, 0.6)',
          },
        },
      },
      animation: {
        'pulse-quantum': 'pulse-quantum 1.5s infinite',
      },
    },
  },
  plugins: [],
}
