import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
  server: {
    hmr: process.env.DISABLE_HMR !== 'true',
    proxy: {
      '/health': 'http://localhost:8000',
      '/meta': 'http://localhost:8000',
      '/reset': 'http://localhost:8000',
      '/step': 'http://localhost:8000',
      '/state': 'http://localhost:8000',
      '/ohlcv': 'http://localhost:8000',
      '/council': 'http://localhost:8000',
      '/training': 'http://localhost:8000',
    },
  },
}));
