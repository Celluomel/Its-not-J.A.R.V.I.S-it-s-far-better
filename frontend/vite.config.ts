import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/next/',
  server: { proxy: { '/api/interface': { target: process.env.LUMINA_BACKEND_URL || 'http://127.0.0.1:8080', changeOrigin: true } } },
});
