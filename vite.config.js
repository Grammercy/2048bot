import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    // The UI also has a self-contained demo fallback, but proxying makes the
    // live monitor work with the dependency-free Python API out of the box.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
