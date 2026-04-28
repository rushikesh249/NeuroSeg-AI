import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/progress': 'http://localhost:8000',
      '/result': 'http://localhost:8000',
      '/model-info': 'http://localhost:8000',
    }
  }
})
