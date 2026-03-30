import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// VITE_API_PROXY_TARGET: where the FastAPI app listens (no /api suffix).
// Local uvicorn default: http://127.0.0.1:8000
// Docker API published port: http://127.0.0.1:8010
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const target = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

  const apiProxy = {
    '/api': {
      target,
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
      timeout: 600_000,
      proxyTimeout: 600_000,
    },
  }

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 3000,
      proxy: { ...apiProxy },
    },
    // npm run preview uses this — without it, /api hits the static server and large uploads get 413.
    preview: {
      port: 4173,
      proxy: { ...apiProxy },
    },
  }
})
