import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), wasm(), topLevelAwait()],
  worker: {
    // Ensure WASM plugins run inside Web Workers (needed for ketcher-standalone)
    plugins: () => [wasm(), topLevelAwait()],
  },
  optimizeDeps: {
    // Prevent Vite from pre-bundling ketcher-standalone (it uses dynamic WASM workers)
    exclude: ['ketcher-standalone'],
  },
  define: {
    'process.env.PUBLIC_URL': JSON.stringify(''),
  },
})
