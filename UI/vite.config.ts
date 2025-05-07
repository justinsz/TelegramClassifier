import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react({
      jsxRuntime: 'automatic',
      jsxImportSource: 'react',
      babel: {
        plugins: ['@emotion/babel-plugin']
      }
    })
  ],
  server: {
    port: 3000,
    strictPort: false,
    proxy: {
      '/telegram_messages_classified.csv': {
        target: 'http://localhost:3001',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'build',
    sourcemap: true,
    minify: 'terser',
    target: 'es2018'
  },
  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx']
  },
  esbuild: {
    loader: 'tsx'
  }
}) 