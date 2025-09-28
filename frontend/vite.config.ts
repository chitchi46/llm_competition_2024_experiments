import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,          // 0.0.0.0 でバインド（WSL/別ホストからのアクセス許可）
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: true,
    port: 5173,
  },
})
