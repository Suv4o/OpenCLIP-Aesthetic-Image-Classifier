import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/images": { target: BACKEND, changeOrigin: true },
    },
  },
});
