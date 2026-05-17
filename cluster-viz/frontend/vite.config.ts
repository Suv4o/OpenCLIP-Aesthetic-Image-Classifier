import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

const SEARCH_APP_BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [tailwindcss()],
  server: {
    port: 5174,
    proxy: {
      // Reuse the search-app backend's static image endpoint.
      "/images": { target: SEARCH_APP_BACKEND, changeOrigin: true },
    },
  },
});
