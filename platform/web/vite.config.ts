import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const target = "http://localhost:8787";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": target,
      "/ws": { target, ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
