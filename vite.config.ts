import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 4096,
  },
  server: {
    port: 5173,
    strictPort: true,
    // tool-driven file writes are missed by fsevents on this setup; poll so
    // the module graph never serves stale code (cost: dev-only CPU)
    watch: { usePolling: true, interval: 200 },
  },
  esbuild: {
    target: "esnext",
  },
  optimizeDeps: {
    // three's capabilities/WebGPU.js uses top-level await; the dep optimizer's
    // default esbuild target rejects it. Match the app target so cold-cache
    // (re)optimization succeeds instead of relying on a warm .vite cache.
    esbuildOptions: { target: "esnext" },
  },
  base: command === "build" ? "/laas/" : "/",
}));
