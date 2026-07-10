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
    // the module graph never serves stale code (cost: dev-only CPU). BUT: polling
    // stats EVERY watched file per interval — with .claude/worktrees (3.9 GB, 12
    // full repo copies) + shots (3 GB) in-tree that was a constant 50-60% CPU on
    // the dev-server node process (found 2026-07-02). Ignore everything the module
    // graph can never import; 500 ms is plenty for tool-driven src edits.
    watch: {
      usePolling: true,
      interval: 500,
      ignored: [
        "**/.claude/**",
        "**/shots/**",
        "**/docs/**",
        "**/.cache/**",
        "**/dist/**",
        "**/.git/**",
        "**/tools/**",
        "**/asset-gen/**",
        "**/temp/**",
        "**/profile-results-**",

      ],
    },
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
  base: command === "build" ? "/laas-nanite/" : "/",
}));
