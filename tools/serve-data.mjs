#!/usr/bin/env node
// Local dev server for the cooked Estonia asset tree (asset-gen/data/out) — the stand-in
// for the S3 bucket until we can push. Zero deps; CORS-open so the vite-served runtime
// can fetch across ports.
//
//   node tools/serve-data.mjs            # http://localhost:8787
//   PORT=9000 node tools/serve-data.mjs
//
// Caching mirrors the S3 plan: latest.json is the ONLY mutable file (no-store);
// m/<hash>/ manifests and c/ chunks are content-addressed -> immutable.
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { join, normalize, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../asset-gen/data/out", import.meta.url));
const PORT = Number(process.env.PORT ?? 8787);

const TYPES = {
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".png": "image/png",
};

const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const rel = normalize(path).replace(/^([/.]|\.\.)+/, ""); // no traversal above ROOT
  const file = join(ROOT, rel);

  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method === "OPTIONS") return res.writeHead(204).end();
  if (req.method !== "GET" && req.method !== "HEAD") return res.writeHead(405).end();
  if (!existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404, { "Content-Type": "text/plain" });
    return res.end(`404 ${rel}\n`);
  }

  res.writeHead(200, {
    "Content-Type": TYPES[extname(file)] ?? "application/octet-stream",
    "Content-Length": statSync(file).size,
    "Cache-Control":
      rel === "latest.json" ? "no-store" : "public, max-age=31536000, immutable",
  });
  if (req.method === "HEAD") return res.end();
  createReadStream(file).pipe(res);
});

server.listen(PORT, () => {
  console.log(`serving ${ROOT}`);
  console.log(`  http://localhost:${PORT}/latest.json`);
});
