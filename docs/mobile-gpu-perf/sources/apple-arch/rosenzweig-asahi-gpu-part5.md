# Asahi GPU Part 5 — Alyssa Rosenzweig

Source: https://rosenzweig.io/blog/asahi-gpu-part-5.html (redirects to https://alyssarosenzweig.ca/blog/asahi-gpu-part-5.html)

NOTE: This post is a debugging narrative about a rendering fault in the Asahi AGX (M1 GPU) driver, NOT an ISA/register reference. It does NOT cover register counts, fp16, scalar execution, or occupancy. Content relevant to the TBDR execution model below.

## GPU Architecture Classification
AGX is a **tile-based deferred renderer** (TBDR), derived from Imagination PowerVR. Key features:
- "tiled vertex buffer" (TVB) / "parameter buffer" (PB) system.
- Tilebuffer: "just a few kilobytes" for cached tile framebuffer data.
- Two passes: vertex shading for the entire frame, then per-tile fragment shading.

## Partial Renders (overflow handling)
Rendering fails when the "total amount of per-vertex data" exceeds parameter buffer capacity. The GPU implements **partial renders** — splitting render passes to flush the buffer when full. Auxiliary programs required:
- Load program (clears / reloads framebuffer into tilebuffer)
- Store program for final renders
- Store program for partial renders
- Load program for partial render resumption

Implication for us: heavy per-vertex/geometry data amplification can trigger costly partial renders on TBDR — favor bounded per-tile vertex footprints.
