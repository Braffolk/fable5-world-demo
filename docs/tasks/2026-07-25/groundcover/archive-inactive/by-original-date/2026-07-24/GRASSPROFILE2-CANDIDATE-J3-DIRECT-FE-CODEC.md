# Candidate J3: direct shared-vertex finite-element carrier

Date: 2026-07-24  
Status: pure mathematics and offline node gate; runtime not authorised

Candidate J proved the 64-cell C0 topology but its eight-bit VQ symbols were
RED.  J3 spends sixteen direct bits per scale vertex and recovers the bytes by
halving the filtered atlas from 256 to 128 texels per side.

Each `RGBA32Uint` cell texel stores four shared vertices for both scales:

```text
scale-1 vertex[4]: each A5 | vertical-height7 | palette-class4
scale-2 vertex[4]: each A5 | vertical-height7 | palette-class4
```

The same canonical 16-bit vertex record is copied bit-for-bit into all incident
cell pages.  The 48 inter-ring quads and 16 degenerate pole quads therefore keep
the exact C0 proof from Candidate J.  No angular selection or codebook exists.

Memory is

```text
64 pages * 128^2 texels * 16 B = 16 MiB scale atlas
256^2 * 65 * 4 B              = 16.25 MiB near atlas (16 MiB if the pole replaces a regular page layout)
```

The declared scale allocation is exactly 16 MiB, one physical texture read,
and no table loads.  The total is about 32 MiB, far below Candidate H.

For each angular node and scale, a 2x2 fine block is reduced by positive
moments:

```text
Abar = mean(Ai)
Pbar = mean(Ai Ci)
ybar = sum(Ai yi) / sum(Ai),   yi = H + taui*d.y
Cbar = Pbar/Abar.
```

The record stores `round(31*Abar)/31`, `round(127*ybar/H)/127`, and the nearest
accepted palette class to `Cbar`.  Reconstruction is

```text
tauhat = (yhat-H)/d.y
premulRGBhat = Ahat * palette[class].
```

Normal is a continuous statistical filtered normal derived analytically from
the growth axis and live ray.  It is not a face covector and never enters the
exact Candidate-F plane solve.

The exact-node gate reports both intrinsic codec error against the 128-square
moment field and full spatial error after nearest expansion against the original
256-square filtered truth.  Only the latter can authorise progression.  Frozen
limits remain coverage `.08/.20`, premultiplied RGB `.06/.15`, representative
depth `.05/.10 m`, and connected p99 exceedance below 1%.

If exact nodes are GREEN, J3 still requires held-out arbitrary-angle ray truth.
The four ring elevations are then selected cook-side by a minimax search over
grazing, standing, uphill, and pole families without changing the 64-page
topology.  Node fidelity alone does not prove bilinear occlusion fidelity.

