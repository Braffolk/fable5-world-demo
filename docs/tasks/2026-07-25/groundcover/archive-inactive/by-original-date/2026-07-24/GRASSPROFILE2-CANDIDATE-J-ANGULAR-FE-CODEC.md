# Candidate J: shared-vertex angular finite-element scale carrier

Date: 2026-07-24  
Status: pure mathematics and offline node-fidelity gate; runtime not authorised

## 1. Topology and C0 proof

The 64 regular directions form four rings of sixteen nodes at elevations 15,
35, 55, and 75 degrees, plus one vertical-pole node.  Partition the angular
disk into

```text
3 inter-ring bands * 16 azimuth cells = 48 quadrilaterals
1 pole cap          * 16 azimuth cells = 16 degenerate quadrilaterals
total                                      64 cells/pages.
```

An inter-ring cell stores four shared angular-vertex symbols.  A pole cell
stores its two adjacent 75-degree symbols and the pole symbol twice.  Bilinear
interpolation then reduces on the cap to

```text
(1-t)[(1-a)Vj + a Vj+1] + t Vpole.
```

The cook creates one canonical quantised symbol for every
`(scale,angular-node,world-phase)` and duplicates those exact bits into every
incident cell page.  Adjacent cells therefore evaluate identical linear data
on their shared edge.  At the cap, the two pole symbols and their decoded table
are identical.  C0 continuity across every former ring/azimuth boundary and a
unique pole limit follow exactly; no sampled direction is selected at runtime.

## 2. Two admissible eight-bit vertex codecs

Both spatial scales occupy one 64-bit cell texel: four vertex bytes per scale.

### J1: `A3 + attribute-code5`

Coverage is stored directly as UNORM3.  Five bits select one of 32
self-contained attribute prototypes under `(scale,angular-node)`.  Each
prototype is `verticalDepth12 | octNormal16 | colourClass4` in one `u32`.
Tables cost `2*65*32*4 = 16,640 B`.  This is the direct packing witness.

### J2: joint VQ8

All eight symbol bits select one of 256 joint prototypes under
`(scale,elevation-row)`, where the pole is a fifth row.  Each prototype is

```text
coverage8 | verticalDepth12 | colourClass4 | filteredNormal8 = 32 bits.
```

Ten tables cost `2*5*256*4 = 10,240 B`.  A table is shared across all sixteen
azimuth nodes in its row; the canonical index at a shared node is duplicated
unchanged into incident cells.  Sharing therefore does not weaken the C0 proof.

The first gate binds coverage, representative depth, and premultiplied RGB.
The normal bits remain charged and cannot be reclaimed; oct4+4 statistical-
normal fidelity is a second gate before runtime.

## 3. Exact memory and traffic

The scale atlas uses 64 pages instead of 65:

```text
64 * 256 * 256 * 8 = 33,554,432 B = 32.000 MiB.
```

With the unchanged 16.000 MiB near atlas, J1 totals 48.0159 MiB and J2 totals
48.0098 MiB.  Both are smaller than Candidate H and require exactly one scale
texture read.  The overall profile path remains four R0 + four R1 + one scale
= nine profile reads.

The codebook accesses are real memory traffic.  J1 performs four dynamic loads
per active scale from a 16.25 KiB immutable table; J2 performs four from a
10 KiB table.  If both spatial scales participate, that is eight cached table
loads.  They add no sampled texture, texture binding, pass, loop, march,
candidate list, or runtime geometry.  A shader transcription must unroll four
corners and decode/accumulate sequentially to limit live registers.

## 4. Node fitting and frozen gate

At each exact angular node, filtered truth gives coverage `A`, representative
ray parameter `tau`, premultiplied RGB `Cp`, and a statistical normal.  Store
vertical representative height

```text
y = H + tau*d.y
```

in 12 bits over `[0,H]`; at a fixed row it reconstructs
`tau=(y-H)/d.y`.  This avoids direction-dependent depth ranges inside a
row-shared J2 table and gives a worst height half-step of 0.144 mm for
`H=1.176m` (0.556 mm in ray depth at 15 degrees).

J1 clusters `(y, conditional colour, statistical normal)` independently at
each node into 32 prototypes while coverage is quantised separately.  J2
clusters `(A,y,Cp,normal)` jointly across all sixteen nodes in one elevation
row into 256 prototypes.  Prototype colours are restricted to the accepted
sixteen-entry palette; emitted predictions are coverage-premultiplied.

Exact-node limits, evaluated after all quantisation, are

```text
|Ahat-A| p95 <= .08, p99 <= .20
premul RGB max error p95 <= .06, p99 <= .15
|tauhat-tau| p95 <= .05m, p99 <= .10m
largest connected p99 exceedance < 1%
```

Interpolation cannot repair a failed node.  Only a codec GREEN at all 65 nodes
and both scales proceeds to statistical-normal and interior-direction truth.
Interior fidelity is not a theorem: bilinear finite elements can still miss
occlusion-order changes, thin horizontal branches, or multi-lobed angular
events.  The exact near stage remains categorical and is not blended by J.

