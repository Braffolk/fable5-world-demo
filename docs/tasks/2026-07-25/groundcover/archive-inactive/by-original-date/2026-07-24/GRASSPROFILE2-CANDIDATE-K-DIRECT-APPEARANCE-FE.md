# Candidate K: direct appearance-only angular finite elements

Date: 2026-07-24  
Status: **superseded summary; runtime not authorised**

The authoritative complete derivation, gate result, horizon contract, and
composition audit are in
`GRASSPROFILE2-CANDIDATE-K-BACKGROUND-MEASURE-FE.md`.  The nested one-read
spatial hierarchy is audited separately in
`GRASSPROFILE2-CANDIDATE-K-NESTED-LOD-AUDIT.md`.  This shorter note is retained
only as the first stored-node sketch; do not implement from it.

## 1. Upstream correction

Candidate H's fractional overlay deliberately preserves the real background
depth.  A filtered representative grass depth is therefore neither composited
nor required for HZB/AO/TAA.  Storing and fitting it in Candidates I/J created a
hard problem with no consumer.  Candidate K removes filtered depth entirely.

Scale selection uses the live ray's cover-entry footprint.  Let the cover-top
surface be `F(X)=0`, the camera ray be `X=o+t d`, and `n=grad F` at the entry.
For a screen coordinate `xi`, implicit differentiation gives

```text
t_xi = -n dot (o_xi + t d_xi) / (n dot d)
Q_xi = o_xi + t d_xi + d t_xi.
```

Project `Q_xi` through the authored phase basis to obtain the 2x2 phase
Jacobian.  Its singular values/area select the two prefiltered spatial scales.
This is analytic ALU from the camera ray and cover entry; it consumes no profile
read and cannot swim with a reconstructed grass surface.

## 2. Direct shared vertex

Each scale/angular/world-phase vertex is one `RGBA4444` symbol:

```text
premultiplied linear R4 | G4 | B4 | coverage A4.
```

There is no palette, owner, depth, or codebook.  Four shared vertices for each
of two scales fill one `RGBA32Uint` cell texel (16 bytes).  Bilinear angular
interpolation is applied directly to `(A*Cg,A)`, never to conditional colour.

Canonical vertex bits are duplicated into incident cells.  The 48+16 cell C0
proof from Candidate J applies verbatim, including the duplicated-pole limit.
The filtered normal is a separate analytic statistical normal from live view,
growth axis, and a plume fraction derived continuously from premultiplied
appearance.  It never enters the exact face-plane solve and receives its own
visual/lighting gate.

## 3. Page/resolution candidates

With `R` regular elevation rings and a sixteen-wedge pole cap, cell pages are
`16R`.  All candidates stay below the exact 32.5 MiB scale ceiling:

```text
4 rings:  64 pages * 180^2 * 16 B = 31.640625 MiB
5 rings:  80 pages * 160^2 * 16 B = 31.250000 MiB
6 rings:  96 pages * 144^2 * 16 B = 30.375000 MiB
8 rings: 128 pages * 128^2 * 16 B = 32.000000 MiB
```

The near atlas remains separate and unchanged.  Runtime performs one scale
texture read, fixed unpack, and four-corner interpolation per active scale; no
table read, extra binding, pass, loop, march, candidate list, or runtime mesh.

## 4. Stored-node gate

For each existing filtered angular node and scale, positive area resampling
reduces the 256-square truth to the candidate phase resolution.  Quantise
coverage and each premultiplied channel independently to four bits.  Report
both intrinsic quantisation error at stored resolution and the complete
area-downsample plus nearest-address reconstruction against the 256-square
truth.  Every direction and scale must meet

```text
coverage |error| p95 <= .08, p99 <= .20
premul RGB max error p95 <= .06, p99 <= .15
largest connected p99 exceedance < 1%
```

Only a stored-node GREEN proceeds.  Extra rings then require newly baked
held-out directions.  Ring elevations are selected by minimising the worst
held-out coverage/radiance error over grazing, standing, uphill, and pole
families; pages stay fixed for each candidate and no post-result ring is added.
