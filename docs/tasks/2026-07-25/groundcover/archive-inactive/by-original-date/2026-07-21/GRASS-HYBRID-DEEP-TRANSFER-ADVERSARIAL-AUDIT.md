# Adversarial audit: hybrid structural/deep-transfer factorisation

**Date:** 2026-07-22  
**Scope:** `GRASS-HYBRID-DEEP-TRANSFER-FACTORIZATION.md` before the first
actual-source fit. No runtime shader was changed.  
**Decision:** **NO-GO as written; GO after the five stop corrections below are
frozen in the candidate and its fit recipe.**

The underlying transfer monoid is sound. The current storage variables and
codec contract are not yet safe enough to spend the one allowed actual-source
attempt: they can turn small approximation/FP16 errors into large false depths,
double-own a cell-boundary event, and let an unspecified nonlinear head absorb
the very fifth coordinate the experiment is intended to test.

## What passes

For a survival/transmittance function `T` with `T(0-)=1` and first-interaction
measure `dnu=-dT`, the identities

\[
A=1-T_L,
\qquad
M=\int_{(0,L]}s\,d\nu(s)=Q_L-LT_L
\]

and, for adjacent records,

\[
T_{12}=T_1T_2,\quad
Q_{12}=Q_1+T_1Q_2,\quad
C_{12}=C_1+T_1C_2,\quad
N_{12}=N_1+T_1N_2
\]

are correct. They support continuous extinction and multiplicative alpha jumps,
including an opaque jump, and composition is associative. A finite horizontal
segment also has an ordinary transfer record; no periodic antiderivative,
`d_y` division, raised shell, or grazing epsilon is required.

The four factor texture byte counts are arithmetically correct:

```text
37,973,216 + 2,796,192 + 262,128 + 1,048,512 = 42,080,048 bytes
```

`rgba16float` is filterable in WebGPU, and the proposed 3D, 2D-array, cube, and
cube-array dimensions are individually within core WebGPU limits.

## Stop correction 1: decode the stable moment, not `Q - L*T`

For sparse plume coverage, the proposed live election

\[
\mu=(Q-LT)/(1-T)
\]

is numerically ill-conditioned even though the identity is exact. At a 155 m
horizon and `A=1-T=0.01`, an interaction near 1 m has approximately
`Q=153.46` and `LT=153.45`; their difference is `0.01`. Independent codec or
half-factor errors of only a few centimetres dominate the desired numerator,
then division by `A` amplifies them again. This can recreate camera-relative
height bands without any error in the transfer algebra.

The fitted/stored eight-scalar target must instead be

\[
\boxed{(A,M,C_r,C_g,C_b,N_x,N_y,N_z)}
\]

or, preferably for conditioning, `(A,m=M/L,C,N)` with `0 <= m <= A` and known
query length `L`. Election is then `mu=M/A` (or `L*m/A`) without subtracting two
large nearly equal predictions. The exact stable composition is

\[
\begin{aligned}
A_{12}&=A_1+(1-A_1)A_2,\\
M_{12}&=M_1+(1-A_1)(M_2+L_1A_2),\\
C_{12}&=C_1+(1-A_1)C_2,\\
N_{12}&=N_1+(1-A_1)N_2.
\end{aligned}
\]

Truth generation may retain `Q` as a cross-check, but `Q` and `T` must not be
the independently approximated live depth representation.

## Stop correction 2: specify one boundary-event owner

`R_[A,E] compose R_[E,end]` is ambiguous if both closed records include an
interaction at `E`. Fractional opacity at a cell face would be counted twice.
Use one explicit convention throughout truth tracing, fitting, seam tests, and
runtime coordinates, for example:

```text
first segment: 0 < s <= e
suffix:        0 < s <= L-e, measured from E
```

Thus the first segment owns `E`, while the suffix excludes its origin. Cell
modulo reduction needs the same direction-dependent face/tie convention. A
camera exactly on physical geometry needs a declared one-sided/tie rule; it
must not be resolved with an epsilon that varies by view.

## Stop correction 3: close the Y-terminal and query-domain cases

Only four X/Z boundary-face layers are allocated. If the first exit is the top
or bottom of the botanical slab, the suffix is the identity record, not a
sample from one of those four layers. The fixed query needs a computed
`terminalY` mask which zeros the boundary features/record without adding a
texture read. Exact vertical directions exercise this path.

The candidate must also state its origin domain. The existing split is complete
for `A` inside the finite botanical slab. A camera origin outside the slab first
requires analytic ray/slab entry clipping (empty transfer before entry), after
which the canonical query begins at that entry. Otherwise “camera-inside and
arbitrary camera” is not actually a complete domain statement.

## Stop correction 4: freeze and name the actual rank/decoder

The written candidate is not a falsifiable “rank-four” fit. It contains four
local separable products, four boundary separable products, and `e/155`, then
passes them to an unspecified `D` with permission for a “small bounded nonlinear
correction.” In particular the correct suffix horizon is `155-e`; a merely
linear additive `e` input cannot modulate the boundary transfer by its remaining
length. An unrestricted nonlinear `D` can instead hide an unreported fifth-
coordinate approximation.

Before training, freeze:

1. the exact feature vector and truthful name (**rank four per branch, eight
   separable products total**, plus one length coordinate);
2. every multiplication between the length coordinate and boundary features;
3. all decoder layers, widths, activations, output maps, quantisation, and the
   exact FMA-equivalent count;
4. a split which holds out complete pointed lines and contiguous angular,
   origin-height, cell-face, and `e` intervals, rather than random samples from
   the same dense tables.

No architecture growth after seeing the gate is part of this one attempt. The
fit is then genuinely falsifiable: failure means the fixed eight-feature codec
does not contain the actual five-dimensional transfer field.

## Stop correction 5: make physical output and runtime cost claims exact

A linear head does not preserve transfer bounds. The frozen output map and gate
must enforce/test

\[
0\le A\le1,\quad 0\le M\le LA,\quad
\lVert N\rVert\le A,
\]

plus the authored colour bounds. Clamping after a bad fit is not proof: report
the unclamped violation rate and seam discontinuity. For double-sided botanical
surfaces, accumulate a declared hemisphere-consistent shading normal (normally
face-forward to `-d`) before the normal moment; otherwise opposite faces can
cancel and make `N/|N|` undefined.

The visual contract must also state how fractional `A,C,T` combine with the
terrain/background. A single opaque visibility winner using only `C/A` does not
render fractional plume transfer; it requires stable coverage/compositing
semantics. This can be evaluated in the offline path, but a fit must not be
called runtime-ready until the same semantics exist at the live election and
resolve boundary.

Finally, “four reads” means **four filtered texture-sampling instructions**, not
four physical texel or memory reads: trilinear 3D filtering and mip filtering
touch multiple texels internally. `42,080,048` bytes is the exact factor-texture
payload, not the complete resident allocation. The acceptance bound must say

```text
factor payload: exactly 42,080,048 logical bytes
complete payload: <= 51,121,152 bytes after constants, metadata, and alignment
shader resources: four sampled-texture bindings + at least one sampler,
                  replacing enough old atlas bindings to pass the actual stage limit
```

Actual driver allocation can exceed logical texel bytes. A later runtime gate
must record generated binding count and generated shader sampling instructions;
the isolated WebGPU format check alone cannot prove “no binding-limit increase.”

## Modified gate

After the five corrections above, one bounded actual-source fit is justified.
It must report both FP32 continuous-factor and quantised/filter-sampled runtime-
shape results. Acceptance is on held-out low-oblique camera paths, exact
horizontal/vertical rays, inside origins, both sides of every cell face, sparse
plume opacity, and Y-terminal exits. A fit that passes only training/random-ray
metrics, requires output clamps to hide invalid transfer, or changes rank/head
after inspecting errors is rejected without a runtime shader attempt.

