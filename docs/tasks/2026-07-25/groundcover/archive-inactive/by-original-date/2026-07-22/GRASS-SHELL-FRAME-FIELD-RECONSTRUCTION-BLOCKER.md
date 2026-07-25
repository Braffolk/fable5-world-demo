# Shell-frame field Section-12 reconstruction blocker

Date: 2026-07-22
Status: **RED; model and runtime implementation PARKED**
Owner: Codex, single-owner offline gate

## Decision

The Section-12 measurements are complete. The result is not the earlier
over-broad residual-law rejection:

- the connected dense-cover premise is **confirmed** in its proper regime;
- the fixed-97 shell-frame reconstruction is **empirically rejected** for the
  sparse/open authored community;
- the failure affects the outer silhouette crest and the content below it,
  so a silhouette-only lattice-budget revision cannot rescue this model;
- no runtime or shader implementation is authorized.

## Bound artifacts

Dense connected control:

`data/work/groundcover-shell-frame-field/2f1dca02ed141e8f/718dd44345aba66f/`

- source: authored `Sphagnum capillifolium` connected carpet;
- packed SHA-256:
  `cb61dd42c6265067f0be9a320d763da928b1e65a60ae3ebb359f49b3137a9959`;
- metrics SHA-256:
  `ec07d239b489c5681500e348f9ff3995d8ef6244338003fa4c897bfcab413e14`.

Final E2-compliant sparse reconstruction:

`data/work/groundcover-shell-frame-field/2f1dca02ed141e8f/cca44e1d11aa283e/`

- source: authored `Agrostis capillaris` sparse/open stand;
- packed SHA-256:
  `aa7af040b5b7aa66e45a7d5e3cf708a147eb53cda3de90de35275f9e7332daf4`;
- metrics SHA-256:
  `4251ca7a5456bccf54298e424774663668d7371c40f06aef8be7c5b6f46c6e5a`;
- QA SHA-256:
  `5af8ae56ed1ed88403256945144719c9dccfae978d8fbe5fc54cb248a40ec502`.

The final run uses the exact slope-chart epipolar identity

\[
L_i(u,\lambda)=B(u)+\lambda(s_i,-1),\qquad
u=q+\lambda(s-s_i),
\]

three regular-lattice barycentric nodes, per-view cover-only shells, and the
strict 9-read algebra represented offline as one shell read plus preliminary
and corrected categorical records per node. In the mean-plane XZ frame chart,
E2's invariant denominator condition is

\[
|\nabla o_i|\,\max_{\text{cell}}|s-s_i|\le 0.5.
\]

The earlier 64-pass smoother stop was removed. Every shell is now smoothed
until this inequality is true; a constant-shell fallback exists but the dense
control reached the bound normally in 0--210 passes. Thus the final RED is not
an ill-conditioned solve, a depth-unit error, pooled azimuth statistics, or the
old wrong silhouette mask.

## Dense-control result

The correct control has `100%` top-down interception, above the required
`95%`. Raw crisp residual p95 stays approximately `10.6--12.5 mm` from 90°
through 5° and falls to `3.4--4.0 mm` at 1°. After the exact E2 conditioning,
the 1° p95 is `7.0--7.8 mm`; every reported conditioning product is at most
`0.5`.

Therefore the designer's reservation was correct: grazing residual collapse
is real for a connected crest-forming carpet. It is not a general property of
sparse open cover.

## Fixed-97 reconstruction result

The fixed lattice is the Experiment-024 allocation: one shared vertical node
and 16 azimuth nodes at each of `85/60/35/15/5/1` degrees, 97 directions total.
Its estimated cost is not the blocker: `21,621,536 B` raw and `6,177,582 B`
under the proposal's 3.5:1 estimate, below the `50,844,684 B` baseline.

Strict 9-read aggregate versus the binding thresholds:

| Metric | Required | Measured |
|---|---:|---:|
| silhouette IoU | `>= 0.97` | `0.61974` |
| RGB max-channel p95 | `<= 0.15` | `0.36095` |
| largest connected wrong region | `< 0.01` | `0.44531` |
| crisp winner geometry p95 | `<= 0.05 m` | `12.88498 m` |
| unforced class change | `< 0.05` | `0.1966--0.5252` by view |

The stratified result is independently RED:

| Stratum | IoU | RGB p95 | crisp geometry p95 | largest wrong region |
|---|---:|---:|---:|---:|
| outer crest, +/-2 px | `0.59585` | `0.36500` | `13.99834 m` | `0.04036` |
| in-band content | `0.63567` | `0.36500` | `12.72273 m` | `0.40365` |

At 1° the binary silhouette is almost filled and reaches `0.99870` IoU, but
this does not constitute a correct reconstruction: RGB p95 is `0.26905`, crisp
winner geometry p95 is `15.45908 m`, and class changes are `52.23%`. At 18°,
10°, and 5° the strict IoUs are only `0.156/0.331/0.638`. The failure is not
confined to a subpixel horizon strip.

## Mathematical diagnosis

The shell split removes the *smooth mean* lever, but one residual record per
spatial texel and direction node does not make a sparse first-hit field locally
single-valued. A direction or sub-texel phase change can select a different
blade, height stratum, or periodic successor. The stored categorical point then
jumps by metres along a grazing line even though the plant-height residual is
only centimetres. Projecting that point onto the live ray does not undo the
successor change.

The proposed one-step correction assumes the preliminary residual belongs to
the same locally smooth event sheet as the corrected record. In sparse open
cover it frequently does not. The ablations expose this directly:

| Variant | IoU | RGB p95 | crisp geometry p95 |
|---|---:|---:|---:|
| 6 reads, no realignment | `0.62839` | `0.35118` | `12.25988 m` |
| 7 reads, winner only | `0.62110` | `0.35709` | `12.88498 m` |
| 9 reads, strict | `0.61974` | `0.36095` | `12.88498 m` |
| 12 reads, second step | `0.58276` | `0.37124` | `12.77573 m` |

More correction does not converge toward truth; it usually changes which
unrelated event is read and worsens radiance. This falsifies the claimed
second-order `O(Delta^2 sigma)` behavior for the sparse field. That bound is
conditional on a locally Lipschitz residual event sheet, exactly the condition
that multimodal first-hit ownership violates.

## Park record

- Effort spent: corrected slope-chart derivation; exact per-view conditioning;
  dense-control measurement; fixed-97 6/7/9/12-read reconstruction; crest and
  in-band radiance, geometry, connected-region, and translation metrics.
- Reusable result: an offline actual-mesh harness, dense-regime residual
  evidence, content-addressed QA, and a precise failure mechanism.
- Exact blocker: one categorical residual/event per node and spatial texel does
  not preserve the live first-event stratum under sparse-cover phase and
  direction changes. The alignment step has no information with which to select
  the correct successor.
- Why focus moves: the binding strict model misses every quality threshold by a
  large margin after the one premise-audit/fix cycle; further shell smoothing,
  filtering, or correction iterations would grind a disproved local premise.
- Fallback now active: none in this track. The earlier Class-E authoring route
  remains preserved separately; runtime grass remains on its current path.

## Objective resume condition

Resume shell-frame work only after an upstream mathematical revision supplies
all of the following:

1. a fixed-size representation of the multiple depth/owner strata in each
   sparse shell footprint (the proposed 2--3 categorical strata is one possible
   starting point);
2. an O(1), loop-free, non-blending rule that selects the live first stratum
   without an oracle, per-species work, a runtime candidate list, or marching;
3. explicit fixed tap, ALU, binding, and resident-byte accounting acceptable
   for low/mid-end hardware;
4. a new error argument that does not assume one locally Lipschitz residual
   sheet where ownership is discontinuous;
5. a prediction that can be rerun through this same actual-mesh gate.

Do not resume by adding direction rows, extra Newton steps, cross-owner depth
blending, a correctness filter, or a runtime mesh. Those do not supply the
missing stratum identity.
