# Coupled marked transport: the required grass truth object

Status: pure-mathematics replacement contract, 2026-07-22. No runtime shader
implements this document. No codec may be fitted from the old independent beam
marginals and no runtime change is authorised by this derivation.

## 1. Why the target changed

Coverage, depth, colour, and normal histograms at one ray do not determine how
visible events connect to neighbouring rays or to the same ray after the origin
moves. Two image fields can have identical marginals while one connects the
same blade between pixels and the other swaps blade and panicle identities at
every sample. The latter can pass aggregate reconstruction metrics and still
produce camera-centred fans, widening triangles, wrong-perspective sheets,
panicle disappearance, and shimmer.

The acceptance truth must therefore be a coupling, not a collection of
independent distributions:

\[
\boxed{G_m(q,\xi)=\text{one exact marked interaction or miss}.}
\]

Here `q` is a pointed oriented-line query, `m` is a declared footprint level,
and `xi` is a persistent micro-sample label. The same label must name the same
transported micro-line and event sheet wherever the mathematical invariants say
that it should. All returned attributes come from that one interaction.

## 2. Exact marked event sequence on an oriented line

Let the complete offline-composed community be periodic under the horizontal
lattice `Lambda` and have a finite declared botanical slab. Its opaque geometry
is the union of every grass species, flower, litter element, and geometric moss
element in the community state. Every primitive carries the immutable mark

\[
\mu=(\text{community},\text{species},\text{plant},\text{part},
      \text{material},\text{primitive}).
\]

For a signed dominant-axis chart `j`, write an oriented line as

\[
X_\ell(\tau)=Q+\tau v,\qquad
v=e_j+r_a e_a+r_b e_b,
\]

where `Q_j=k_j`, and orient the chart so travel in the live ray direction
increases `tau`. The pointed origin is

\[
A=X_\ell(\eta).
\]

The exact offline tracer returns the ordered event sequence

\[
\mathcal E(\ell)=
\{E_n=(\tau_n,P_n,\mu_n,c_n,n_n,\rho_n,\ldots)\}_{n\in\mathbb Z},
\qquad \tau_n<\tau_{n+1}.
\]

Shared-edge ties use one declared deterministic primitive rule. Periodic-copy
identity is canonicalised before comparison. A query has the exact successor

\[
S(\ell,\eta)=E_{N(\ell,\eta)},\qquad
N(\ell,\eta)=\min\{n:\tau_n>\eta+\epsilon\},
\]

or miss when no event occurs before the finite endpoint. Physical distance is

\[
t=(\tau_n-\eta)\lVert v\rVert.
\]

This already gives the mandatory right-censoring law. If two origins do not
cross the selected event, their event id and world point are identical and only
the local distance changes. When an origin crosses events, the successor index
may only increase:

\[
\begin{aligned}
N(\ell,\eta_0)=N(\ell,\eta_1)&\Rightarrow
  (P,\mu)_0=(P,\mu)_1,\\
\eta_1>\eta_0&\Rightarrow
  N(\ell,\eta_1)\ge N(\ell,\eta_0).
\end{aligned}
\]

No comparison of two mean depths is a substitute for this identity.

## 3. Persistent micro-samples and transported bundles

Let `(Xi, B, p)` be a fixed labelled sample space. A footprint construction is
a deterministic map

\[
B_m(q,\xi)=(\ell_\xi,\eta_\xi),\qquad \xi\in\Xi,
\]

from one central query and one persistent label to one pointed micro-line. The
coupled truth is

\[
G_m(q,\xi)=S(\ell_\xi,\eta_\xi).
\]

The word *persistent* is essential. Rebuilding an unrelated phase disk and
then independently sorting its hits at every query produces only marginals. It
does not define `G`.

For the decisive same-line test, construct the bundle once. When its origins
move by a chart increment, every member travels along its own direction:

\[
\ell_\xi'=\ell_\xi,
\qquad
A_\xi'=X_{\ell_\xi}(\eta_\xi+\Delta\eta_\xi).
\]

For a requested common vertical displacement `Delta y` and a non-horizontal
micro-direction this is equivalently

\[
A_\xi'=A_\xi+d_\xi\frac{\Delta y}{d_{\xi,y}}.
\]

At grazing angles the dominant-axis chart expression is used; there is no
division by a vanishing vertical component. A fixed local XZ offset at the new
height is not the same micro-line and is forbidden in this test.

The exact bundle record retains, for every label and every ordered origin:

- the oriented-line key and chart-independent line id;
- origin parameter `eta`;
- miss/hit, event rank, canonical primitive/event id, and complete mark;
- world position and barycentric coordinates;
- local distance and all authored attributes.

Thus identity is tested directly rather than inferred from similar colours or
distances.

## 4. Exterior top-entry invariance

Consider two exterior datums on one oriented line, both before its first entry
into the botanical slab. Re-anchoring either datum to the same algebraic entry
must produce the same pointed query. For every fixed `xi`, both queries must
therefore return the identical marked event and world point. Only the distance
from the original exterior datum differs.

This gate is executed for above-top, below-bottom, exact horizontal, and both
vertical senses where a forward slab interval exists. A floating slab surface
is never rendered; the slab is only the finite domain of this equality.

## 5. Pinhole and camera-path bundles are a separate coupling

The transported same-line bundle proves the line/successor invariant, but it
is not by itself a camera footprint. For a pinhole camera state `C` and sensor
subpixel label `xi`, the exact micro-ray is

\[
B_{\mathrm{pin}}(C,p,\xi)=
\left(C,\operatorname{normalize}
  (d_p+J_p u(\xi))\right),
\]

where `J_p` is the exact pixel-direction differential and `u(xi)` is a fixed
labelled sensor sample. Across camera frames the same `xi` remains the same
sensor label. The corresponding world line is allowed to change because a real
pinhole ray changes; it must not be replaced by an independently hashed sample.

Truth sequences must include:

- neighbouring regular pixels at near, middle, and 155 m range;
- camera translation along one central line;
- lateral and vertical camera motion;
- flat and sloped root charts;
- footprint levels derived from the actual pixel differential;
- transitions between adjacent footprint levels.

There is no claim that arbitrary camera motion preserves a world hit. Instead,
the coupled labels reveal the exact occlusion boundaries and connected event
sheets. Candidate output is compared to those sheets, connected angular-sector
width, and screen-space flow. This prevents a narrow global error rate from
hiding a sector whose physical width grows as

\[
w(r)=2r\tan(\Delta\theta/2).
\]

## 6. A marginal quantile or three means is not `G`

Let a beam contain events `E_i` with labels `xi_i`. Sorting the events only by
depth and storing three means discards the map `xi_i -> E_i`. It also changes
probability whenever the finite hit count is not divisible by three, and an
arithmetic mean can invent a point, colour, normal, or material belonging to
no event.

Therefore the previous coverage-plus-three-strata record is diagnostic only.
It is neither training truth nor an acceptance target. Any later reduction
must encode an explicit coupled map from persistent sample labels to marked
events, or prove an equivalent line- and frame-consistent construction. Rank,
colour error, and within-stratum variance are evaluated only after this hard
identity contract passes.

## 7. Line-anchored runtime sample label

If a future fixed-cost decoder emits one stochastic visibility event, its
sample label must be a canonical function

\[
\xi=h_m([\ell]_\Lambda,\text{community supertile}),
\]

not a function of `eta`, time, frame number, camera range, screen tile, or a
camera-centred radial sector. Equivalent dominant-axis charts and sheared
periodic representatives must give the same label.

The exact construction of `h_m` is deliberately not assumed here. A dense
cryptographic hash of continuous line coordinates can itself create a particle
field; a quantised hash creates cell boundaries. The mapping must be specified
and then pass the coupled camera-path gate. Until then, “stable random sample”
is not an implemented invariant.

## 8. Charts, periodicity, and the sphere are quotient identities

For a lattice translation `L in Lambda`, the line chart obeys

\[
(Q_a,Q_b)\sim(Q_a+L_a-r_aL_j,\ Q_b+L_b-r_bL_j).
\]

Every overlap of the six signed dominant-axis charts is another representation
of the same line. The truth set must contain paired encodings through both
charts and through the sheared periodic action. They must return identical
sample labels, event ids, marks, world points, and decoded attributes.

Likewise, an octahedral square is a quotient representation of `S^2`, not a
clamped rectangular domain. A future angular field must use either a
sphere-native basis or explicit edge/corner identification, including the
required edge reversal. Paired directions on both sides of each fold and
corner are hard tests. Clamping an edge texel and hoping training learns the
other side is mathematically the wrong boundary condition.

Finite quadrature must not introduce its own tangent-frame seam. A disk kernel
may be generated in a rotation-equivariant frame or be integrated densely
enough that frame rotation is provably immaterial; a helper-axis switch is not
allowed to become a visible latitude.

## 9. Marked overlap and moss

Opaque multi-species overlap is exact in this contract because the offline
event sequence is traced over the union before the successor is selected. The
winner carries one complete mark. Runtime plant count and species count remain
absent.

Translucent or volumetric moss cannot be represented by averaging its transfer
with an unrelated blade and calling the result one surface. It has two valid
research paths:

1. bake moss as explicit marked microgeometry into the same event sequence; or
2. define an exterior-anchored marked interaction point process on every line,
   with its random sequence keyed by `(ell,xi)`, then use the same successor
   law after the origin moves.

Resetting a free-flight random number at every inside origin would select a new
world interaction and violate right-censoring. No moss path proceeds until its
marked process has the same identity tests as opaque events.

## 10. Minimum acceptance truth before any new codec fit

The next truth artifact must contain a small but decisive structured set before
large random aggregates:

1. ordered same-line origin sequences with transported micro-lines and fixed
   labels;
2. exterior top-entry datum pairs;
3. pinhole bundles and camera paths with persistent sensor labels;
4. exact-horizontal, exact-vertical, and held-out
   `0.1/1/5/15/35/75` degree rows;
5. dominant-chart overlaps, sheared-periodic pairs, octahedral folds/corners,
   and the old tangent-frame switch latitude;
6. a screen-derived footprint ladder with separately traced levels;
7. a marked overlap community containing Calamagrostis, a visibly different
   cover, and moss;
8. scene-depth probes and flat/sloped root charts.

Hard gates precede aggregate metrics:

- **identity:** 100% event-id and mark agreement before a transported event is
  crossed;
- **ordering:** zero successor-rank regressions after origin motion;
- **quotients:** 100% chart, periodic, and exterior-datum identity agreement;
- **sphere:** no across-fold discontinuity beyond the declared numeric packing
  tolerance;
- **support:** no sampled runtime/mip texel outside proved interpolation support;
- **cost:** at most four fixed reads, 256 FMA-equivalent operations, and
  51,121,152 resident bytes, with no loop, march, traversal, candidate list,
  shell stack, pass, or runtime species query.

Only after those pass are coverage, depth, colour, normal, material, connected
fan width, shimmer, scene-depth straddling, and visual sequences considered.
Rank eight may never repair a failed identity or truth contract; it is relevant
only when a valid coupled rank-four representation fails by measured capacity.

## 11. Consequence for the current rank-four attempt

The preserved beam artifact is an independently sorted marginal dataset. Its
`192x64x192` sparse embedding would also leave at least 91.949% of spatial
texels untouched, and its clamped octahedral sampler does not implement the
spherical quotient. Fitting that codec cannot answer the active visual defect
and is stopped before training.

The reusable pieces are the actual-mesh tracer, authored colour/normal recovery,
periodic source binding, finite endpoint, deterministic split machinery, cost
accounting, and the negative intrinsic-spread measurement. The old beam files
remain preserved with hashes as a rejected experiment; they are not renamed as
coupled truth.

## 12. Provenance and paper boundary

Prior-art ancestry remains narrow:

- Sannikov supplies the repeating O(1) precomputed-field objective and public
  evidence about missing-elevation and direction-mismatch artifacts.
- View-Dependent Displacement Mapping supplies fixed-cost direct view
  conditioning and evidence that silhouette topology is a distinct signal.
- Generalized Displacement Maps supplies the origin-aware full-sphere query
  semantics and periodic-neighbour bake.
- the audited fur/PRT literature supplies negative boundaries: lighting
  transfer, shells, fins, and traversal do not solve this first-event contract.

The persistent labelled coupling, transported-bundle truth, marked
right-censoring tests, and quotient hard gates in this document are LAAS
derivations. They must not be attributed to those papers without a newly found
primary source. Exact links, local files, transferred/rejected claims, artifact
hashes, and failed experiments are maintained in
`../../deep-research/grass/GROUNDCOVER-LINE-FIELD-SOURCE-AND-EXPERIMENT-LEDGER.md`.

If the method eventually passes, the defensible paper claim is a fixed-cost,
coupled marked successor/transfer field for periodic ground-cover communities,
not exact ray tracing from an independent marginal texture.
