# Cell-exit factorization of the pointed periodic grass field

**Date:** 2026-07-22  
**Status:** exact decomposition retained; primary codec rejected by the first
actual-source support gate. No runtime code or asset implements it.

## 1. Decision in one page

Moving the depth-zero plane to the botanical middle is a useful quantisation
convention, but it does not remove a coordinate. A two-plane or epipolar
parameterisation is likewise only a coordinate system for the four-dimensional
exterior oriented-line field. Neither determines the successor after an
arbitrary inside origin. No centered-depth or projective warp closes the
missing pointed-line dimension for finite Calamagrostis geometry.

Horizontal periodicity does provide one exact factorisation which the previous
direct-field attempts did not exploit. Split a pointed ray once, at the first
exit from its current canonical periodic cell (or from the finite botanical
slab). The exact successor is either:

1. a hit in that one finite cell segment; or
2. on a miss, the successor of the boundary-entry ray beginning at the exit.

The first oracle is a bounded local pointed field. The second is a
four-dimensional boundary-entry field which already includes every later
periodic copy. Runtime evaluates both with a fixed schedule and selects the
local event when present. It never walks cells, events, triangles, species, or
height shells.

This factorisation removes the **unbounded deep-list burden** from the local
inside query. It does not remove the difficult long-range exterior field. On
the existing 174,720-ray actual-mesh validation split, only 9.24% of all rays
and 15.18% of true hits resolve before the first cell/slab exit. Near horizontal,
only about 10.3% of true hits are local. Thus almost every low-oblique query
still depends on `B`, the same four-dimensional long-range visibility object
whose missing angular support produces the reported stretching.

A concrete four-read allocation would fit in 42,080,048 resident bytes and
below 256 FMA-equivalent operations, but fitting it cannot be the next primary
attempt: the new local factor controls too little of the failing image and the
boundary factor has no new closure mechanism. The allocation is preserved as
costed reusable work, not as an accepted low-memory theorem.

## 2. Exact pointed-line object

Let the offline-composed community geometry `G` be periodic under

\[
\Lambda=\{(nT_x,0,mT_z):n,m\in\mathbb Z\}
\]

and bounded by `y_min <= y <= H`. A live pointed ray is

\[
R(A,d)=\{A+t d:t>0\},\qquad \lVert d\rVert=1.
\]

Its exact answer is the first marked interaction

\[
S(A,d)=\operatorname*{argmin}_{P\in G\cap R(A,d)}\lVert P-A\rVert,
\]

or miss. The input has five continuous degrees of freedom: four for an
oriented line and one for the origin phase along that line.

All event attributes are coupled. Position, normal, authored colour/material,
species, plant part, primitive id, and coverage come from one selected event.
The union of all ground-cover species and geometric moss is formed before this
selection.

## 3. Why centering is not a dimension reduction

For a nonparallel reference plane `n dot X = k`, the line intersects at

\[
Q_k=A+d\frac{k-n\cdot A}{n\cdot d}.
\]

Changing the plane from `k` to `k+c` gives

\[
Q_{k+c}=Q_k+d\frac{c}{n\cdot d}.
\]

This is an invertible shear of the same oriented-line coordinates. Encoding a
hit by signed offset from `Q_{k+c}` instead of `Q_k` subtracts a known affine
term. It can reduce numeric range and direction-mismatch displacement, but it
does not identify two different lines or two different origins on one line.
The Jacobian has the same rank wherever `n dot d != 0`.

For an inside origin, choose two events `tau_1 < tau_2` on one line and origins
`eta_0 < tau_1 < eta_1 < tau_2`. Both origins have identical centered
oriented-line coordinates and different correct successors. Thus no choice of
center plane, cosine scale, tangent frame, or absolute-versus-relative depth
can remove the pointed coordinate.

Centering remains the right **payload convention** after the owner is known:
store the absolute chart parameter `tau_*` (or its centered quantisation), not
a range-dependent displacement tied to a camera or carrier. While an origin
moves without crossing the event, `tau_*` must remain constant.

## 4. Why an epipolar warp is not an exact closure

A two-plane light-field coordinate `(u,v,s,t)` is another four-dimensional
chart of oriented-line space. Within an occlusion-free patch, a visible 3D
point traces a line in an epipolar-plane image and a depth-aware projective
warp can transport that same event exactly.

The inverse is not globally defined at visibility changes. For any fixed set
of `K` reference directions, place a small opaque patch so that it is hidden by
foreground slats in all `K` references and visible in a direction between
them. A second scene without the patch has identical `K` reference records and
a different target answer. No decoder of those records can distinguish the
two scenes. Adding a centered plane only shears the same construction.

Consequently a finite projective warp is exact only under an additional
visibility hypothesis such as one common extrusion, one depth layer, or proved
same-owner support. The actual-source angular-flow oracle already measured the
failure of that support: even ideal access to the target event in four
neighbouring source views did not close all exterior or inside queries. A
projective factorisation may be a codec feature, but it is not a mathematical
dimension collapse.

## 5. Exact periodic cell-exit identity

Reduce `A_x,A_z` to the canonical cell

\[
D=[0,T_x)\times[y_{min},H]\times[0,T_z).
\]

Let `e(A,d)>0` be the smallest positive parameter at which `A+e d` reaches a
face of `D`. Parallel faces contribute `+infinity`; ties use one fixed face
priority. Explicitly, for axis interval `[a_j,b_j]`, its forward candidate is

\[
e_j=\begin{cases}
(b_j-A_j)/d_j,&d_j>0,\\
(a_j-A_j)/d_j,&d_j<0,\\
+\infty,&d_j=0,
\end{cases}
\qquad e=\min_j e_j.
\]

This is fixed arithmetic and a three-way minimum, not a traversal. Define the
half-open local segment

\[
I_0(A,d)=\{A+t d:0<t\le e(A,d)\}.
\]

Let

\[
L(A,d)=\text{first marked hit in }G\cap I_0(A,d)
\]

and let `B_f(u,v,d)` be the first marked hit after a ray starts on face `f` at
face coordinates `(u,v)` and continues through the complete future periodic
population. `B` includes the finite 155 m acceptance horizon but not a cell
loop at runtime.

If `L` is a miss and `E=A+e d`, then `E` lies on one selected face. For an X or
Z face, periodic reduction maps it to canonical boundary coordinates
`(f,u,v)`. For a Y face, the ray has exited the complete botanical slab, so the
answer is a miss. Therefore

\[
\boxed{
S(A,d)=
\begin{cases}
L(A,d),&L(A,d)\ne\varnothing,\\
B_{f(E)}(u(E),v(E),d),&L(A,d)=\varnothing\text{ and }f(E)\in\{X^\pm,Z^\pm\},\\
\varnothing,&L(A,d)=\varnothing\text{ and }f(E)\in\{Y^\pm\}.
\end{cases}}
\]

The proof is just ordered-set partition. Every positive point on the ray is
either in the first segment or strictly after it. A hit in the first segment
precedes every later hit; if there is none, the boundary suffix contains every
possible answer. No event or cell count appears in the formula.

Exact horizontal rays leave an X or Z face. Exact vertical rays leave the top
or bottom slab face. Camera-near and camera-inside origins are ordinary points
of `D`. There is no grazing denominator and no raised or rendered plane.
If `B` returns boundary-relative distance `t_B`, the live query accepts it only
when `e+t_B <= 155 m`; thus the finite horizon is still measured from the
original pointed origin without adding a horizon coordinate to `B`.

### 5.1 Right-censoring and cell seams

The local and boundary fields are not independent visual layers. On a local
miss they must reconstruct one world event:

\[
P_B=E+t_Bd,
\]

and moving an origin forward without crossing `P_B` must retain that same
marked point. Paired truth on both sides of every cell face is therefore a hard
identity test. A blend between unrelated `L` and `B` depths is forbidden.

The boundary field itself satisfies the stationary first-return equation

\[
B(q)=
\begin{cases}
L_{boundary}(q),&\text{current-cell hit},\\
B(Tq),&\text{current-cell miss},
\end{cases}
\]

where `T` is the affine entry-state map to the next cell. This equation is used
offline to trace or certify `B`; it is never iterated in the shader.

## 6. Rejected four-read resident candidate

The exact identity permits the following codec shape, but the support gate in
Section 7 rejects training it as the primary visual solution. Two separately
trained complete-record fields would be:

\[
\begin{aligned}
\widehat L(x,d)&=D_L(U_L(x)\odot V_L(d)),\\
\widehat B_f(u,v,d)&=D_B(U_{B,f}(u,v)\odot V_{B,f}(d)).
\end{aligned}
\]

`U_L` is periodic in X/Z and clamped in Y. `U_B` is periodic along the
horizontal face coordinate and clamped in Y. `V` is stored on a seam-correct
cube map or another sphere-native layout. `D_L,D_B` are small fixed decoders
which emit one complete event or miss, preferably absolute chart event
parameter plus coupled authored attributes.

One feasible rank-four layout is:

| allocation | format and dimensions | bytes including complete mips | reads |
|---|---|---:|---:|
| local spatial factors | `208 x 208 x 96`, RGBA16F | `37,973,216` | 1 |
| four boundary-face factors | `4 x 256 x 256`, RGBA16F | `2,796,192` | 1 |
| local spherical factors | cube `64 x 64 x 6`, RGBA16F | `262,128` | 1 |
| face-conditioned spherical factors | cube array `4 x 64 x 64 x 6`, RGBA16F | `1,048,512` | 1 |
| **total** |  | **42,080,048** | **4 fixed reads** |

This leaves `9,041,104` bytes below the existing `51,121,152`-byte cap for
decoder constants, packing metadata, gutters/alignment, or a measured payload
revision. There is one field for the complete community, not one field per
species.

With eight decoded scalars per field, a linear rank-four head costs 32 dense
FMAs plus four feature products. Both fields therefore cost 72 dense
FMA-equivalent operations before coordinate transforms and bounded output
maps. A small fixed nonlinear head can remain below the 256-FMA ceiling, but
its exact generated count must be recorded. Both fields are evaluated and
selected branchlessly, so there is no divergence-dependent work, traversal,
barrier, synchronization, dispatch, pass, or downstream species election.

The four samples would be from two 3D/2D factor fields and two sphere fields. Their
locality is coherent under camera motion. Register pressure consists of two
four-component factor pairs plus one local and one boundary record; a runtime
implementation must avoid materialising both full decoder output vectors at
once if generated code retains them unnecessarily.

### What this allocation does not prove

Rank four can still be too weak for local thin-owner topology, and a
`64 x 64` sphere factor can still underfit visibility boundaries. Linear
filtered factors can invent a transition unless the decoder treats hit/miss
and event selection categorically. The allocation is intentionally a single
go/no-go candidate, not permission to add ranks, candidates, or reads after a
failure.

The local field has a much shorter and qualitatively simpler burden than the
rejected 155 m direct pointed field: it only selects within one canonical cell
segment. The boundary field has no arbitrary origin coordinate and can be
prefiltered for the distant footprint after a local miss. These are concrete
reasons its separation rank might be lower, but the measured support shows that
the boundary field governs the failing low-oblique image. A rank-four `B` is
therefore merely another unproved codec for the already rejected exterior
field, not a new geometric solution.

The accepted cell diagonal is still about `1.38634 m`, and the existing
actual-mesh census already observed as many as 69 surface events in a 1 m
horizon. Therefore "one cell" must not be silently replaced by one or four
primitive candidates. The codec either learns/selects the complete local
successor at fixed work or fails this gate.

## 7. Actual-source support gate

Immutable single-species source:

```text
src/assets/groundcover/calamagrostis-canescens.gcrp
SHA-256 2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c
tile 0.51999998 m x 0.51999998 m
botanical height about 1.175222625 m
2,171,134 decoded triangles
```

The existing origin-aware validation truth was sufficient to execute the first
and cheaper required gate before any fit. Its immutable record is:

```text
data/work/groundcover-origin-aware-codec/
  2ed57f59d86e8376/5c69afaee4f55926/truth/validation.f32
records 174,720
SHA-256 10a6ec9143a844d2d54564bc430296ed8755f79d5273e1a71c77515564ec1a01
```

Reproduction:

```bash
node --import tsx tools/groundcover-bake/analyze-cell-exit-support.ts
```

- tool SHA-256:
  `5aef0178d46c17bc4b92139a57544e20d89f73f536e57f15f2ce82a9b90450ca`;
- recipe SHA-256:
  `ae91218c866d05bc965700bcb04cb9050231c96855db97178b378e2301764aa7`;
- report:
  `data/work/groundcover-cell-exit-support/2ed57f59d86e8376/ae91218c866d05bc/report.json`;
- report SHA-256:
  `3735dae7208ba2ee9472f249aa12acb6e59f4e3f017cfcc0987fa0a63d6a9f14`.

For each record, reconstruct the accepted origin from `(phaseX,
heightFraction,phaseZ)`, compute the three analytic face-exit candidates in
Section 5, and classify an existing hit as local iff `t_hit <= e+1e-5 m`.
This uses the already traced exact periodic successor; no new renderer or
approximation enters the measurement.

| absolute elevation | rays | true hits | local hits | local / all | local / true hit |
|---:|---:|---:|---:|---:|---:|
| 0 degrees | 13,312 | 13,178 | 1,356 | 10.19% | 10.29% |
| 0.1 degrees | 26,624 | 25,731 | 2,650 | 9.95% | 10.30% |
| 1 degree | 26,624 | 23,745 | 2,544 | 9.56% | 10.71% |
| 5 degrees | 26,624 | 19,841 | 2,499 | 9.39% | 12.60% |
| 15 degrees | 26,624 | 13,294 | 2,372 | 8.91% | 17.84% |
| 35 degrees | 26,624 | 7,220 | 2,235 | 8.39% | 30.96% |
| 75 degrees | 26,624 | 3,159 | 2,322 | 8.72% | 73.50% |
| 90 degrees | 1,664 | 167 | 167 | 10.04% | 100.00% |
| **all** | **174,720** | **106,335** | **16,145** | **9.24%** | **15.18%** |

The split helps vertical rays exactly as the derivation predicts, but those are
already the visually good regime. It barely affects horizontal, 0.1-degree,
and 1-degree rays, which are the widening/stretching regime. Training would
therefore spend the one permitted uncertain attempt primarily relearning `B`
without a new mechanism for its angular/disocclusion complexity.

If this exact decomposition is reused after an independently successful
boundary-field representation exists, its inspectable artifact belongs under

```text
data/work/groundcover-cell-exit-field/
  2ed57f59d86e8376/<recipe-hash>/qa/
```

and must contain numbered truth/prediction/error PNGs for:

1. isolated local-cell hits and misses, including origin moves through the
   botanical band;
2. boundary suffixes at exact horizontal and
   `0.1/1/5/15/35/75/90` degree elevations;
3. composed pinhole images at the user-reported top-down, far-oblique,
   ground-oblique, edge-on, and camera-inside paths;
4. paired frames on both sides of every X/Z cell face;
5. violet panicle mass versus green culm/leaf mass, never just aggregate green;
6. one offline-composed overlap of the accepted grass with a visibly different
   cover and geometric moss before any runtime generalisation.

Hard gates precede aggregate image scores:

- exact truth obeys the cell-exit identity on every sampled ray;
- predicted local/boundary composition has no face-linked world-point jump;
- same-line origins retain one absolute marked event until it is crossed;
- no connected camera-centred widening sector, constant-range height band,
  flat boundary plane, panicle dropout, or view-locked shimmer appears;
- fixed generated work remains at four reads, <=256 FMA-equivalent operations,
  and <=51,121,152 resident bytes.

No fit or PNG was generated after the support rejection: a pretty local 9.24%
subset could not demonstrate progress on the user-visible low-oblique defect.
Do not answer the failed support gate with more ranks, directions, events,
stages, or a hidden traversal.

## 8. Multispecies, moss, wind, and terrain boundary

For opaque cover, construct the marked union offline and trace `L` and `B`
against that union. A nearer moss or low leaf wins naturally; a violet panicle
retains its own colour and normal. Runtime performs exactly the same four reads
regardless of how many source species overlap.

Volumetric/translucent moss requires a marked segment-transfer record rather
than pretending one averaged depth is opaque geometry. Its local and boundary
records still use the same one-split composition; stochastic interactions must
be keyed to the canonical line so moving the origin does not reset the world
event.

A global affine wind transform conjugates the ray into the canonical field and
preserves the identity. Spatially varying wind does not. It requires separately
baked states or a proved deformation conjugacy and is not smuggled into the
first representation gate.

The derivation is exact in the community's local periodic root chart. Terrain
draping must use the same packed surface and a chart whose approximation error
is bounded; a floating constant-height carrier is neither needed nor allowed.

## 9. Honest conclusion

There is no centered-depth or finite epipolar formula which turns arbitrary
finite Calamagrostis into an exact three-dimensional Sannikov texture. The
missing dimensions are real. The cell-exit identity is a genuine new exact
factorisation: it converts the difficult pointed successor into one bounded
local query plus one boundary exterior query with constant runtime work.

The actual-source support gate rejects that split as the next visual solution:
90% of grazing hits still require the unsolved boundary field. Preserve the
identity for camera-inside composition after a future exterior carrier exists,
but do not build the 42.08 MB rank-four pair now. The active representation
question remains compression of the coupled exterior/pointed visibility field,
not trigonometry or another depth convention.
