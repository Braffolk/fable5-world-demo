# GBC2-K6: guarded boundary CUT2 plus six-plane terminal

**Date:** 2026-07-23
**Scope:** pure mathematics, compiler contract, record format and prospective
runtime cost; no shader or runtime implementation
**Selected codec name:** **GBC2-K6** (`G`uarded `B`oundary, analytic
two-region `C`ut, plus a 4D `K`-planes-six terminal).
**Status:** **complete bounded compiler target; actual Calamagrostis fit is
unproved and is the binding gate.** The construction removes stored camera
standoff from a certified one-separator visibility cell by projecting the live
pixel Jacobian through that separator and integrating the cut analytically.
It does not claim that arbitrary visibility complexity fits a bounded record.
An arbitrary triangle soup is accepted as compiler input; publication fails
when its cells cannot be certified as REGULAR, CUT2, or bounded-error KPLANE
within 250 MiB.

The view contract is exterior to the **entire carrier box**, not merely
exterior to the triangles.  Once the camera origin crosses any carrier face,
the cover fades through the agreed guard band and no reconstruction quality is
claimed until it exits again.  A camera in empty air between blades but still
inside that box is therefore deliberately out of contract.  This relaxation
removes the interior-origin/succession state; it does not hide any exterior
direction.

## 1. Purpose and non-claim

The guarded common-slab factorization gives an exact central point-ray query

\[
 (o,d)\longmapsto q\longmapsto T(q,d),
\]

where `q` is the first carrier boundary point and `T` is the whole-forward
first-triangle transfer.  `(q,d)` has four continuous coordinates.  A camera
translated backward along the same central ray has the same `(q,d)` but a
different pixel footprint in this four-dimensional ray space.  A stored
point value alone therefore cannot be a physical pixel filter.

This codec does not add standoff as a stored fifth coordinate.  Instead it
stores the local algebraic visibility boundary.  The live camera supplies the
two-column pixel Jacobian, so the effect of standoff is fixed ALU.  This is
exact for a regular cell and for the area/moments of an affine binary
separator.  Low-degree curvature, ray-chart nonlinearity and attribute
variation are bounded and admitted only below declared screen-space errors.

The finite arrangement of an arbitrary soup can have unbounded complexity.
No fixed cell format can represent an unbounded multiway junction exactly.
Such a junction must become a directly decoded filtered terminal within the
error limit or make the cook red.  There is no runtime retry, traversal,
candidate list, or silent owner blend.  The concrete dense cap layout is a
`128 x 128 x 32 x 16` grid of 64-bit descriptors.  Ordinary cells decode
one/two source triangles analytically; complex cells directly select one
global physically-filtered six-pair-plane terminal.

## 2. Pole-free four-dimensional ray coordinates

### 2.1 Boundary face chart

Every planar carrier cap or side face has an affine chart

\[
 q(u)=q_f+B_f u,\qquad u=(u_1,u_2),
\]

where the columns of `B_f` span the face.  Rectangle-union edge/corner ties use
one fixed face order.  A pixel which crosses a face tie is not certified by a
single ordinary cut cell; it uses a jointly trained cap/side KPLANE seam cell
or makes the cook red.

For the common slab `P x I`, caps use periodic XZ phase.  A guarded rectangle
non-top face uses contour distance/segment coordinate and height.  A finite ecological
patch edge additionally carries its finite D4 edge chart; modulo-one community
phase alone is insufficient at a patch corner or finite far exit.

### 2.2 Signed dominant-direction slope chart

Choose

\[
 k=\operatorname*{argmax}_{a\in\{x,y,z\}}|d_a|
\]

with a fixed axis tie order, and let `sigma=sign(d_k)`.  If `i,j` are the other
axes, define the unnormalised direction

\[
 D_k=\sigma,\qquad D_i=p_1={d_i\over |d_k|},\qquad
 D_j=p_2={d_j\over |d_k|}.
\]

Both slopes lie in `[-1,1]`.  The ray is

\[
 x(\lambda)=q(u)+\lambda D(p),
\]

and metric distance is `lambda |D|`.  Six signed charts cover the sphere.
Exact vertical and horizontal rays are ordinary members of a dominant chart;
no division by the carrier vertical component and no grazing epsilon occurs.

The complete analytic cell address inside one chart is

\[
 z=(u_1,u_2,p_1,p_2)
\]

plus finite face and signed-direction chart ids.  For direct storage, the
finitely many signed charts are packed into one two-dimensional angular atlas;
the chart id is implicit in the addressed angular bin, not stored per cell.

## 3. Why triangle visibility boundaries are cheap algebraically

For triangle `a`, let its plane be

\[
 n_a\mathbin\cdot x=c_a.
\]

Put

\[
 A_a(u)=c_a-n_a\mathbin\cdot q(u),\qquad
 B_a(p)=n_a\mathbin\cdot D(p).
\]

Both are affine in their respective two variables.  Where `B_a` has a fixed
nonzero sign, the axial hit parameter is

\[
 \boxed{\lambda_a(u,p)={A_a(u)\over B_a(p)}}.
\]

Suppose one triangle-edge half-plane on triangle `a` is
`m_e dot x <= b_e`.  At the plane hit its numerator is

\[
 E_{a,e}(u,p)=
 (m_e\mathbin\cdot q(u)-b_e)B_a(p)
 +A_a(u)(m_e\mathbin\cdot D(p)).
\]

After the certified denominator sign is applied, `E=0` is the projected
triangle-edge boundary.  It is bilinear in `(u,p)`.  Similarly, the depth-order
boundary between triangles `a` and `b` is

\[
 \boxed{H_{ab}(u,p)=A_a(u)B_b(p)-A_b(u)B_a(p)=0,}
\]

again bilinear.  Positivity, finite-horizon, face-exit and denominator-pole
events have the same affine/bilinear form after denominators are cleared under
a certified sign.

Consequently the exact first-hit field of a finite soup is a finite
semialgebraic arrangement in each 4D chart.  The number of surfaces can still
be enormous; low polynomial degree does not imply low arrangement complexity.

## 4. The live pixel footprint supplies standoff as ALU

Let local screen coordinates be `eta=(eta_x,eta_y)` in
`[-1/2,1/2]^2`.  The unnormalised camera ray is affine,

\[
 R(\eta)=R_0+R_x\eta_x+R_y\eta_y.
\]

Normalisation cancels from slope ratios.  Within one signed dominant chart,
the derivatives of `p_i=R_i/|R_k|` follow from one quotient rule.  For a planar
carrier face `n_f dot x=c_f`,

\[
 t_f(\eta)={c_f-n_f\mathbin\cdot o\over n_f\mathbin\cdot R(\eta)},
 \qquad q(\eta)=o+t_f(\eta)R(\eta).
\]

Differentiating at the pixel centre gives

\[
 J={\partial z\over\partial\eta}
   =\begin{bmatrix}J_x&J_y\end{bmatrix}\in\mathbb R^{4\times2}.
\]

Equivalently, for a direction variation `delta d`,

\[
 \boxed{
 \delta q=t_f\left[\delta d-d\,{n_f\mathbin\cdot\delta d
                         \over n_f\mathbin\cdot d}\right].}
\]

Camera standoff is contained in `t_f`, hence in `J`; it is not an atlas
coordinate.  The selected physical target is the production pixel
parallelogram

\[
 \boxed{z(\eta)\simeq z_0+J\eta.}
\]

The compiler bounds the second-order difference between this parallelogram
and exact rectilinear subrays.  The bound includes chart curvature, face
intersection curvature and any inverse affine wind transform.  A cell is
publishable only when the induced RGB/coverage/depth error is below the frozen
production-pixel tolerance.  Exact corner subrays remain the offline truth;
large diagnostic pixels may not replace the production pixel angle.

## 5. Certified cell modes

Every directly addressed cell has one of four modes.

### 5.1 MISS

Interval evaluation proves no triangle hit before the finite horizon for the
whole cell and its supported pixel-footprint halo.

### 5.2 REGULAR

One coupled surface chart wins throughout that same domain with positive
denominator, triangle-support and depth-order margins.  The live central ray
is intersected with its stored plane exactly.  Normal, colour, material and
species mark follow this chart; no sampled depth or cross-owner arithmetic is
used.

### 5.3 CUT2

Exactly two coupled outcomes `A,B` occur in the certified halo (either may be
MISS).  Their boundary is one bilinear event surface `H=0`, and no third owner,
pole, edge or order event enters the halo.  The cell centre `z_c` is used only
by the cook to bound the cell.  At the **live** ray coordinate `z_0`, decode
the source payloads and evaluate

\[
 h_0=H(z_0),\qquad g_0=\nabla H(z_0),\qquad
 H(z_0+J\eta)=h_0+g_0\mathbin\cdot J\eta+R_2(\eta).
\]

No tangent coefficient is stored, and the tangent is not frozen at the cell
centre.  The exact bilinear Hessian is constant.  The cook charges

\[
 e_H\ge {1\over2}
 \max_{\eta\in[-1/2,1/2]^2}
 |\eta^T J^T(\nabla^2H)J\eta|
 +e_{\rm ray-map}+e_{\rm quant}.
\]

The area of the strip where the tangent could choose the wrong side is
computed, not hand-waved.  It must imply coverage/RGB/depth error below the
cell tolerance for every supported `J` in the address cell.  Otherwise the
cell is `KPLANE` or the cook is red.

### 5.4 KPLANE

A multiway/pole/face-tie/copy-overflow cell uses the one global K6 filtered
terminal only when its direct decode passes the declared disjoint held-out
empirical gate against converged offline pixel integration over supported
centres and `J`.  The decode contains premultiplied appearance,
coverage, two depth strata and normal/material moments.  It does not pretend
to be a real surface and cannot be depth/owner blended with regular charts.

This is not an interval proof over the continuous `(z,J)` domain.  Such a
claim would require an offline branch-and-bound certificate with interval
bounds for the reference transfer and the complete K6 decoder.  GBC2-K6 does
not currently claim that stronger certificate; its KPLANE acceptance is an
explicit empirical quality gate.

If K6 varies incorrectly with standoff/J or misses a resolved multiway event,
the community is RED.  There is no correction page or deeper mode.  Runtime
never enumerates the three or more owners.

## 6. Closed-form binary pixel integration

Project the live tangent separator through the live footprint:

\[
 h_0+g_0\mathbin\cdot J\eta
 =c+a\eta_x+b\eta_y.
\]

For `X,Y` uniform on `[-1/2,1/2]`, define

\[
 A=|a|,\quad B=|b|,\quad
 T=-c+{A+B\over2},\quad r(x)=\max(x,0).
\]

For nonzero `A,B`, the exact fraction on the CDF side
`a eta_x+b eta_y+c <= 0` is

\[
 \boxed{
 w_-={r(T)^2-r(T-A)^2-r(T-B)^2+r(T-A-B)^2\over2AB}.}
\]

The fraction on the positive side `c+a eta_x+b eta_y >= 0` is

\[
 \boxed{w_+=1-w_-.}
\]

The descriptor owner order fixes which decoded chart receives `w_-`; the
other receives `w_+`.  Reversing the convention without this complement is a
real full-versus-empty bug, not a harmless separator-sign choice.

The zero-axis limits are the ordinary clamped one-dimensional interval
fraction.  This formula is branchless apart from fixed `max` operations and
does not construct a polygon or candidate list.

Affine attribute/depth variation can also be integrated exactly.  In shifted
coordinates `x in [0,A]`, `y in [0,B]`, the unnormalised first x-moment is

\[
\begin{split}
 J_x={}&{r(T)^3\over6}
 -\left({A r(T-A)^2\over2}+{r(T-A)^3\over6}\right)
 -{r(T-B)^3\over6}\\
 &+\left({A r(T-A-B)^2\over2}+{r(T-A-B)^3\over6}\right).
\end{split}
\]

`J_y` is the same expression with `A,B` exchanged.  Transforming back and
applying `sign(a),sign(b)` gives the clipped first moments of `eta`.  Second
moments use the same four shifted positive-part terms through degree four;
they are a fixed formula.  Thus a region signal

\[
 f(\eta)=f_0+f_x\eta_x+f_y\eta_y
\]

has exact premultiplied integral under the affine separator.  The chart plane
gives central depth and its two live derivatives analytically.  Cooked colour
and shading-normal derivatives are used only when their certified linear fit
beats the tolerance; otherwise the cell is `KPLANE` or red.

The complementary region has weight `1-w_-`; because the whole symmetric
pixel has zero first moments, its first moments are the negatives of the first
region's.  At no point is a categorical plane, normal, mark or copy id
numerically blended and called geometry.  Only the physically filtered output
combines the two region contributions.

## 7. Multiway junctions and the exact limit

Two algebraic facts must not be conflated:

1. one triangle-order/edge boundary is low degree;
2. the number of such boundaries in a cell is not bounded by degree.

At a generic intersection of two separators, three or four visibility
regions meet.  An arbitrary soup can place an unbounded number of wedges at
one ray.  Exact pixel weights then require the whole local arrangement.  A
fixed record cannot contain it without becoming a candidate list.

This codec therefore has no `K=3`, `K=4`, or hidden overflow chain.  A
multiway cell is accepted only as a directly decoded filtered terminal whose
complete held-out error is bounded, or the cook fails.  This is the only
honest way to retain O(1) and arbitrary-soup input without claiming a
universal arbitrary-soup fit theorem.

The guard matters: geometry remains a positive distance inside the carrier,
so the smallest supported production pixel has a positive object-space
footprint.  Microscopic multiway structure can therefore collapse into a
terminal.  Large resolved multiway sectors do not collapse and are expected
to fail this bounded codec.

## 8. One direct 4D descriptor grid

The cap path has exactly one direct address and one 64-bit descriptor read:

\[
 128^2\times32_{\rm azimuth}\times16_{\rm elevation}
 =8,388,608\ \text{cells}.
\]

The two phase axes are periodic.  The `azimuth/elevation` labels state the
physical partition size: the 512 angular bins are distributed across one
packed atlas of the signed-dominant charts, not replicated for every chart.
Inside each ordinary bin the equations use the local slopes of Section 2, so
neither a spherical-coordinate pole nor an elevation division enters the
decode.  A bin touching a signed-chart seam is KPLANE and its seam band is
duplicated consistently in the packed angular atlas.  The bins cover the
inward cap hemisphere plus a margin bin at the face tie.  Exact horizontal entry is owned
by the finite side chart, not clamped into the last cap bin.  Deterministic
face ordering resolves the measure-zero edge/corner tie.  Thus there is no
direction hole and no grazing epsilon.

There is no page pointer, adaptive tree, retry, hash probe or correction
lookup.  A descriptor is valid only when its mode is certified over its
entire supported live-footprint halo.  If that halo leaves the proved CUT2
domain, crosses a face/pole, contains a third outcome, or exceeds the tangent
curvature bound, the descriptor is `KPLANE`; it does not fetch adjacent cell
records.  The finite non-top-face/corner charts use the same direct descriptor
semantics in separately streamed pages.  Their `16--64 MiB` working set
includes the canonical side/corner K6 features and jointly trained cap/side
seam cells; D4-equivalent edges reuse one canonical page.  Consequently a
side ray keeps the same one-descriptor and five/seven-read mode schedule.

## 9. Exact 64-bit descriptor and source chart

### 9.1 Descriptor packing

| bits | field |
|---:|---|
| 22 | source token `A` |
| 22 | source token `B` |
| 4 + 4 | signed periodic-copy residual `(dx_A,dz_A)` |
| 4 + 4 | signed periodic-copy residual `(dx_B,dz_B)` |
| 2 | mode: `MISS=00`, `REGULAR=01`, `CUT2=10`, `KPLANE=11` |
| 2 | CUT2 kind: pair-order / edge-0 / edge-1 / edge-2 |

This is exactly 64 bits.  A 22-bit token addresses at most `4,194,304`
source charts, including any reserved MISS token.  The accepted 2.17-million
triangle source fits this field, but an arbitrary soup with more tokens is a
format failure unless the cook proves legal coplanar/attribute merges.

The four-bit signed copy components represent `[-8,+7]` tiles.  They are not
absolute copy coordinates.  For a descriptor cell centre `q_c` and source
triangle centroid `g_a`, the compiler and decoder share the constant base

\[
 k_{0,a}=\operatorname{round}_{\rm ties\,fixed}
  \left({q_{c,xz}-g_{a,xz}\over L_{xz}}\right).
\]

The stored nibble is the residual from `k_0`.  Because `q_c`, not live `q`,
defines the base, this introduces no within-cell rounding boundary.  If either
winning copy needs a residual outside `[-8,+7]^2` anywhere in the certified
halo, that cell is `KPLANE`.  This range gate is binding at grazing angles;
it may not wrap or saturate silently.

`MISS` ignores both tokens.  `REGULAR` uses only `A`.  `CUT2` uses both; one
token value is reserved for a MISS side.  `KPLANE` ignores the tokens and
directly queries the global terminal.  The two subtype bits are therefore
available to identify the active edge for an edge cut without any separator
index.  For an edge cut, token order is canonicalised so the active edge
always belongs to `A`; a boundary requiring simultaneous edges is not CUT2.

### 9.2 Thirty-two-byte source triangle payload

Each source token owns one 32-byte, two-cache-line payload:

| bytes | field |
|---:|---|
| 18 | three tile-local positions, `3 x xyz16` |
| 6 | three premultiplied vertex colours, `3 x RGB565` |
| 6 | three shading normals, `3 x oct8x2` |
| 2 | categorical species/material mark |

The periodic copy translation is applied before intersection.  Plane,
barycentric gradients, active edge polynomial and pair-order polynomial are
reconstructed from the decoded vertices.  Therefore CUT2 needs no separator
atlas or stored tangent coefficient: `E` or `H` and its exact gradient/Hessian
come from the same two coupled charts whose appearance is integrated.

The stated encodings are capacity formats, not presumed sufficient quality.
Position, colour and normal quantisation are charged to the cell certificate.
A source needing UV texture detail can first be deterministically diced into
attribute-linear source tokens, provided it remains below the 22-bit and byte
ceilings; otherwise the cook is red or those cells use KPLANE.

For `2,171,134` source triangles the payload is `69,476,288` bytes =
`66.26 MiB`.  One payload is two aligned 16-byte reads.  No periodic copy
duplicates the payload.

## 10. K6 physically-filtered terminal

`KPLANE` is not a variable owner overflow.  It is a global, directly addressed
approximation of the already-filtered four-coordinate boundary transfer.
Normalise `z=(u_1,u_2,v_1,v_2)` to four unit coordinates, where `v` is the
packed signed-chart angular coordinate and is locally an affine rescaling of
`p`.  For each of the six
unordered coordinate pairs, one `448 x 448 x RGBA16F` plane stores four fitted
features.  The six samples form 24 features.  One fixed, shallow head also
receives analytic `z` and a compact fixed set of `J` invariants (pair footprint
areas, principal log-scales and orientation signs, computed with no reads),
then decodes two premultiplied depth strata: front/back coverage and RGB,
depth moments, normal moments and material/species mixture moments.  The two
strata are a filtered compositing representation, never two asserted triangle
owners.  Head width and instruction count are frozen by the `<100`-FMA lane
envelope; these analytic inputs do not create a candidate path.

All six planes have a complete two-dimensional mip chain.  Their exact bytes
are

\[
 6\cdot8\sum_{m\in\{448,224,112,56,28,14,7,3,1\}}m^2
 =12,844,752\ \text{bytes}=12.25\ \text{MiB},
\]

leaving about `0.75 MiB` inside a `13 MiB` K6 budget for the fixed head,
scales and alignment.  These are **fitted scale levels**, not ordinary averaged
feature mips: every stored level is jointly fitted and validated against
converged production-pinhole integrals at its projected footprint scale, and
training/validation includes samples in every trilinear transition band.

Each plane is sampled once with gradients obtained by projecting the live `J`
onto that coordinate pair.  The gradients only select/interpolate the fitted
levels; no claim that a nonlinear head commutes with ordinary mip averaging is
made.  Six logical samples do not mean six physical texels under anisotropic
filtering; actual sampler traffic and the maximum anisotropy used are part of
the performance gate.  Replacing the gradients with one isotropic LOD is a
different, lower-fidelity ablation.

This terminal is approximate and may be non-local.  It is accepted only when
the disjoint held-out empirical gate passes RGB, coverage, two-stratum depth
and temporal stability over the declared KPLANE sample distribution and all
adversarial boundary families in Section 13.3.  Failure is a codec failure,
not permission to add owners or samples.

This is not a revival of the rejected six-coordinate, fifteen-pair pointed-ray
field in
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-21/GRASS-PAIRWISE-PLANE-TRANSFER-GATE.md`.  That experiment
factorised a finite pointed ray before common-slab reduction and failed with a
rank-four product plus quadratic head.  K6 acts after the exact reduction to
the four-dimensional boundary transfer, uses six coordinate pairs, and is
only the directly filtered terminal.  That distinction justifies a new gate;
it does not prove that K6 will fit the Calamagrostis field.

## 11. Fixed read, ALU and resident-memory envelope

### 11.1 Reads and live work

For one top-cap lane:

| mode | logical texture reads |
|---|---:|
| MISS | 1 descriptor |
| REGULAR | 1 descriptor + 2 payload lines = 3 |
| CUT2 | 1 descriptor + 4 payload lines = 5 |
| KPLANE | 1 descriptor + 6 filtered pair planes = 7 |

The prospective resource mapping makes those counts literal:

- descriptors are a packed `RG32Uint` 2D/array texture and use one integer
  `textureLoad`; they must be declared with unsigned-integer sample type, not
  rebound through a float texture declaration;
- source payloads are a 32-byte-aligned read-only storage-buffer array; one
  token causes two aligned 16-byte loads, and adjacent pixels normally reuse
  the same one/two tokens through cache;
- all six K6 pair planes and their fitted levels are layers/mips of one
  `RGBA16F` 2D-array texture binding, sampled with one shared sampler and six
  statically named `textureSampleGrad` operations.  Canonical face-page bases
  select a fixed six-layer group; there is no loop or dynamic candidate index.

Thus one community adds one integer descriptor texture, one read-only payload
buffer, one filtered K6 array and one sampler (`4` binding entries; the control
field is existing).  Cap/non-top streaming replaces or rebases views in those
same slots; it does not bind one resource per face or species.  Descriptor and
K6 access are screen-coherent; payload locality is reported because divergent
token loads are the remaining bandwidth risk.

Two Tier-1 layers reuse the resident bytes but perform two queries.  Both
CUT2 lanes plus the existing control read cost `11`; CUT2 + KPLANE costs `13`;
both KPLANE lanes cost `15`.  The last case is one read above the nominal
fourteen-read preference, not hidden as fourteen.  It buys direct bounded
handling of arbitrary multiway/pole cells without a candidate list.  A hard
fourteen-read SKU may fade or replace the second KPLANE lane only as an
explicit quality tier; the full contract keeps the fifteenth read.

That control read is load-bearing: its packed record supplies the selected
community, carrier/terrain affine chart, and both Tier-1 lane affines.  The
fifteen-read total does **not** hide a later terrain-anchor or winner-root
lookup.  GBC2-K6 deliberately makes every published carrier a locally affine
terrain/motion domain and charges the residual curvature, smooth world-keyed
tint/vigor/height field, and wind variation to the carrier certificate.  If
the residual is too large, the cook must use a smaller carrier within the
declared side/control-page bytes or fail.  A production variant which instead
samples the real terrain or deformation at the winning root adds that
dependent read and has a declared worst case of `16`, not `15`.

CUT2 live work is fixed: inverse affine ray/J transform, direct index, at most
two triangle intersections, one analytically regenerated `E` or `H`, its
live tangent, and the closed-form half-plane area/moments.  Its Hessian/error
bound is a cook certificate over the supported `J` domain, not runtime work.  K6 is
six fixed samples plus one fixed head.  Target envelopes are `<100` scalar
FMA-equivalents per lane, two reciprocals for CUT2 and no dependent chain
longer than descriptor-to-payload.  Generated Metal, register live ranges,
occupancy and actual filtered sampler traffic decide the performance gate.
There is no loop, march, candidate network, per-species work, runtime
geometry, dispatch, barrier or synchronization.

### 11.2 Resident bytes

The complete simultaneous working-set components are:

| component | bytes | MiB |
|---|---:|---:|
| cap descriptors `128^2 x 32 x 16 x 8` | 67,108,864 | 64.00 |
| source payloads, accepted 2,171,134-triangle source | 69,476,288 | 66.26 |
| K6 planes + mips + fixed head | <=13,631,488 | <=13.00 |
| finite non-top-face/corner direct pages | 16,777,216--67,108,864 | 16--64 |

The subtotal is `159.26--207.26 MiB`.  The complete published artifact must
report control fields, alignment, the finite-side directory, carrier metadata,
padding and
all decoder constants; the intended complete working range is
`195--230 MiB`, with a hard `250 MiB` ceiling.  The side figure is actual
simultaneous residency, not total disk bytes relabelled as streamed.  A view
requiring more than 64 MiB of resident side/corner pages is red.

The two golden-angle layers do not duplicate any table.  They query the same
resident community through different inverse affines.  A different source
soup changes the `32 N_triangle` term and must still satisfy the 22-bit token
and 250 MiB limits.  Thus this is a bounded general compiler for arbitrary
soup input, not a soup-size-independent memory theorem.

## 12. Two affine layers, overlap, wind and species

For Tier-1 layer `l`, wind is applied at the representation boundary: the
authored community and its carrier are conceptually transformed by one live
affine map, while the central ray and both pixel differentials are transformed
through `A_l^{-1}` before `(z_l,J_l)` is computed.  This is why global
height-proportional shear is an exact conjugation rather than a post-hit bend.
D4/golden-angle placement likewise changes coordinates, not resident bytes.
Species identity is a categorical chart mark, so a union-baked community has
no per-species loop.  Non-affine per-blade flutter or terrain deformation
inside one query patch is outside Tier 1 and cannot be smuggled into this
proof; curved terrain is handled by smaller cook-side carriers whose local
affine error is certified.  Likewise the required smooth world-keyed tint,
vigor and `+-15%` height modulation are evaluated once per carrier/lane from
the control record.  Their within-carrier remainder is part of the same
certificate; GBC2-K6 does not silently turn them into uncharged per-root
queries.

The exterior-box rule is tested before either lane query.  During the guard
fade after the camera enters the complete cover envelope, neither CUT2 nor K6
is asked to solve an interior-origin ray.

There is one remaining exactness boundary.  Independently filtering two
opaque lanes and then choosing one mean depth is not the exact filter of their
pointwise nearest union.  The exact joint pixel arrangement can contain up to
four binary regions plus inter-lane depth-order boundaries.  Enumerating them
would be a fixed candidate network, which this contract forbids.

The prospective two-lane compositor is therefore accepted only when the
offline gate proves that the fixed coupled composition used by the runtime
meets pixel RGB/coverage/depth tolerances over both layer phases.  Simultaneous
resolved cuts or order swaps which fail that test make the cell/community
red; they are not repaired with a four-chart live list.  This qualification is
load-bearing for overlapping species with different affine motion.

## 13. Offline compiler and certification gate

The compiler may use BVHs, exact predicates, subdivision, large memory and
loops.  Runtime may not.

### 13.1 Chart preparation

1. Decode the arbitrary opaque, marked triangle soup and prove carrier
   containment.  Alpha-cut source surfaces must be diced into equivalent
   opaque attribute-linear tokens or delegated to K6; they are not silently
   treated as opaque.
2. Canonicalise triangle planes, edge half-planes, attributes and periodic
   copies within the finite horizon.
3. Merge tessellation triangles into one chart only when plane/attribute/
   normal deviation has an independently certified screen-space bound.  A
   shared botanical label is not by itself permission to merge curved pieces.
4. Quantise the 32-byte payloads, reconstruct them through the prospective
   decoder, and include their geometric/attribute error in every later proof.
5. Fit K6 only on the training split of converged physical pixel integrals.
   The per-cell mode classifier and all quality decisions use held-out rays.

### 13.2 Cell proof

For every direct descriptor cell and its full production-footprint halo over supported
camera standoff, FOV, resolution and affine phases:

1. interval-bound all ray denominators, positivity and carrier/scene exits;
2. use the offline BVH to prove the complete first-owner set;
3. emit MISS only for an empty whole halo;
4. emit REGULAR only for one owner with positive margins and an encodable copy
   residual;
5. emit CUT2 only for exactly two outcomes governed by one regenerable
   edge/order polynomial, encodable copy residuals and a certified tangent/
   filter error;
6. otherwise emit KPLANE and require its already-trained two-stratum decode to
   pass the disjoint empirical gate on the held-out centres and live `J`
   classes assigned to that cell family, or fail.

There is no modal-owner fallback.  Unseen owners, denominator sign changes,
face-chart changes inside an ordinary cell, copy residual overflow and
unsupported footprint radii force KPLANE; if K6 does not pass them they are
proof failures.  There is no refinement or second descriptor read.

### 13.3 Train/held-out rate test

Fit source quantisation and K6 on disjoint cells and camera families.  Held-out
data must include:

- production-size pixels, exact shared-origin subrays and converged adaptive
  or randomized quadrature;
- top, oblique, one-degree grazing, exact horizontal and exact vertical rays;
- finite-patch side/corner entries and carrier seams;
- near guard-boundary, 1.6 m, 4 m, 16 m and horizon-scale standoffs;
- `1, 2.5, 4.5 mm` camera translations scored as excess decoded-vs-truth
  temporal change, not triangle-id churn;
- both Tier-1 affine phases, overlap/order swaps, D4 seams, moss-under-grass
  where enabled, and terrain/control cutoffs.

Required metrics are silhouette IoU, premultiplied RGB max-channel p95,
coverage p95, crisp regular-cell world/depth error, mixed-cell compositing
depth error, normal/material error, maximum/p95 cut-cell seam jump, and
unforced temporal error.  Distant geometry is judged in screen space using its
actual footprint; it is neither held to invisible millimetres nor waved away.

The frozen first gate is: silhouette IoU `>=0.97`, RGB max-channel p95
`<=0.15`, connected wrong-view regions `<1%` of the evaluated image,
unforced class/output change `<5%` under the millimetre translations, and
crisp winner world-position p95 `<=5 cm`.  K6 is also scored by stratum:
front/back coverage, premultiplied RGB and composited depth may not exchange
strata under an infinitesimal camera move except where exact truth does.  The
thresholds apply separately to ordinary and KPLANE pixels so an easy REGULAR
majority cannot hide a failed terminal.

### 13.4 Hard resource report

The artifact reports the exact source-token count and maximum token id,
32-byte payload error, fixed `64 MiB` cap descriptor bytes, mode fractions,
copy-residual range/overflow fraction, K6 plane/head bytes, actual simultaneous
side/corner residency, control/metadata/padding bytes, total working bytes,
logical-read histogram, descriptor-to-payload locality, maximum requested
anisotropy and measured sampler traffic.

Green requires `<=4,194,304` tokens, `<=250 MiB` complete simultaneous
residency, the Section 13.3 quality thresholds, and the stated fixed schedules:
at most `5` reads per CUT2 lane, `7` per KPLANE lane, and `15` including control
when both Tier-1 lanes are KPLANE.  Fifteen is the declared full-quality
worst-case, not a failure against the older fourteen-read preference.  One
diagnose/fix rerun is permitted.  A second end-to-end red parks the codec with
the measured resume condition.

## 14. Adversarial expectation for the accepted Calamagrostis

The accepted source has about `2.17 million` triangles.  Earlier point-field
sampling saw roughly `94` owners in a sampled `4^4` macroblock and `100%`
unique sampled categorical patterns.  Those figures do **not** refute cut
cells: a coherent boundary can replace many point labels, and adjacent
tessellation triangles may certify as one smooth chart.  They are nevertheless
a severe warning.

The likely failure modes are:

- many glume/hair silhouettes crossing the same halo, so cells are multiway
  rather than CUT2 and the KPLANE fraction is high;
- long grazing free paths overflowing the signed four-bit copy residual and
  transferring too much of the field to K6;
- distant footprints spanning several unrelated separators whose filtered
  response K6 cannot reproduce from six pair planes;
- simultaneous cuts in both Tier-1 phases breaking the fixed compositor;
- cap-plus-side simultaneous residency exceeding 250 MiB;
- K6 passing colour while failing two-stratum depth or temporal seams;
- RGB565/oct8 payload quantisation erasing the accepted plume detail even
  where topology is regular.

The first offline artifact must therefore census **MISS / REGULAR / CUT2 /
KPLANE / failed**, copy overflow, exact complete bytes and separately scored
K6 held-out error before building any shader.  A high KPLANE fraction is not
itself red because its cost is fixed; failed K6 pixels, failed two-layer
composition, token overflow, or total residency above the inequality are red.

## 15. Exactness ledger and verdict

The following are exact:

- common-slab central-ray factorisation;
- face plus signed-dominant slope coverage, including horizontal/vertical
  poles;
- triangle-plane intersection and bilinear edge/order equations;
- live first-order pixel Jacobian;
- regular-cell central geometry;
- affine-separator pixel area and polynomial moments, with `w_-` and `w_+`
  explicitly distinguished;
- the one-grid 64-bit direct descriptor address and bit packing;
- the logical read ceilings for each declared mode.

The following are bounded approximations and must be measured:

- exact rectilinear pixel cone versus its live Jacobian parallelogram;
- bilinear separator versus the live tangent plane derived from quantised
  source charts;
- chart attribute/normal linearisation and quantisation;
- the six-pair-plane K6 filtered multiway/pole/copy-overflow response;
- fixed two-affine-layer filtered composition.

The construction is therefore a concrete general **compiler** for arbitrary
opaque marked triangle-soup input and a named, directly indexed low-cost
runtime format for communities which pass: **GBC2-K6**.  It handles all
exterior ray directions; the only camera-domain relaxation is the explicit
fade after entering the whole carrier box.  It is not an unconditional theorem
that an unbounded soup fits 22-bit tokens, 250 MiB, or a six-plane approximate
terminal.  The next authorized action is one offline certification/rate gate
on the actual Calamagrostis plus a dense low community.  Runtime/shader work
is authorized only if that artifact is green.
