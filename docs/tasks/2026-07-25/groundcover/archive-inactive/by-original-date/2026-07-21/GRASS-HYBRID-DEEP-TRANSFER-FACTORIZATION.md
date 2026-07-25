# Hybrid structural/deep-transfer ground-cover factorisation

**Date:** 2026-07-22  
**Status:** transfer algebra retained; the proposed rank-four-per-branch codec
is **rejected before the actual-source fit**. It is not implemented in a
runtime shader.

## 1. Inspectable outcome and decision boundary

The intended inspectable artifact was one low-oblique camera-path comparison
of the accepted Calamagrostis shoot, with its broad leaves and culm still crisp
and its complete hairy panicle represented as fractional deep opacity. The
adversarial audit found two prior blockers, so that fit must not consume the
one allowed actual-source attempt:

1. the frozen cheap decoder cannot represent the boundary-state-dependent
   finite-horizon suffix; and
2. the current live grass interface is an opaque one-winner visibility event.
   It carries no fractional alpha/transmittance to resolve and returns final
   alpha one, so it cannot preserve the proposed deep plume transfer.

The exact source partition and stable transfer truth remain reusable. The
four-read codec is not a runtime solution and has no test URL.

This proposal does **not** repair any existing runtime depth atlas. Every
currently live nearest/categorical/projected reconstruction remains visually
rejected. It changes the source visibility object so that millions of hair and
spikelet events no longer have to be encoded as mutually exclusive opaque
owners.

The split is deliberately narrow:

- foliage ribbons, culms, and rhizomes remain the opaque structural carrier;
- panicle axes, spikelets, callus hairs/filaments, and anthers become one
  fractional marked plume transfer;
- both are traced together offline and reduced to one composable segment
  record before any fit or filtering;
- later grass species, flowers, litter, and moss enter that same offline
  transfer. Runtime never queries one field per species.

Here “opaque structure” describes a delta interaction in the **offline ray
transfer**, not retained grass triangles, a hardware mesh, or a second runtime
query. The live representation is still only the precomputed field.

## 2. Why this changes the actual-source lower bound

The deterministic accepted generator partitions its `2,171,134` triangles
into the following real semantic families:

| source role | charts | triangles | representation role |
|---|---:|---:|---|
| foliage ribbon | 102 | 1,914 | opaque structure |
| culm tube | 6 | 336 | opaque structure |
| rhizome tube | 5 | 150 | opaque structure |
| panicle-axis tube | 20,969 | 450,280 | fractional plume |
| spikelet surface | 18,478 | 240,214 | fractional plume |
| callus hair/filament | 203,258 | 1,117,919 | fractional plume |
| anther surface | 27,717 | 360,321 | fractional plume |

Thus the opaque categorical burden falls from `270,535` charts and
`2,171,134` triangles to **113 charts and 2,400 triangles**. The remaining
`270,422` charts / `2,168,734` triangles are not deleted or replaced by a
coarse head. Their authored positions, colours, normals, recursive upward
branching, spikelets, anthers, and hairs contribute to the offline fractional
plume transfer.

This is not yet proof that the structural field fits rank four. It is a proof
that the previous event-count and owner-code lower bounds no longer apply to
the plume: any number of ordered fractional interactions closes into the fixed
record below. The 113-chart structural subset still has to pass the one
actual-source angular/camera-path gate; chart count alone is not acceptance.

## 3. Exact segment-transfer record

Consider one oriented ray segment of physical length `L`. Let `T(s)` be its
right-continuous transmittance from the segment start to distance `s`, including
fractional volume opacity and multiplicative alpha jumps at structural
surfaces, with `T(0-)=1`. Define the positive first-interaction measure

\[
d\nu(s)=-dT(s).
\]

The fixed record is

\[
\boxed{
\mathcal R_L=(A_L,M_L,C_L,N_L)
}
\]

with

\[
\begin{aligned}
A_L &= 1-T(L),\\
M_L &= \int_{(0,L]} s\,d\nu(s),\\
C_L &= \int_{(0,L]} c(s)\,d\nu(s),\\
N_L &= \int_{(0,L]} n_f(s,d)\,d\nu(s).
\end{aligned}
\]

`C` is premultiplied authored colour/material response and `N` is an
unnormalised linear normal moment. For a double-sided botanical surface,
`n_f` is made hemisphere-consistent toward the viewer before accumulation:
`dot(n_f,-d) >= 0`. The eight fitted scalars are

```text
A, m=M/L, C.r, C.g, C.b, N.x, N.y, N.z
```

where `m=0` when `L=0`. This is the stable payload and obeys `0 <= m <= A`.
`Q` may remain in offline truth only as an independent identity check.

An opaque structural surface is not approximated by a mean. It is the exact
limit in which `T` jumps to zero. Fractional plume fibres contribute smaller
jumps or continuous extinction. Arbitrarily many source events still produce
exactly eight scalars.

## 4. Representative election depth from the same record

Stieltjes integration by parts gives the exact first moment

\[
\boxed{
M_L=\int_{(0,L]}s\,d\nu(s)=Q_L-LT_L.
}
\]

Therefore the conditional first-interaction mean is

\[
\boxed{
\mu=\frac{M_L}{A_L}=L\frac{m_L}{A_L}
}
\qquad(A_L>0).
\]

The representative point written to the visibility election is

\[
P=A_0+\mu d.
\]

The physical inequalities are useful fit gates:

\[
0\leq A_L\leq 1,
\qquad 0\leq m_L\leq A_L,
\qquad \lVert N_L\rVert\leq A_L,
\qquad
0\leq\mu\leq L.
\]

Any decoded record which violates them is not an alternate visual
interpretation; it is outside the segment-transfer model and fails the codec.

Colour and normal follow the identical interaction measure:

\[
c_*=C_L/A_L,
\qquad
n_*=N_L/\lVert N_L\rVert.
\]

This closes the earlier depth/colour phase mismatch by construction. No
separate winner-only re-query exists.

The limiting cases are diagnostic:

- one opaque hit at distance `t_*`: `A_L=1`, `M_L=t_*`, hence `mu=t_*`
  exactly;
- empty segment: `A_L=M_L=0`, `C=N=0`, hence a miss;
- a fractional plume: `mu` is honestly the conditional first-absorption mean,
  not a claim of one microscopic triangle owner.

Near-field acceptance must measure the conditional variance offline. A broad
or multimodal fractional interaction can still make `mu` a poor opaque-depth
representative. That failure rejects the source split or codec; it is not
hidden by a confidence fade, a second event list, or a depth filter.

This payload correction is necessary because decoding `T,Q` independently and
then subtracting `Q-LT` is catastrophically ill-conditioned for sparse plume.
For `L=155 m`, `A=0.01`, and an interaction near 1 m, `Q` and `LT` are about
`153.46` and `153.45`; centimetre-scale codec or FP16 error overwhelms the
desired `0.01` numerator before division by `A`.

## 5. Exact fixed-size composition

For adjacent segments of lengths `L_1,L_2`, with the second record measured
from its own start, front-to-back composition is

\[
\boxed{
\begin{aligned}
A_{12}&=A_1+(1-A_1)A_2,\\
M_{12}&=M_1+(1-A_1)(M_2+L_1A_2),\\
C_{12}&=C_1+(1-A_1)C_2,\\
N_{12}&=N_1+(1-A_1)N_2.
\end{aligned}
}
\]

This operation is associative. If the local segment contains an opaque leaf,
`A_1=1` and every later plume/structure contribution vanishes automatically.
If it contains only translucent plume, the suffix is attenuated correctly.
No event count, species count, or periodic-copy count enters the live work.

Linear footprint filtering is meaningful only after each persistent micro-ray
has composed its complete structural-plus-plume transfer. Independently
averaging local and suffix records and then multiplying their averages is not
exact because `E[(1-A_1)M_2]` need not equal
`E[1-A_1]E[M_2]`. The fitted decoder
must therefore train against the already composed full-ray record, even though
the cell-exit factors below are its coordinate features.

## 6. Camera-inside and exact-horizontal existence without a torus primitive

A global antiderivative of positive extinction generally does not exist on the
periodic XZ torus. At exact horizontal, its mean mode has nonzero monodromy;
near rational directions the usual Fourier corrector also has small-divisor
poles. The proposed representation does **not** assume such an antiderivative.

Instead use the proved first-cell split. Reduce the physical origin `A` into
the canonical XZ cell and let `e(A,d)` be the first positive exit from that
cell or the botanical Y slab. This is one three-axis minimum. Let

\[
E=A+e d.
\]

The exact transfer of a deterministic ray is

\[
\boxed{
\mathcal R(A,d)=
\mathcal R_{(A,E]}\circ
\mathcal R_{(E,\operatorname{end}(A,d)]}.
}
\]

This half-open ownership is normative: the first segment owns an event at
`E`; the suffix excludes its origin. Face ties and modulo reduction use one
fixed direction-dependent convention, so a fractional boundary event is never
counted twice.

The endpoint is the first slab exit or the declared 155 m horizon. Exact
horizontal rays always leave an X or Z cell face, exact vertical rays leave a
Y face, and an inside origin is an ordinary point in the first cell. There is
no division by `d_y`, grazing clamp, raised shell, or rendered plane.

The canonical query starts inside the finite botanical slab. A camera outside
the slab is clipped analytically to its first slab entry; the prefix transfer
is empty. A ray which never enters is a miss. If the first-cell exit is a top
or bottom Y face, the boundary suffix is the identity record. The fixed query
therefore needs a computed `terminalY` mask; the four allocated X/Z face layers
do not and cannot name a Y suffix.

For a finite horizontal horizon the suffix length is `155 m-e`, so a boundary
field which ignores `e` would be mathematically wrong. The cheapest decoder
compatible with the four-read allocation was frozen for the adversarial gate
as

\[
\widehat{\mathcal R}(A,d)=
W
\begin{bmatrix}
U_L(A)\odot V_L(d)\\
U_B(E,f)\odot V_B(d,f)\\
r
\end{bmatrix}+b,
\qquad r=(L-e)/155,
\]

with four products per branch, one affine `9 -> 8` head, and physical output
maps for `(A,m,C,N)`. This is **rank four per branch / eight separable products
total**, not one rank-four field.

That frozen decoder is mathematically insufficient. It only adds `W_r r`; the
effect of remaining length cannot depend on the boundary state. Its mixed
cross-difference in boundary state `b` and length `r` is identically zero.
Even a homogeneous suffix has `T(b,r)=exp(-kappa(b)r)`, whose mixed
cross-difference is nonzero whenever two boundary states have different
extinction. An unrestricted nonlinear `D` could hide this missing interaction,
but would make “rank four” unfrozen and unfalsifiable. The candidate is
therefore rejected before fitting rather than expanded after seeing errors.

An empty exact-horizontal line has the unambiguous truth record

\[
(A,m,C,N)=(0,0,0,0).
\]

It does not require an infinite primitive, a mean-mode convention, or an
epsilon elevation. Nonempty horizontal lines are traced to exactly the same
finite endpoint. Dominant-axis chart overlaps and the sheared periodic action
must decode the same complete record.

## 7. Why continuous transfer removes angular cones

The current widening triangles are created when one neighbouring direction's
categorical event is relifted onto a live line. Their width grows as
`range * angular spacing`.

The hybrid field never relifts a neighbouring opaque owner. Its reconstruction
target is the complete, footprint-prefiltered transfer tuple. Structural
silhouettes enter as fractional pixel coverage at the filter footprint and the
plume is fractional by definition. The joint factor field is continuous on
the declared chart support; direction enters through a seam-correct
sphere-native factor rather than a nearest angular category.

This does not prove that rank four contains enough bandwidth. It does prove
that a successful fit cannot contain a camera-centred angular Voronoi cone:
there is no discrete direction selection in its evaluation. A visible cone,
fan, or wrong-view sheet is therefore a direct rejection of the fit rather
than an artifact to mask.

### 7.1 Fractional transfer does not fit the live visibility interface

The current grass query writes one opaque election: packed depth plus one
`body` id. Its only auxiliary screen value is `vec4(normal,tip)`. The resolve
then shades that winner and returns `vec4(lit,1)`. The previous opaque scene
winner is overwritten; neither its identity nor its colour is retained as a
suffix to composite through `T=1-A`.

Consequently `(A,C)` cannot survive the current interface. The only mappings
available without changing it are all rejections of the proposed transfer:

- write the mean event whenever `A>0`: invents an opaque floating surface;
- threshold `A`: causes view/distance-dependent plume dropout;
- stochastic/screen-door acceptance: can preserve expectation but introduces
  a sampling/noise/temporal contract which this candidate explicitly does not
  contain; or
- bake the background into `C`: invalid because terrain, objects, lighting,
  and later community overlap are not periodic source constants.

Honest fractional rendering needs `C + (1-A)B`, where `B` is the already
shaded opaque scene, or an explicitly accepted stochastic estimator. That
requires a different live resolve contract even if it reuses existing passes.
Until such a contract is derived and costed, a successful offline transfer fit
would still not be an implementable grass result. This is an independent
representation blocker, not a shader-plumbing TODO.

## 8. Multi-species overlap and moss

For every persistent offline micro-ray, form one physical extinction/interaction
stream over the union

\[
G=G_{grass,1}\cup\cdots\cup G_{forb}\cup G_{moss}\cup G_{litter}.
\]

Compose that stream in depth order into `(A,M,C,N)`. An opaque nearer leaf
blocks later material because its opacity is one. Translucent moss,
panicle hairs, and fine seed heads contribute fractional transfer. Authored
non-green colours survive in `C`; no green-only palette assumption is part of
the algebra.

The ecological control field chooses one offline-composed community state.
Adding source species or overlapping cover does not add a binding, read,
candidate, or branch to the ground-cover query. Arbitrary independent runtime
Boolean species mixtures remain outside the one-field contract.

## 9. Exact fixed runtime cost envelope

Reuse the already costed cell-exit factor allocation, but decode the eight
transfer scalars rather than one categorical microscopic owner:

| allocation | format/dimensions | bytes incl. mips | reads |
|---|---|---:|---:|
| local spatial factors | `208 x 208 x 96`, RGBA16F | 37,973,216 | 1 |
| four boundary-face factors | `4 x 256 x 256`, RGBA16F | 2,796,192 | 1 |
| local sphere factors | cube `64 x 64 x 6`, RGBA16F | 262,128 | 1 |
| face-conditioned sphere factors | cube array `4 x 64 x 64 x 6`, RGBA16F | 1,048,512 | 1 |
| **factor payload** |  | **42,080,048** | **4 sampling instructions** |

The remaining `9,041,104` bytes under the existing `51,121,152`-byte cap fund
decoder constants, packing metadata, measured gutters, and the eventual
community manifest. The complete logical payload must stay at or below
`51,121,152` bytes; driver allocation may be higher because of alignment and
implementation layout. No binding-limit result follows from the byte count.

The rejected frozen head used eight feature products and one affine `9 -> 8`
map (72 FMAs including the length coordinate). It met the arithmetic ceiling
but failed the finite-horizon identity in Section 6. A later codec may not
silently spend the remaining ALU on an unspecified nonlinear correction.

The isolated texture layout requires four sampled-texture bindings (3D,
2D-array, cube, and cube-array) plus a filter sampler. “Four reads” means four
filtered sampling instructions, not four physical texel or memory accesses;
3D/trilinear/mip filtering fetches multiple texels internally. The expected
live state is four factor `vec4` values and at most two output vectors. There is no data-dependent
loop, march, traversal, candidate list, species minimum, extra pass, dispatch,
barrier, or synchronization. Samples are coherent under neighbouring pixels;
the two spatial factors dominate memory traffic and must be measured for cache
locality before promotion.

## 10. Gate decision

The exact production Calamagrostis partition and stable `(A,m,C,N)` truth are
worth preserving. They can test a future representation without retracing the
2.17-million-triangle source. The proposed fit itself is **NO-GO**:

- the affine length coordinate cannot condition the boundary suffix;
- replacing it after inspection with an unspecified nonlinear head would make
  the rank/cost claim unfalsifiable; and
- even a numerically perfect fitted transfer cannot be expressed by the live
  opaque grass election/resolve interface.

Do not train this codec, add rank/views/reads, or map fractional `A` to an
opaque representative. Resume only after a different fixed-cost factorisation
contains the endpoint coordinate jointly and a no-extra-pass live compositing
contract preserves `C + (1-A)B`. Those are objective resume conditions, not
permission for a march, candidate list, shell, or stochastic coverage shortcut.

## 11. Source and contribution boundary

Primary-source ancestry is deliberately limited:

- Sannikov supplies the repeating precomputed-field/O(1)-copy objective and
  the evidence that direction mismatch stretches a reconstructed event.
- Lokovic and Veach, *Deep Shadow Maps* (SIGGRAPH 2000), supply the established
  idea of a prefiltered fractional visibility function through hair/fur/smoke:
  <https://graphics.stanford.edu/papers/deepshadows/>.
- Yuksel and Keyser, *Deep Opacity Maps* (Computer Graphics Forum 2008), supply
  the established depth-distributed opacity treatment for complex hair:
  <https://www.cemyuksel.com/research/deepopacity/deepopacitymaps.pdf>.
- VDM/GDM supply fixed-cost view-conditioned and origin-aware field ancestry,
  but neither supplies this record or factorisation.

The semantic structure/plume split, stable `(A,M,C,N)` monoid, exact
`mu=M/A` election identity, rejected finite-horizontal joint cell-exit decoder,
four-sample layout, and multi-species offline-union contract are LAAS
derivations/hypotheses. They must not be attributed to those papers. A future
publication claim requires the immutable source/recipe/tool/result hashes and
the failed opaque-field experiments alongside any successful result.
