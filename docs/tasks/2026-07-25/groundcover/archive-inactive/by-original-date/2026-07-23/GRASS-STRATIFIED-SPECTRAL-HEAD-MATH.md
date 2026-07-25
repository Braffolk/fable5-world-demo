# Stratified finite-spectrum heads over an affine-extrusion core

Date: 2026-07-23  
Scope: pure mathematics only; no CPU/GPU gate and no runtime or shader change  
Status: **rejected as the complete near-field head representation; retained only as a conditional far-field/common-colour medium**

## 1. Question and conclusion

The candidate deliberately changes level after the F4/F6 failure:

1. keep only long foliage, culms, and at most an indispensable rachis in a
   small union of exact affine-extrusion fields;
2. remove every broad ruled panicle sheet;
3. compile the finite reproductive head into a jointly authored,
   nonnegative, horizontally periodic finite-spectrum extinction/emission
   field, split into a fixed number of globally ordered height/colour
   strata; and
4. evaluate every finite ray prefix analytically, including exact horizontal,
   exact vertical, resonant, air-gap-origin, and opaque-cutoff cases.

The ray algebra exists and is clean. It has no sampled camera direction, no
march, no candidate list, no hidden carrier, and no resonance pole. Two
Tier-1 global affine layers are also exact. The construction is written down
in Sections 3--5 because it is a reusable result.

It nevertheless **does not survive as the complete Calamagrostis head or as a
general compiler target for arbitrary triangle-soup heads**. There are four
independent blockers:

- a finite nonzero trigonometric density has no open spatial support, while a
  resolved sparse head has positive-width air corridors;
- stable localisation of the accepted `~0.09--0.11 m` head envelopes and
  their `4.5--6 mm` spikelet structure requires hundreds to thousands of
  two-dimensional modes, not a small `Q`;
- the violet/brown axes, pink glumes, cream hairs, and dark anthers coexist at
  the same heights, whereas exact small coloured-medium composition requires
  one common colour or globally nonoverlapping colour strata; and
- a participating medium has transfer but no categorical first surface,
  normal, or exact opaque depth for the finite glumes and anthers it replaced.

These are representation failures, not a nine-read bookkeeping problem. A
couple of additional texture reads cannot encode the missing spatial modes or
the missing colour/depth order. The spectral construction remains a strong
far-field LOD and the exact continuation of the **common-colour microscopic
fuzz** result in `GRASS-PLUME-TRANSPORT-MATH.md`; it is not an authorization to
turn the whole head into fog.

## 2. Promised input and domain

The cook input remains an arbitrary triangle soup. Semantic sidecars are
useful when present, but the mathematical target may not require manual
re-authoring or assume that future imported meshes expose Calamagrostis
generator roles. A compiler may fit a perceptually filtered object rather
than preserve triangle identity, but it must work at the declared closest
exterior distance and from every exterior direction.

The quality domain contains:

- every ray direction, including exact horizontal and exact vertical;
- origins in exterior air gaps at any height in the botanical slab;
- arbitrary forward prefix cutoffs from the nearest structural/scene event;
- infinitely repeated community copies;
- two globally affine Tier-1 populations; and
- jointly compiled overlapping species without a runtime species loop.

A camera literally inside authored plant matter may fade. This does not
remove the origin coordinate for an ordinary air gap between blades or
spikelets.

For the accepted Calamagrostis source, the immutable procedural author in
`tools/groundcover-bake/EstonianGraminoids.ts` supplies the relevant physical
scale. One `0.52 m` tile contains six flowering shoots. Each panicle is about
`0.18--0.23 m` tall and `0.09--0.11 m` wide, has 22 recursively branching
primary axes, and carries `4.5--6 mm` spikelets with cream callus hairs and
dark-purple anthers. The source colours for panicle axes, spikelets, hairs,
and anthers are materially different and occur throughout the same panicle
height range.

The analysis below first grants, without relying on it, that a small
affine-extrusion core can represent all long foliage and culms adequately.
The head failure therefore stands even under the strongest interpretation of
the proposed split. In reality the arbitrary-soup-to-core compilation remains
its own visual gate; F4/F6 already showed that broad global-height rulings are
not a valid way to absorb the omitted head.

## 3. Strongest exact spectral construction

### 3.1 Canonical field

Let the horizontal community be the torus

\[
  q\in\mathbb T^2_\Lambda,
\]

and partition head height into a compile-time set of half-open intervals

\[
  J_s=[h_s^-,h_s^+),\qquad s=1,\ldots,S,
\]

with disjoint interiors. In one stratum author

\[
 \kappa_s(q)=a_{s0}
   +2\operatorname{Re}\sum_{j=1}^{Q_s}
       a_{sj}e^{i k_{sj}\cdot q}\ge 0,
 \qquad k_{sj}\in\Lambda^*.
\]

Nonnegativity is a hard authoring certificate, not a runtime clamp. A general
certificate is a positive-semidefinite Gram form

\[
  \kappa_s(q)=v_s(q)^*H_s v_s(q),\qquad H_s\succeq0,
\]

where `v_s` is a finite vector of reciprocal-lattice phasors. Equivalently,
`kappa_s` is a sum of squared magnitudes of finite trigonometric polynomials.
The density spectrum is the corresponding finite difference set. This is the
strongest useful sum-of-squares form: it permits arbitrary fitted phases and
does not depend on clamping a signed Fourier approximation.

Give the whole stratum one source colour `c_s`. Its emission coefficient is

\[
  j_s(q)=c_s\kappa_s(q).
\]

This shared-colour condition is what makes exact Beer--Lambert composition
close on total optical depth.

### 3.2 Exact finite-prefix integral

After inverse-transforming one global affine layer, a world ray has canonical
form

\[
  q(t)=q_0+v t,\qquad y(t)=y_0+w t.
\]

Intersect the forward interval, `J_s`, the finite cover horizon, and the
nearest opaque/scene cutoff. Call the result `[a,b]`; an empty interval is a
miss. Put

\[
  \Delta=b-a,\qquad m=(a+b)/2.
\]

Then the exact stratum optical depth is

\[
\boxed{
 \tau_s(a,b)=a_{s0}\Delta
  +2\operatorname{Re}\sum_{j=1}^{Q_s}
  a_{sj}\Delta
  e^{ik_{sj}\cdot(q_0+vm)}
  \operatorname{sinc}\!\left({\Delta k_{sj}\cdot v\over2}\right)
}
\]

with `sinc(0)=1`. This has all of the required limiting cases:

- if `k dot v=0`, the mode is constant on the ray and contributes its value
  times `Delta` exactly;
- exact horizontal and exact vertical world rays need no epsilon or substituted
  angle;
- moving the origin within an air gap merely changes `[a,b]` and the phase;
- clipping at an arbitrary opaque event is the same formula with a shorter
  `b`; and
- periodic cell crossings require no special action because the phase lives
  on the torus.

The cancelled `sinc` form is essential. An endpoint potential would contain
the false pole `1/(k dot v)` and is already ruled out in
`GRASS-PLUME-TRANSPORT-MATH.md`.

### 3.3 Exact coloured composition under the stratum restriction

Within one stratum and colour,

\[
  T_s=e^{-\tau_s},\qquad
  C_s=c_s(1-e^{-\tau_s}).
\]

If the two Tier-1 populations use the same `c_s`, their densities may overlap
inside that stratum: optical depths add before applying the exponential.
Across disjoint height strata, the sign of `w` determines a fixed ascending or
descending front-to-back order. At `w=0` an exterior ray belongs to at most
one half-open stratum. A fixed `S`-way composition therefore is exact and has
no traversal.

An opaque event at `t_o` is handled by clipping every stratum interval to
`t<=t_o` and then composing

\[
  C_{out}=C_{head}+T_{head}C_{opaque}.
\]

This is exact for the authored shared-colour media and never invents a
floating representative plane.

The two global layer transforms must preserve the common height partition, or
overlapping layers of different `c_s` cease to be globally ordered. A
rotation/translation in the horizontal plane and an affine-in-height
horizontal wind shear satisfy this condition. Independently varying vertical
scales, per-cell height warps, or layer-specific shifted colour bands do not.
They need either one common head colour or an additional approximation.

### 3.4 Structural union

Let `t_o` be the fixed minimum over exact opaque affine-extrusion families in
both Tier-1 populations and the scene depth. The structural query and the
spectral query couple only through this cutoff. Thus the ideal live shape is

```text
fixed exact core-family reads -> t_o
fixed spectral prefix algebra on [0,t_o] -> (C_head,T_head)
front-to-back composite with core/scene
```

Species do not appear in the live count: every permitted species contribution
is compiled into the same core fields and same stratum coefficients. This is
correct only when the colour-order restrictions above hold for the compiled
species union.

## 4. What the construction really solves

For its authored medium, the preceding algebra proves:

1. no camera-direction lattice and therefore no `r DeltaTheta` wedge;
2. no direction interpolation of depth, normal, colour, or owner;
3. exact finite-prefix transport at every resonant direction;
4. exact exterior air-gap origins;
5. exact affine wind and global affine anti-tiling transforms;
6. exact periodic seams;
7. fixed work independent of copies and species; and
8. no F4/F6-style broad ruled head sheet.

These are meaningful results. In particular, a vertically stratified medium
is not geometrically the F6 pair of horizontal opaque masks: it has no sheet
intersection or sheet depth. What remains wrong is the authored object and
its required bandwidth, not the ray reconstruction.

## 5. The support and corridor obstruction

### 5.1 Exact open support is impossible

A finite trigonometric polynomial is real analytic. If a nonnegative
`kappa_s` is zero on an open two-dimensional patch, analytic continuation
makes it zero everywhere on the connected torus. Hence a nontrivial finite
spectrum cannot have the compact horizontal support of isolated finite
panicles.

Zeros on individual lines are possible; positive-width corridors are not.
The binding quantity is consequently not density RMS but maximal false prefix
optical depth in every source-clear, still-resolved ray bundle:

\[
 \boxed{
 E_{corr}=\sup_{\ell\in\mathcal C}
           \sup_{0<R\le R_{res}(\ell)}
           \int_0^R\kappa_Q(\ell(t))\,dt .
 }
\]

For a false-alpha allowance `epsilon_A`, every such prefix must satisfy

\[
  E_{corr}\le-\log(1-\epsilon_A).
\]

If the candidate has a roughly constant corridor floor `eta`, this becomes

\[
  \eta\le{-\log(1-\epsilon_A)\over R_{res}}.
\]

The earlier `6.48e-5 m^-1` number corresponds to the strongest possible
`epsilon_A=0.01,R_res=155 m` contract. Distance filtering makes the honest
resolved length shorter: if a clear corridor has width `g` and pixel angular
footprint `theta_p`, then, up to the declared filter radius convention,

\[
  R_{res}\asymp {g\over\theta_p}.
\]

Beyond this range the pixel bundle legitimately spans neighbouring heads and
the target itself becomes averaged. This is the correct use of distance. It
does **not** excuse leakage on the near prefix where the same corridor still
occupies pixels.

### 5.2 A useful constructive benchmark

The simplest nonnegative periodic localiser is

\[
 W_p(x)=\cos^{2p}\!\left({\pi x\over\lambda}\right).
\]

It has Fourier degree `p`, `2p+1` one-dimensional modes, unit peak, and tail

\[
  W_p(x)=\left|\cos({\pi x/\lambda})\right|^{2p}.
\]

A two-dimensional spot `W_p(x)W_p(z)` already needs
`(2p+1)^2` distinct Fourier modes in its exact ray integral. Shifted copies
for many panicle centres can share this support--their phases combine in the
coefficients--so plant count is not the problem. Localisation is.

For example, `p=7` uses `225` density modes. Its full width at half maximum is

\[
  {2\lambda\over\pi}
  \arccos\!\left(2^{-1/(2p)}\right),
\]

which is about `0.10 m` at `lambda=0.52 m`: only the broad envelope scale of
one accepted head. This is already far beyond a small `Q=8--16`, before
adding irregular internal spikelet clusters, steep gaps among the six nearby
heads, vertical colour structure, or a second Tier-1 population.

The benchmark is optimistic. Powers large enough to suppress tails in the
gaps narrow the central lobe; restoring its width with additional harmonics
or an equiripple window spends more coefficients. A low-rank sum of
nonnegative one-dimensional ridge functions does not escape this count: it
produces unions of stripes. Multiplying independent ridge localisers creates
spots but expands to their cross-product spectrum. Signed cancellation can
make sparse spots, but it destroys the nonnegative extinction certificate;
clamping it makes the analytic integral invalid.

### 5.3 A rigorous edge-frequency lower bound

For a bounded one-dimensional trigonometric polynomial of degree `N`,
Bernstein's inequality gives

\[
  \|f'\|_\infty\le {2\pi N\over\lambda}\|f\|_\infty.
\]

If a normalised head edge must fall from `0.9` to `0.1` across visible world
width `delta`, then necessarily

\[
  N\ge {0.8\lambda\over2\pi\delta}.
\]

At `lambda=0.52 m`, even a deliberately soft `delta=5 mm` edge needs
`N>=14`; a `2 mm` edge needs `N>=34`. A spatially localised head needs such
control in two independent horizontal directions. The inequality itself is a
frequency-degree bound, not a claim that every rectangular mode is always
needed. But an arbitrary stable two-dimensional compiler cannot obtain
independently placeable features from a few sparse high frequencies: sparse
high-frequency sets yield repeated fringes. The corresponding complete-grid
counts, `29^2=841` and `69^2=4761`, correctly show the scale of a robust
sum-of-squares localiser.

The accepted source is harder than its outer envelope. Its `4.5--6 mm`
glumes/spikelets remain several pixels wide until their distance is roughly

\[
  D_{spikelet}\asymp {0.0045\hbox{--}0.006over\theta_p}.
\]

For a representative `theta_p~1.0--1.1 mrad`, this is approximately
`4--6 m`. At one metre their geometry is plainly resolved. Treating the whole
head as a `Q=8--16` smooth medium is therefore not a slight loss of distant
detail; it deletes resolved near-field botanical structure.

## 6. Colour-order obstruction

For overlapping coloured densities, exact premultiplied radiance is

\[
 C=\int c(t)\kappa(t)
       \exp\!\left(-\int_0^t\kappa(u)du\right)dt.
\]

Separate total optical depths and separate unattenuated colour integrals do
not determine this value when violet, cream, and dark-purple material changes
order along the ray. The finite-prefix `sinc` identity integrates `kappa` and
`c kappa`; it does not close the exponential of a spatially varying
cumulative trigonometric integral.

Exact small composition therefore requires one of:

1. one common source colour wherever media interpenetrate;
2. globally nonoverlapping colour intervals with a fixed order; or
3. deliberately optically thin approximate colour.

The actual head violates the first two: axes, glumes, hairs, and anthers of
different colours coexist throughout each recursive panicle. Quantising them
into horizontal colour layers produces a false vertical palette and, when
the bands are visible, a subtler volumetric version of the horizontal banding
that made F6 unacceptable. More thin layers only moves the error scale while
multiplying work.

For the optically thin approximation, the conservative bound from
`GRASS-PLUME-TRANSPORT-MATH.md` is

\[
  \|C-C_{thin}\|_\infty\le c_{max}\tau_{max}^2.
\]

Keeping this below `0.15` requires roughly `tau_max<=0.39` even with
`c_max=1`, hence alpha at most about `0.32`. Dense fluffy heads and grazing
paths exceed that optical depth. An optically thin whole-head model therefore
cannot both retain the accepted opacity and certify its colours.

This obstruction becomes stricter for generic multi-species cover: arbitrary
species palettes interleave rather than acquiring a universal global height
order. Joint cooking removes the runtime species loop but does not make
noncommuting coloured transport commute.

## 7. Depth and first-surface semantics

The spectral field returns exact transfer for an authored medium. It does not
return the categorical first hit, coupled surface normal, or material mark of
the finite glume/anther geometry that it replaces. A mean optical depth or
mean interaction distance is not a first surface and can move continuously
through empty space, recreating a floating-depth cue.

Using the nearest structural/scene event as `t_o` gives exact compositing of
the medium **in front of that event**. It does not make a plume-only pixel
opaque, nor can it correctly depth-test another finite surface that passes
through differently ordered coloured head material unless that surface is
the supplied cutoff and the medium is recomputed to that prefix.

For genuinely microscopic common-colour hairs this is the right semantic
object: filtered fuzz is transparent transfer. For `4.5--6 mm` glumes and
`1.5 mm` dark anthers visible near the camera, changing from surface events to
a depthless medium is a material-model approximation as well as a geometric
one.

## 8. Fixed cost and why a few reads do not rescue it

Assume the core uses `K_o` exact affine-extrusion fields. Two Tier-1
populations plus the existing control read cost

\[
  T_{core}=2K_o+1.
\]

The spectral coefficients can be constants/uniform data, so the medium adds
zero filtered texture reads and negligible resident bytes. Its real cost is
arithmetic and live state. For every `(layer,stratum,mode)` the exact formula
needs a phase dot product, an axial-frequency dot product, a `sinc`, a complex
phasor, coefficient products, and accumulation. Frequencies can be shared
across strata, but different clip midpoints and lengths make their phasors and
`sinc` factors different.

The number of mode-segment evaluations is

\[
  N_{eval}=2\sum_{s=1}^S Q_s.
\]

Even `S=3,Q_s=16` already means 96 mode-segment evaluations, not 16 cheap
FMAs. The constructive macro-envelope example `S=3,Q_s=225` means 1350 such
evaluations. A rectangular harmonic recurrence can reduce transcendental
calls, but it cannot remove the independent `k dot v` and coefficient
accumulations from the exact integral. Register pressure also scales with the
active accumulators and recurrence state.

A phasor lookup only exchanges this arithmetic for approximately one lookup
coordinate per independent phase. Four extra coherent texture reads do not
return hundreds of unrelated phases. A preintegrated texture indexed by
camera direction and prefix would restore the very sampled direction/origin
coordinates whose amplification and memory failures motivated this track.

The user's flexibility around nine reads is useful when one or two additional
exact core families buy a genuine invariant. It does not change the spectral
bandwidth or coloured-transport arguments. This rejection therefore does not
depend on treating nine as an absolute number.

## 9. Arbitrary-soup compiler theorem

There is a short information-theoretic reason the finite-spectrum head cannot
be a universal complete compiler target.

Take `N` uniformly spaced vertical-view sample rays at the closest promised
filter footprint. For a fixed set of at most `Q` horizontal frequencies per
stratum/layer, the candidate log-transmittance vector

\[
  (-\log T(q_1),\ldots,-\log T(q_N))
\]

lies in a real linear subspace whose dimension is at most the number of real
spectral coefficients. On the uniform samples, all reciprocal-lattice
frequencies alias to only `N` discrete Fourier columns. Hence even if the
compiler may choose the frequency set, the possible supports of size `Q<N`
form a **finite union** of proper low-dimensional subspaces.

An arbitrary triangle soup can independently vary filtered partial coverage
near each sample by placing or resizing disjoint visible micro-surfaces around
those rays. It therefore supplies an open set of target transmittance vectors
in `(0,1)^N`. A finite union of proper subspaces cannot contain that open set;
indeed some targets remain at positive distance from it.

Thus, for every fixed small `Q`, there are exterior-visible triangle-soup
heads that no such compiler can reproduce to an arbitrarily chosen small
error, even in one top view. Adding all view directions, prefix cutoffs,
colours, and depths only strengthens the result. This does not forbid a
measured `Q` approximation for a particular far-filtered community. It
forbids claiming the small spectral class as a general high-fidelity bake
target for arbitrary soups at unrestricted exterior near distance.

## 10. Relation to F4/F6

This candidate does **not** repeat F6's horizontal opaque masks or its broad
ruled head intervals:

- density has no opaque sheet intersection;
- all view directions use one analytic finite-segment identity; and
- long foliage/core events remain categorical and independent of the head
  transfer.

However, forcing the spectral budget down by using a few nonnegative ridge
terms produces stripes, and forcing colours into a few height strata produces
bands. Those are the analytic medium analogues of the visual structures that
already failed F4/F6. The remedy would be the large two-dimensional spectrum
and fine colour/depth order just shown to exceed the intended low-end fixed
cost. The prior failure is therefore accounted for, not renamed.

## 11. Decision and pre-CPU condition

No CPU gate is authorized for the proposal as stated. Its complete-head
version is red before fitting because the accepted closest exterior views
resolve finite, differently coloured, depth-owning reproductive geometry that
a small common-colour spectrum cannot contain.

The construction may be resumed only under one of two objectively different
contracts:

### A. Far-field head LOD

Declare a switch distance `D_switch` no closer than the distance at which the
source head's spikelets and colour-order landmarks are sub-pixel under the
actual anisotropic footprint (expected scale `~4--6 m`, but it must be measured,
not assumed). A CPU gate may then fit the **already filtered** arbitrary soup
and must simultaneously certify:

1. maximal source/candidate prefix-transfer error, not density RMS;
2. source-clear corridor error only until each corridor becomes unresolved;
3. a nonnegative sum-of-squares certificate;
4. colour-stratum error against the same filtered source;
5. two-layer affine composition and every exact pole; and
6. a frozen `2 sum Q_s` ALU/register budget that is demonstrably suitable for
   the low/mid target.

This still needs a separate correct near-field head representation. It cannot
by itself complete the grass shader.

### B. Common-colour microscopic residual

Keep every resolved axis, glume, lemma, spikelet body, and anther in a future
categorical core and use the spectrum only for pale callus hairs/fuzz. Then the
source-colour, depth, and closest-distance objections disappear, and the
binding gate is exactly the partial-ray mode-count test already specified in
`GRASS-PLUME-TRANSPORT-MATH.md`.

For the requested complete head, the strongest objective resume condition is
a new fixed-cost invariant that returns finite coloured successor structure
near the camera, while the spectral medium takes over only after physical
minification. More head strata, more nonnegative ridges, more sampled camera
directions, or a few extra reads are not that invariant.

## 12. Provenance boundary

- Sannikov's locally stored GameDev.ru article supplies the parallel-extrusion
  projected-ray scaling used by the granted structural core. It does not
  propose the stratified spectrum in this note.
- `GRASS-PLUME-TRANSPORT-MATH.md` supplies the resonance-free finite-segment
  identity, prefix norm, common-colour closure, and endpoint-potential no-go.
- `GRASS-INVERSE-AUTHORED-FOUR-FIELD-MATH.md` supplies the measured F4/F6
  failure and the requirement not to reintroduce broad ruled/sheet heads.
- `tools/groundcover-bake/EstonianGraminoids.ts` supplies the tile, head,
  spikelet, branching, and palette facts used here.
- The sum-of-squares construction, vertical-stratum composition, Bernstein
  bandwidth application, fixed-sample arbitrary-soup theorem, and resulting
  complete-head rejection are local derivations in this note.

No runtime code, shader, WebGPU path, or CPU/GPU gate was changed or run for
this result.
