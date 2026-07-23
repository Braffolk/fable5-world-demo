# Complete ground-cover model boundary

Date: 2026-07-23
Scope: pure mathematics and product contract; no runtime/shader change
Status: **superseded in part by the user's box-volume fade relaxation.** The
finite-state bound remains valid, but the pointed-origin obstruction now has a
constructive common-slab solution. The active model is
`GRASS-BOX-BOUNDARY-TRANSFER-MATH.md`.

## 1. The question this note closes

The desired system combines all of the following:

1. any periodic triangle soup is valid cook input;
2. every exterior pointed ray is supported, including an origin in an air gap
   inside the botanical height range and exact horizontal/vertical directions;
3. near-field resolved colour, silhouette, ordering, depth and normal remain
   faithful rather than becoming a volume or view blend;
4. runtime work is a tiny fixed lookup/transform path: approximately 9--14
   reads, no loop, march, traversal, event candidate list, runtime geometry or
   per-species query;
5. resident ground-cover data remains below 250 MB; and
6. two independently transformed Tier-1 layers, affine wind, overlapping
   species and later moss use the same representation.

The word *unconditional* matters.  It would mean that every admitted triangle
soup receives the promised fidelity rather than merely being accepted by a
compiler which may fail a quality gate.

## 2. Finite-state lower bound for arbitrary soup appearance

Let a compiler map a soup `S` to at most `B` bits and let the live query use
that state.  Put `M` independently coloured, pixel-resolvable patches in one
tile.  Each patch has either colour `c0` or contrasting colour `c1`; an
exterior near-field ray can isolate every patch.  There are `2^M` different
soups.  Any result whose error is less than one quarter of the colour contrast
must distinguish all of them, hence

```text
2^B >= 2^M, therefore B >= M.
```

An arbitrary soup has no fixed upper bound on `M`.  Consequently no fixed
250 MB representation can give an unconditional high-fidelity guarantee for
every arbitrary soup.  Physical distance filtering does not remove this
counterexample because all patches can be placed at the declared closest
exterior distance and made larger than its pixel footprint.

This does **not** prevent arbitrary soups from being valid input.  It proves
that source-to-runtime fidelity must be a measured rate-distortion result, or
the admitted source bandwidth/complexity must be bounded.

## 3. Independent pointed-successor lower bound

Along one oriented line place `M` opaque intervals separated by air gaps.
Put the camera origin in each gap.  Every origin is outside triangle matter,
but every origin has a different next surface:

```text
V(o_j,d) = (t_j, normal_j, colour_j, owner_j).
```

Therefore exterior first visibility is a pointed-line problem.  Direction and
line phase alone do not determine the answer; the origin coordinate cannot be
discarded.  A fixed event tail fails when `M` exceeds its rank.  A sampled
camera direction moves the missing information into coherent angular sectors.
Blending depth, normal, colour or owner produces a record belonging to no
surface.

An exact visibility DAG can encode a *particular finite soup*, but the existing
measurements do not fit the frozen product shape: exact predicate lookup is
estimated around 17--27 serial reads, existing incomplete pages are already
219.5--357.5 MB, and the query is a data-dependent traversal.  A small increase
above nine reads does not close those three gaps.

## 4. Why the known alternatives do not evade the bound

- Sannikov's small field is exact for a parallel-extrusion invariant.  His
  arbitrary-model experiment restores sampled view dimensions and, by his own
  description, stretches.  It is not a counterexample to the pointed-line
  bound.
- Full direction/frame fields accept arbitrary soup but the measured
  one-record shell reconstruction is multimodal: strict nine-read alignment
  gives IoU `0.61974`, RGB p95 `0.36095`, geometry p95 `12.88498 m`, and large
  temporal class changes.  More angular rows do not select the missing owner.
- Low-rank five-dimensional fields, pairwise planes, codebooks and direct
  radiance fields were measured against the actual source.  They exchange
  geometric stretching for wrong-view sheets, blur or missing categorical
  topology.
- Finite interval relays have no universal rank.  Exact-horizontal air-gap
  rays give an open-set `K+1` counterexample for every finite `K`; packing the
  ranks merely turns the work into the forbidden candidate list.
- A finite analytic medium is exact for its authored density, but a whole
  resolved Calamagrostis head needs hundreds to thousands of modes, cannot
  preserve open air corridors, cannot exactly compose its interleaved
  violet/cream/green owners, and has no categorical surface depth/normal.
- F4/F6 affine unions are direction analytic but fail compilation fidelity;
  the extra four F6 reads localise some head mass without preventing wrong
  periodic successors.

These are different manifestations of the same missing information, not
independent shader bugs.

## 5. The incompatibility theorem

No model can simultaneously provide all three, unconditionally:

1. arbitrary triangle soups with guaranteed resolved exterior appearance;
2. fixed approximately 9--14-read, low-ALU reconstruction with no traversal,
   candidate enumeration or runtime geometry; and
3. a fixed resident-memory ceiling.

The finite-state construction disproves the guarantee for arbitrary visual
complexity.  The air-gap construction independently disproves a universal
fixed-rank successor.  This is a product-contract contradiction, not an
unfinished reconstruction formula.

## 6. Strongest complete implementable contract

The following contract is mathematically honest and keeps arbitrary soup as
input:

1. **Input generality:** every triangle soup is accepted by the offline
   compiler; species are unioned before compilation, so runtime never loops
   over species.
2. **Frozen physical bandwidth:** declare closest supported exterior distance,
   pixel footprint, finite horizon and colour/depth tolerances.  The source is
   filtered through that same footprint before comparison.
3. **Bounded compiled class:** the compiler targets a fixed union of exact
   direction-analytic categorical extrusion fields plus a finite-prefix
   nonnegative medium used only for common-colour unresolved fuzz and far LOD.
   Two Tier-1 affines are queried separately and composed categorically/
   optically as appropriate.
4. **Exact reconstruction:** every asset which passes compilation is queried
   exactly for the compiled categorical fields at horizontal, vertical and
   air-gap origins.  The medium integral is exact for every prefix and opaque
   cutoff.  Affine wind is removed analytically before both queries.
5. **Conditional source fidelity:** triangle-soup-to-compiled fidelity is a
   compulsory exterior image/geometry gate.  A soup may fail.  It may not be
   silently blurred, stretched, converted to a broad sheet, or called
   supported merely because it cooked.
6. **Resource closure:** the compiler rejects any result exceeding the chosen
   fixed field/mode/read/byte envelope.  Raising reads from nine to roughly
   10--14 is allowed only when it adds an independent exact field and the
   complete measured path remains low/mid-GPU suitable.

This is complete as a **compiler contract**, not an unconditional promise that
every arbitrary soup is compressible.  The currently accepted Calamagrostis
does not yet pass it: F4/F6 fail the categorical structural gate, while the
whole-head spectral replacement is analytically over budget.  Claiming that it
does pass would contradict the measured artifacts.

## 7. Required product decision

To obtain an implementation rather than another false model, exactly one
upstream requirement must move:

- **keep arbitrary-soup fidelity:** permit a measured fixed-depth visibility
  traversal and its actual read/memory cost;
- **keep the tiny runtime:** accept compile-time rate-distortion gating and a
  bounded proxy class, with some arbitrary soups rejected for quality; or
- **keep both for ordinary exterior cameras:** remove air-gap origins inside
  the cover support from the quality domain and use a full exterior light
  field, fading that whole interior domain.

The user subsequently selected a precise form of the third relaxation: a
carrier lane may fade while the camera lies inside the deliberately guarded
box volume containing that plant/overlap component. Enlarging each lane to a
common slab `P x I` makes first entry an exact three-dimensional periodic
first-passage query; its boundary point then seeds a complete four-dimensional
whole-community transfer. This removes the pointed-successor obstruction
without a live traversal. The finite-state bound still requires physical
filtering plus an actual-community error/byte gate under 250 MB. Runtime work
remains held: the periodic point-ray factorization passed, but the first
finite macrobrick codec did not establish a legal under-250-MiB filtered
representation. Moreover, filtered pixels depend on camera standoff even
when their central `(q,d)` is identical, so a successor must either store the
finest supported 4D field and filter it live or explicitly charge a finite
standoff/covariance family. The current runtime is not authorized.

## 8. Provenance

The construction/lower-bound ledger is in:

- `GRASS-EXTERIOR-VISIBILITY-COMPLEX-MATH.md`;
- `GRASS-PACKED-INTERVAL-RELAY-MATH.md`;
- `GRASS-STRATIFIED-SPECTRAL-HEAD-MATH.md`;
- `GRASS-INVERSE-AUTHORED-FOUR-FIELD-MATH.md`;
- `GRASS-EXACT-REPRESENTATION-THEORY.md`;
- `GRASS-MATHEMATICAL-EXPLORATION-SUMMARY.md`; and
- `GRASS-STATUS-AND-ISSUES.md` Experiments 18--50.

No theorem here is attributed to Sannikov.  His published invariant-ray lift
is retained for the exact extrusion subproblem; the finite-state and pointed-
successor bounds are local project derivations.
