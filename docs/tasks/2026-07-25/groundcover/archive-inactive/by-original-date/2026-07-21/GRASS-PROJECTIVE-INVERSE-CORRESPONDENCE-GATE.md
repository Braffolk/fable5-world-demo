# Projective inverse correspondence gate

Date: 2026-07-22  
Decision: **reject and park after one corrected actual-source attempt**  
Runtime/shader changes: none

## Question

Can a tiny fixed set of canonical complete-event fields be inverted into a
held-out live view using each record's depth and geometric normal, without
blending events, intersecting live triangles, marching, or traversing a
candidate structure?

The strongest admitted form uses a closed-form rank-one projective inverse,
then a second categorical record whose exact line residual must pass. The one
permitted extension updates the plane from that record and makes one final
categorical lookup. Four directions therefore cost twelve fixed record loads;
there is no data-dependent iteration.

## Exact exterior derivation

Use the top plane `y=H`. A downward direction `d` has horizontal slope

\[
s(d)=\frac{(d_x,d_z)}{-d_y}.
\]

For live top-plane phase `q`, live slope `s`, canonical slope `s_i`, vertical
drop `h`, and canonical phase `x`, equality of the two world points is

\[
P=x+s_i h=q+s h,
\qquad
\boxed{x=q+(s-s_i)h}.
\]

This fixes the sign unambiguously. Let the canonical first-event field be
`h=D_i(x)`. The inverse is the fixed-point problem

\[
F(x)=x-q-(s-s_i)D_i(x)=0.
\]

At a canonical record with geometric face normal
`n=(n_x,n_y,n_z)`, implicit plane differentiation gives

\[
\nabla D_i=
\frac{(n_x,n_z)}{n_y-(n_x,n_z)\cdot s_i}.
\]

Writing `Delta s=s-s_i`, the Jacobian is the rank-one matrix
`I-Delta s (nabla D_i)^T`. Sherman--Morrison therefore reduces its inverse
exactly to one scalar denominator. Starting from `x_0=q`,

\[
\boxed{
h_p=
h_0\frac{n_y-n_{xz}\cdot s_i}{n_y-n_{xz}\cdot s},
\qquad
x_1=q+\Delta s\,h_p .
}
\]

The second lookup returns one complete categorical event
`(h_1,n_g,n_s,colour,coverage,mark)`. It is accepted only when

\[
\boxed{\lVert x_1-q-\Delta s\,h_1\rVert\le\epsilon}.
\]

If this residual is zero, that *finite source event*, rather than an infinite
plane extension, lies on the live ray. Colour, shading normal, coverage, and
botanical/material mark all come from the same record.

The permitted second update intersects the plane of the second record with the
live ray, obtains `x_2`, reads a third complete categorical event, and applies
the identical residual. Four directions times three reads is twelve fixed
loads. The minimum positive accepted `h` is selected branchlessly.

## What the residual does and does not prove

The residual proves collinearity of one actual canonical first event with the
live ray. It does **not** prove that this event is the live ray's first event.
If the true nearer event is absent from every converged candidate, a farther
self-consistent event passes. Therefore the gate reports three distinct
quantities:

1. **target-event visibility**: using the exact truth `h`, does any candidate
   direction see that event first at `x=q+Delta s h`? This is independent of
   the inverse algorithm;
2. **inverse-basin recoverability**: does one/two closed-form updates actually
   reach the truth event, before selection?;
3. **selected exact-event recall**: does the minimum residual-valid candidate
   equal the true first event?

## Exact limits

- **Finite triangles and disocclusion.** The inverse is exact while one planar
  first-event chart is retained. A normal cannot create a target event hidden
  in all source directions, nor provide finite-triangle containment.
- **Discontinuous basins.** `D_i` is a discontinuous first-event field. The
  local rank-one inverse may begin on a miss or on an unrelated owner even when
  the target event is visible at the exact inverse phase.
- **Grazing surfaces.** `n_y-n_xz dot s_i=0` is a canonical depth-field fold;
  `n_y-n_xz dot s=0` is live ray/plane parallelism. Neither admits the local
  inverse.
- **Exact horizontal.** The top-plane slope and phase diverge at `d_y=0`.
  A bounded dominant-axis chart can remove this coordinate pole, but it is a
  different boundary field and does not appear in the tested top field.
- **Signed/upward directions.** A top-entry field covers downward exterior
  rays only. Full signed directions require compatible dominant-axis charts
  whose overlaps return the same marked event.
- **Camera-inside.** The answer is a pointed-line successor and depends on
  origin phase. The exact first-cell split can resolve the local prefix; on a
  miss it still requires a boundary suffix field containing boundary height.
  This projective inverse neither stores nor reconstructs that fifth
  coordinate. Combining it with the prefix does not solve the suffix.

## Actual isolated-source gate

The corrected gate uses the immutable isolated production Calamagrostis shoot:

- source SHA-256
  `8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d`;
- 333,520 vertices / 353,245 triangles;
- the actual production `0.52 m` periodic lattice, not a nonperiodic isolated
  object;
- full geometric and authored shading normals, authored RGB, and the exact
  structure/plume partition;
- continuous f64 phase and events with no atlas quantisation, filtering,
  packed-normal error, or depth quantisation;
- 21,504 held-out rays at two half-bin azimuths and elevations
  `0.1/1/5/10/25/45/65/82.5` degrees;
- source directions: 16 azimuths, `5/15/35/55/75` degree rings, and one
  vertical singleton.

The sign/control gate uses identical live and canonical directions. Both the
one- and two-update paths reproduce all 868 true hits and 476 misses, with
exact-event recall `1.0`, zero residual, and maximum position error
`4.44e-16 m`. The implementation and sign are therefore not the held-out
failure.

### Strict result (`epsilon=20 micrometres`)

| fixed path | target event visible | inverse-basin recoverable | selected exact event | panicle exact | structure exact |
|---|---:|---:|---:|---:|---:|
| K=8, one update, 16 loads | 97.905% | 0.863% | 0.344% | 0.518% | 0.035% |
| K=4, two updates, 12 loads | 94.790% | 4.147% | 3.722% | 5.375% | 0.782% |

K=4 produces 792 residual-valid selected hits, but 197 are farther wrong
self-consistent events. The accepted position-error maximum is `2.952 m`
despite the `20 micrometre` line residual. This directly demonstrates why
residual validity is not first-event validity. Relaxing the residual to 1 mm
raises K=4 hit recall to 13.823% while *lowering* exact-event recall to 3.459%
and admitting 1,657 wrong self-consistent events; tolerance cannot repair the
visibility topology.

The corrected diagnosis is therefore:

- K=4 also has a real source-visibility deficit, especially for panicles
  (`93.305%` target visibility);
- the dominant loss is much larger: the two rank-one updates recover only
  `4.147%` of true events even when `94.790%` are present in the four source
  fields;
- K=8 clears aggregate target visibility but its one-step basin remains below
  one percent;
- exact horizontal and arbitrary inside-origin successors remain outside the
  representation.

This is a hard **NO-GO**. A third iteration, more directions, looser residual,
triangle candidate, or runtime search would grow the rejected family rather
than repair its missing global first-event map. No runtime shader integration
is authorised.

## Fixed resource ceiling

An optimistic eight-byte complete record can contain `depth16`, geometric
`oct16`, shading `oct16`, and a 16-bit material/coverage payload. With 81
directions and `258x258` wrapped phase records, resident storage is exactly
`43,133,472` bytes, under the `51,121,152`-byte cap. K=4/two-update uses twelve
categorical loads and can stream its current minimum without an array, loop,
march, pass, barrier, or traversal. It still needs two divisions per candidate
and has incoherent phase reads at grazing angles. This cost is recorded only
to show that rejection is geometric, not a hidden resource overrun.

The eight-byte payload is not automatically sufficient for arbitrary
continuous premultiplied plume colour plus coverage plus a large community
mark. Any future accepted codec must cost that packing honestly. Overlapping
species and moss remain one offline-composed marked community; runtime work
must not multiply by species.

## Artifact and provenance

- Tool: `tools/groundcover-bake/analyze-projective-inverse-correspondence.ts`
- Artifact:
  `data/work/groundcover-projective-inverse-correspondence/8cd69c2a6c61043c/2967319b3f513252/`
- `qa/001-heldout-10deg-k8-strict.png`: grey target miss, green/pale exact
  structure/plume recovery, orange selector failure despite basin recovery,
  red unrecovered/wrong event.

Source boundaries:

- Sherman and Morrison (1950), *Adjustment of an Inverse Matrix Corresponding
  to a Change in One Element of a Given Matrix*, supplies the rank-one inverse
  identity.
- Lin and Shum (2004), *A Geometric Analysis of Light Field Rendering*,
  supplies geometry-assisted neighbouring-ray reconstruction and its
  disocclusion boundary.
- The exact phase/sign derivation, complete-event residual, periodic
  actual-source gate, target-visibility versus basin decomposition,
  pointed-line composition limit, and fixed byte/read accounting above are
  LAAS-original work.

