# Class-E exterior ground cover — CPU gate result

Date: 2026-07-22
Status: **RED — structural field budget**
Runtime/shader changes: **none**

## 1. Question tested

The corrected Class-E specification selected an opaque field of the form

```text
G_f = A_f(M_f x I_f)
```

and allowed at most nine total samples for two global anti-tiling layers plus
all non-opaque work. Before building a baker or shader, the CPU gate asked
whether the accepted Calamagrostis and dense Sphagnum sources can compile into
few enough finite affine fields at plausible geometric error.

This was the first decisive implementation prerequisite. Passing the affine
ray algebra alone is not sufficient: the accepted plant must fit the field
and read budget.

## 2. Source binding and new offline metadata

The deterministic source generators now retain cook-only primitive recipes.
No mesh geometry changed.

- Accepted production Calamagrostis remains exactly `2,049,985` vertices and
  `2,171,134` triangles, with mesh SHA-256
  `37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0`.
- Its `270,541` stable recipes partition every source triangle exactly once
  and retain the real blade curves, tube centre lines/radii, glumes, lemmas,
  anthers, true callus hairs, ownership, and cap/root semantics.
- The dense Sphagnum source remains exactly `25,397` vertices and `41,184`
  triangles, with mesh SHA-256
  `ec8612e197dcc9a801c6146059cf8f85b7549ef33ac189252e91c8efbf01caa7`.
  Its `1,405` recipes also partition every triangle exactly once.
- Only true callus-hair ribbons are routed to the Calamagrostis plume
  candidate. Crisp panicle axes, glumes, lemmas, and anthers are not hidden in
  a broad colour-based plume class.

## 3. Algebra sanity gate — GREEN

`ClassEExteriorMath.test.ts` independently checks the renderer-free equations.
All seven tests pass:

- affine reduction and `t=t_a+rho/s_p` agree with an independently solved
  finite transformed cylinder;
- the exact projected-axis pole uses categorical occupancy, with no epsilon;
- exact horizontal world rays remain defined;
- a fixed minimum elects the correct field across owner-order swaps;
- tangent-line reintersection is exact inside a straight same-segment cell;
- root-local affine wind fixes the root and gives height-proportional motion;
- the closed-form Fourier plume integral matches independent dense quadrature.

This establishes that the ideal per-field algebra is sound. It does **not**
establish that the plant fits a small field union.

## 4. Structural theorem used by the CPU gate

On a flat ground chart, every field `A_f(M_f x I_f)` has one finite axial
interval and therefore at most two horizontal world cap planes. Every retained
crisp primitive endpoint must lie within the allowed compilation error of one
of those planes. The exact one-dimensional interval-piercing number of the
authored endpoint heights, divided by two, is therefore a lower bound on field
count.

This bound is deliberately optimistic:

- either endpoint may use either cap;
- root/tip pairing is ignored;
- axis, taper, curvature, width, owner, support, mask conflicts, attributes,
  plume, minification, and the reproductive head are ignored;
- every non-field cost is compressed into one fictional sample.

It is nevertheless catalogue-, assignment-, and subdivision-independent for
the current `M x I` representation. Splitting a curve cannot erase the
original outer endpoints, and each new finite piece still needs cap support.

## 5. Binding result

For only the `102` blades and `6` culms in the accepted six-shoot production
community, while omitting the entire reproductive head:

| endpoint tolerance | minimum opaque fields | optimistic two-layer reads |
|---:|---:|---:|
| 0.5 mm | 49 | 99 |
| 1 mm | 41 | 83 |
| 2.5 mm | 30 | 61 |
| 5 mm | 21 | 43 |
| 10 mm | 14 | 29 |
| 20 mm | 9 | 19 |
| 50 mm | 5 | 11 |
| 100 mm | 3 | 7 |

The hard ceiling is nine total reads. The lower bound still exceeds it at
`50 mm`, before axis compatibility or any flower-head field is charged.

The isolated one-shoot diagnostic gives the same conclusion in more detail.
At `10 mm`, its `17` blades plus one culm have a global cap-plane lower bound
of `7` fields / `15` reads. For the emitted four-axis catalogue, the coupled
axis-and-interval lower bound is `83` fields / `167` reads; a constructive
grouping needs `122` / `245`. At `20 mm` the fixed-catalogue lower bound is
`19` / `39`; at `50 mm`, `8` / `17`. Only at `100 mm` does that lower bound
reach `4` / `9`, while the constructive grouping still needs `10` / `21`.

`100 mm` is not a slightly looser grass tolerance. It is roughly one tenth of
the whole plant height and many times the `3–6 mm` authored blade width. It
would destroy the individual silhouette fidelity the user explicitly requires.

## 6. Dense moss result

Sphagnum stems and cores fit one vertical direction. Its `1,228` short,
near-horizontal curved branches are explicit analytic-medium candidates, not
credited as opaque Class-E axes. They cannot be called accepted until a
continuous source medium and partial-ray error gate exist. The tall-grass
lower bound is already RED without relying on this unresolved moss route.

## 7. Standards audit

The standards are not merely a little too strict.

- Relaxing from `5 mm` to `10 mm` still leaves `29` optimistic reads.
- Relaxing to `20 mm` still leaves `19`.
- Even `50 mm`, already botanically destructive, leaves `11` before the head.
- The first cap-plane non-refutation appears only at `100 mm`, and the more
  complete fixed-catalogue construction still fails there.

The failure is upstream in the representation: one common `I_f` makes finite
per-primitive endpoints consume cap planes/fields. Phase or angular texture
resolution, PCF, tangent correction, atlas compression, and shader tuning
cannot change this lower bound.

## 8. Decision and objective resume condition

Do **not** implement this Class-E `M x I` design in the runtime.

Return to mathematics. A successor proposal must let one fixed-cost record
represent phase-dependent finite endpoints while preserving the correct next
eligible event, without a march, loop, candidate list, per-species work, or
runtime geometry. It must prove that rejecting a nearer interval-ineligible
mask event cannot hide a later eligible event. Alternatively, the user would
have to explicitly raise the read ceiling or accept approximately `100 mm`
botanical deformation; neither is inferred here.

Two other gates remain independently unresolved for any successor:

- a static intrinsic `TANGENT_SAFE(q,omega)` classification cannot by itself
  certify live clip/support state; final safety also depends on the interval
  category `chi`;
- the plume source is opaque microgeometry, not a defined continuous
  extinction density. Either author a continuous kernel per true fuzz
  primitive or call the fitted result empirical filtered transfer rather than
  a certified line-integral approximation.

## 9. Reproduction and artifacts

Commands:

```text
node --import tsx --test tools/groundcover-bake/ClassEExteriorMath.test.ts
node --import tsx --test tools/groundcover-bake/EstonianGraminoids.test.ts
node --import tsx --test tools/groundcover-bake/SphagnumCapillifolium.test.ts
node --import tsx tools/groundcover-bake/analyze-class-e-structural-feasibility.ts
```

All focused suites pass (`7/7`, `7/7`, and `4/4` respectively), and the final
analyzer run exits successfully.

Final artifact:

```text
data/work/class-e-structural-feasibility/
  093984f242f07f37/8da8f737e15dd2ca/
```

- `metrics.json` SHA-256:
  `8f20e70fcb51711c60f5a2a933c9895f49e473f1a3f03b6da12cbfccccaaaf7c`
- source-set SHA-256:
  `093984f242f07f3788a6d27ade12882b49588f4f385790487413d2ff45c333f8`
- recipe SHA-256:
  `8da8f737e15dd2ca20b7fe82c28c357df5b30351f383662acc61c1d629db756c`
