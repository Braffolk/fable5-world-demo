# Candidate K2 nested-parent codec — measured RED blocker

Date: 2026-07-24  
Status: **PARKED; no runtime implementation authorised**

## 1. Why this work was done

Candidate K's direct `128 x 128` scale page passed at the 65 stored angular
nodes, but the complete held-out angular finite element later failed.  Before
that final angular result was available, K2 tested whether a true nested
positive-measure hierarchy could remove camera-centred scale rings and reduce
millimetric particle crawl while retaining:

```text
one RGBA32Uint scale operation
4 R0 + 4 R1 + 1 scale = 9 profile operations
<= 48.75 MiB resident
no runtime geometry, loop, march, candidate list, or extra binding
```

The copied-parent LOD theorem is valid.  The complete carrier is not.

## 2. Valid theorem retained

If a level-`l` record's high endpoint is the exact same quantised parent code
read as level `l+1`'s low endpoint, then

\[
\lim_{\beta\to1^-}((1-\beta)M_l+\beta M_{l+1})
=M_{l+1}
=\lim_{\beta\to0^+}((1-\beta)M_{l+1}+\beta M_{l+2}).
\]

The footprint-level switch is therefore exactly C0.  This theorem addresses
the LOD threshold only.  An ordinary nearest spatial-cell boundary retains

\[
\Delta M=(1-\beta)\Delta M_l+\beta\Delta M_{l+1},
\]

and hence retains the full fine-cell jump near `beta=0`.

## 3. Measured parent-copy versus fidelity conflict

The already-GREEN direct carrier stores both sigma-4 and sigma-16 fields at
`128 x 128`.  Across all 65 stored nodes its worst packed errors were:

| field | A p95 / p99 | premul RGB p95 / p99 | verdict |
|---|---:|---:|---|
| sigma-4 at 128 | .07059 / .09020 | .05429 / .06816 | GREEN |
| sigma-16 at 128 | .03922 / .04314 | .03675 / .03968 | GREEN |

Source: `data/work/groundcover-candidate-k-direct-fe/e3e0a4175b151b89/6e8ba84eda853dab/report.json`.

The memory-exact nested attempt was

```text
level 0: 128 x 128
level 1:  16 x 16
scale allocation: 32.5 MiB
near + scale:      48.75 MiB
```

For copied-parent continuity, the sigma-16 endpoint embedded in every
`128-square` child must be copied from the corresponding `16-square` parent.
On the first accepted direction node, that parent field already failed:

| field | A p95 / p99 | premul RGB p95 / p99 | connected | verdict |
|---|---:|---:|---:|---|
| sigma-4, 128-square | .06667 / .08627 | .05274 / .06234 | 0 | GREEN |
| sigma-16, 16-square parent | .13058 / .16786 | .09954 / .12577 | .0305% | **RED** |

Frozen p95 limits are `.08` for coverage and `.06` for premultiplied RGB.
Because every stored node is binding, one accepted-node counterexample is
sufficient to reject the representation.  Source:
`data/work/groundcover-candidate-k2-pyramid/e3e0a4175b151b89/3dd125e8eb7dd371/report.json`.

This is not a poor choice of hierarchy length.  It is a capacity conflict:

1. a copied parent at resolution `N_p` has at most `N_p^2` independent phase
   values and is block-constant when duplicated into the fine page;
2. the accepted sigma-16 field needs materially more phase freedom than
   `16^2` under the frozen fidelity gate;
3. keeping sigma-16 natively at `128^2` breaks copied-parent identity unless
   the next level is also available at that rate; and
4. another `128-square RGBA32Uint` level costs 32 MiB, taking the resident
   total far beyond the ceiling.

The 128-bit scale record has no spare endpoint: its eight 16-bit symbols are
already `4 angular vertices x 2 footprint endpoints`.  A third endpoint cannot
be recovered algebraically from those bits for an arbitrary positive field.

## 4. Other nested chains and temporal result

The preregistered `96->48->24->12->6->3`, `112->56->28`, and
`120->40->20->10->5` chains were also screened after correcting the oracle to
**filter the positive field first and spatially reduce second**.  Even the raw
nearest variants were RED on node zero; their worst premultiplied-RGB p95
ranged from `.07835` to `.09665` in the corrected screen.  Analytic promotion
to a common parent at spatial boundaries made static fidelity worse, because
it replaces the desired fine-footprint measure by a coarser one over a large
fraction of every cell.

The budget-exact raw attempt also failed the frozen millimetric excess-change
gate at its sigma-16 parent: at 4.5 mm diagonal translation, coverage p95 was
`.17064`, premultiplied-RGB p95 `.11995`, and the connected joint exceedance
was `2.275%`.  Limits were `.04`, `.03`, and `<1%` respectively.  These are a
diagnostic counterexample, not the principal park reason; the static
parent-copy conflict is already conclusive.

An earlier smoke artifact at recipe `f3e844eb84f98f46` is invalid because it
mistakenly treated raw spatial cell averaging as the footprint filter.  It is
retained only for provenance and must not be cited as evidence.

## 5. Independent angular blocker

K2 changes only spatial scale.  The corrected arbitrary-angle Candidate-K gate
is independently RED: unlimited-precision four-corner angular FE failed
`212/224` held-out direction-by-scale cases, with worst p95 coverage `.667`,
premultiplied RGB `.490`, and connected error `.701`.  Therefore no successful
spatial hierarchy could authorise this complete carrier without first replacing
the angular representation.

This is recorded in
`docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-K2-THEOREM-AUDIT.md`.

## 6. Lightweight analytic NDF check retained

The unresolved-lighting derivation is mathematically independent of the RED
spatial/angular codec.  For `c in [0,1]`, define

\[
F(c)=\sqrt{1-c^2}+c\arcsin c.
\]

For a uniform normal ring around growth axis `u`, direct integration gives

\[
Z_r=E|n\cdot v|={2a_v\over\pi},\qquad
N_r=E(|n\cdot v||n\cdot l|)={a_va_l\over\pi}F(c_\delta).
\]

The endpoint checks are exact:

```text
parallel projected directions: c_delta=1, F=pi/2, N_r=a_v*a_l/2
orthogonal projections:        c_delta=0, F=1,    N_r=a_v*a_l/pi
```

For a uniform spherical plume distribution,

\[
Z_p=1/2,\qquad N_p={2\over3\pi}F(|v\cdot l|),
\]

which yields `N_p=1/3` for `v=l`, as required by
`E[(n dot v)^2]=1/3`.  With non-negative mixture weights,

\[
D(v,l)={\sum_i w_iN_i\over\sum_i w_iZ_i}
\]

lies in `[0,1]` because every integrand satisfies
`|n dot v||n dot l| <= |n dot v|`.  The denominator must be kept non-zero by a
visible ring or plume component at axial degeneracies.  These identities are
sound and reusable, but they do not rescue the rejected codec and do not
authorise a lighting implementation.

## 7. Resume condition

Do not resume K2 by adding levels, widening thresholds, or spending another
read.  Resume only if a new angular-and-spatial representation simultaneously
provides:

1. held-out arbitrary-angle fidelity under the existing limits;
2. a spatially continuous or translation-GREEN positive field;
3. one fixed scale operation and at most nine profile operations total; and
4. resident memory at or below the agreed ceiling.

The copied-parent endpoint theorem and analytic NDF integrals may be reused
inside such a representation.  The present hierarchy is parked.
