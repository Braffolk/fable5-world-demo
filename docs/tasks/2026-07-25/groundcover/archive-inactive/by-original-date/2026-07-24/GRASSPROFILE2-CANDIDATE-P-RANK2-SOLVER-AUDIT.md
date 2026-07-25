# Candidate P rank-two capacity audit

Date: 2026-07-24

Scope: pure mathematics and offline capacity certification.  This note does
not authorise runtime, shader, format, read-count, or memory changes.

## 1. Verdict

The corrected v3 gate is a valid measurement of one particular fit, but its
narrow RED result is **not decision-grade evidence against Candidate P's
written representation**.

There are two independent reasons:

1. the 137-plane search is neither the global channel-weighted
   \(L_\infty\) affine-two-flat solver nor a lower-bound certificate; and
2. it freezes the 64 by 64 triangular finite-element coefficients to exact
   values at the phase vertices.  Candidate P fixes the finite-element
   *basis*, not those coefficients.  A cook is allowed to choose coefficients
   jointly to minimise error at vertices, triangle interiors, and translated
   sites.

The present evidence does establish useful localisation:

- the static phase basis is not the main failure: the fixed-coefficient
  phase-only control passes all 48 static cases;
- the only static misses are sigma-4 in the 10--18 degree standing cell;
- the current translation miss contains a phase-coefficient component, since
  even the unrestricted-angular fixed-coefficient control misses one 4.5 mm
  sigma-4 case;
- chord ordering, positivity, canonical shared edges, and packing cannot have
  caused the current RED, because the oracle relaxed all of them; and
- the large measured private-edge mismatch makes the later canonical-chord
  stage a serious risk, but it is not the cause of this oracle's current miss.

Candidate P should therefore receive one bounded capacity-solver attempt using
the existing truth.  A constructive GREEN is conclusive.  A narrow RED without
one of the lower-bound certificates in section 7 remains inconclusive.

## 2. Exact coefficient-space problem

For one phase vertex and angular triangle, assemble the ten sampled joint
two-scale responses as rows of

\[
  X\in\mathbb R^{m\times 8},\qquad m=10,
\]

with channels

\[
  (R_4,G_4,B_4,A_4,R_{16},G_{16},B_{16},A_{16}).
\]

Let the frozen per-channel tolerances be

\[
  t=(.06,.06,.06,.08,.06,.06,.06,.08),
\]

and define \(Z_{sc}=X_{sc}/t_c\).  The favourable unlimited-precision
per-vertex oracle is the affine Chebyshev rank-two problem

\[
 \min_{a,U,V}\|Z-\mathbf 1a^T-UV\|_\infty,
 \quad
 a\in\mathbb R^8,
 U\in\mathbb R^{m\times2},
 V\in\mathbb R^{2\times8}.
\]

This formulation permits arbitrary signed coordinates and states, just like
the v3 oracle.  It is bilinear and non-convex, but each block subproblem is a
linear programme.

The current 137 candidates do not solve this problem.  In particular:

- a Chebyshev-optimal plane need not be a PCA plane or pass through three
  observations;
- after choosing a plane, the script takes Euclidean orthogonal projections,
  whereas the granted oracle requires the closest point under the weighted
  \(L_\infty\) norm; and
- minimising each vertex's maximum normalised residual is only a surrogate for
  the final spatial p95/p99 and translation objectives.

Consequently the search can prove existence when it happens to construct a
GREEN result, but its RED result cannot prove non-existence.

## 3. Exact periodic triangular-FE shift operator

Let \(C[z,x]\) be any channel or latent coefficient on the periodic 64 by 64
phase lattice.  For an offset in lattice units

\[
  \delta=(o_x,o_z),\quad
  i_x=\lfloor o_x\rfloor,\ i_z=\lfloor o_z\rfloor,\quad
  u=\operatorname{fract}(o_x),\ v=\operatorname{fract}(o_z),
\]

the exact evaluation operator \(T_\delta\) for the existing diagonal from
\((0,0)\) to \((1,1)\) is, with periodic indices:

For \(v\le u\),

\[
(T_\delta C)[z,x]
=(1-u)C[z+i_z,x+i_x]
+(u-v)C[z+i_z,x+i_x+1]
+vC[z+i_z+1,x+i_x+1].
\]

For \(v>u\),

\[
(T_\delta C)[z,x]
=(1-v)C[z+i_z,x+i_x]
+uC[z+i_z+1,x+i_x+1]
+(v-u)C[z+i_z+1,x+i_x].
\]

Thus every static interior-phase and translated prediction is a known sparse
linear function of the phase-vertex coefficients.  Fixing the coefficients to
sampled vertex truth, as v3 does, is only nodal interpolation; it is not the
best approximation in this basis.

For a collection of offsets that stay on a fixed half of the phase triangle,
\(T_\delta\) is a circular convolution.  Its Fourier symbol is

\[
H_\delta(k_x,k_z)=e^{i(k_xi_x+k_zi_z)}
\begin{cases}
(1-u)+(u-v)e^{ik_x}+ve^{i(k_x+k_z)},&v\le u,\\
(1-v)+ue^{i(k_x+k_z)}+(v-u)e^{ik_z},&v>u.
\end{cases}
\]

For target pages \(Y_j\) at offsets \(\delta_j\), the exact unconstrained
least-squares coefficient initializer is therefore

\[
  \widehat C(k)=
  \frac{\sum_j\overline{H_j(k)}\widehat Y_j(k)}
       {\sum_j|H_j(k)|^2},
\]

with regularisation only at genuinely null modes.  This is not the final gate
objective; it is a globally optimal, cheap initializer which removes the
known error caused by blindly fixing coefficients to vertex truth.

For weighted Chebyshev or CVaR loss, the same coefficient solve is a sparse
linear programme whenever the angular two-flat coordinates or states are held
fixed.

## 4. Strongest bounded practical solver

The proposed solver consumes the same deterministic truth directions, phase
sites, and translations already used by v3.  It requests no new rays.

### 4.1 Initialisation

1. Solve the unrestricted-angular phase-only coefficient field by the Fourier
   least-squares formula above, channel by channel.
2. Refine it with a sparse weighted Chebyshev or CVaR linear programme over all
   static and translated observations.
3. Initialise angular two-flats from multiple sources: the current best 137
   candidate, joint PCA, separate-scale PCA, robust PCA, and deterministic
   Grassmann perturbations of those planes.

### 4.2 Alternating Chebyshev solve

At fixed phase vertex and fixed state matrix \(V\), solve jointly for
\(a,U,\epsilon\):

\[
-\epsilon\le Z_{sc}-a_c-u_s^Tv_c\le\epsilon
\quad\forall s,c.
\]

At fixed \(U\), solve the corresponding linear programme for
\(a,V,\epsilon\).  Apply a QR or SVD gauge normalisation between blocks so the
same two-flat is represented at controlled scale.  Each accepted block step
must monotonically decrease the exact objective.

Then lift the solve from independent vertices to all phase vertices.  The
predictions at the sampled phase offsets are \(T_\delta C\), so with one
factor block held fixed the full page and translation residuals remain affine
in the other block.  Alternate global sparse LP blocks rather than fitting a
vertex and only afterwards discovering that interpolation failed.

The translation residual for coefficient field \(C\) is explicitly

\[
  (T_{\delta+\Delta}C-T_\delta C)
  -(Y_{\delta+\Delta}-Y_\delta),
\]

so translation stability is part of the fit, not a post-hoc metric.

### 4.3 Percentile objective

Exact p95 is a cardinality objective, not a norm.  Use this hierarchy:

1. optimise a CVaR-95 surrogate first; a CVaR GREEN is a sufficient
   certificate for the percentile gate;
2. if unnecessarily conservative, use a deterministic trimmed active-set
   solve: choose the worst allowed 5 percent as outliers, solve the remaining
   sparse Chebyshev LP, update the outlier set, and retain the best exact-score
   result over deterministic restarts;
3. always decide constructive GREEN using the frozen gate's exact p95, p99,
   connected-region, and translation evaluation, never the surrogate value.

This remains an offline cook/capacity operation.  It adds no runtime reads,
bytes, ALU, branches, or passes.

## 5. Required ablations and what they decide

Run exactly three nested models, in this order, using the coefficient
optimisation above:

### A. Phase basis only

Remove the angular rank restriction and optimise the 64 by 64 coefficients
jointly against all phase and translation samples.

- RED after a globally solved convex coefficient problem means phase64 is a
  real blocker.
- GREEN means the present one-case phase-only translation RED was caused by
  the fixed-coefficient choice, not by the basis.

### B. Independent two-flats per spatial scale

Fit one affine two-flat in \(\mathbb R^4\) for sigma4 and another in
\(\mathbb R^4\) for sigma16, with optimised phase coefficients.

- If B is GREEN and joint model C is RED, sharing one pair of angular
  coordinates across the two scales is the bottleneck.
- If sigma4 is RED even independently, the standing-cell angular rank is the
  bottleneck rather than cross-scale sharing.

This is a diagnostic relaxation only; it is not automatically a legal codec.

### C. Candidate P's joint two-flat

Fit the shared affine two-flat in \(\mathbb R^8\) with the full joint phase and
translation objective.  This is the actual favourable capacity oracle.

- exact-score GREEN authorises the next cook-side canonical-chord fit;
- narrow heuristic RED remains inconclusive;
- certified RED parks the two-event premise.

## 6. What the present evidence says about the likely cause

The most likely ordering is:

1. **solver and fixed phase coefficients**, because both are known departures
   from the favourable mathematical oracle and the margins are narrow;
2. **shared two-scale rank**, if the independent-scale ablation passes while
   the joint solve does not;
3. **phase64**, only if the globally optimised phase-only solve is RED; and
4. **chord topology**, not as a cause of the current RED but as the next major
   implementation risk after oracle GREEN.

The current test cannot blame chord topology: it never imposed chords.
Conversely, the large private shared-edge mismatch warns that oracle GREEN
will not by itself prove the real three-positive-state canonical codec GREEN.

## 7. RED certificates and their limits

The alternating solver supplies an upper bound.  Failure to find a solution is
not a lower bound.  The following certificates use the existing truth only.

### 7.1 Spectral lower bound

For the channel-scaled, row-centred matrix \(HZ\), any affine rank-two
approximation with entrywise error at most \(\epsilon\) obeys

\[
 \epsilon\ge
 \frac{\sqrt{\sum_{k\ge3}\sigma_k(HZ)^2}}{\sqrt{8m}}.
\]

This is rigorous but often weak.

### 7.2 Minor lower bound

Choose one reference direction, three other directions, and three channels.
Let \(D\in\mathbb R^{3\times3}\) contain the three scaled response
differences.  An exact affine two-flat makes \(D\) rank at most two.  If every
original response entry is within \(\epsilon\), each difference is perturbed
by at most \(2\epsilon\), hence

\[
  \sigma_{\min}(D)\le\|E\|_2\le 6\epsilon,
  \qquad
  \epsilon\ge\sigma_{\min}(D)/6.
\]

Maximise this bound over all sample and channel subsets.  Interval determinant
optimisation can sharpen difficult vertices without new ray truth.

These bounds certify a per-vertex minimax error.  They do **not** alone certify
a final image p95 failure, because the bad direction/channel can vary by
vertex and phase interpolation can cancel errors.  A decision-grade p95 RED
must additionally prove that more than five percent of pixels in one frozen
case exceed threshold under every feasible coefficient field.  That requires
either a case-specific mixed-integer/cardinality relaxation or a spatially
aggregated lower bound.  Use it only if the practical solver remains narrowly
RED; do not mistake local-fit exhaustion for such a proof.

## 8. Go/no-go rule

One bounded attempt is justified because the present miss is small and the
current search does not optimise the written degrees of freedom.

Proceed to the canonical-chord cook only if model C constructs an exact-score
GREEN result for every frozen static and translation case.  Quantisation,
packing, and runtime work remain forbidden until then.

Park Candidate P if either:

- model A is RED under the globally solved convex phase-only problem; or
- model C is RED and a valid lower-bound/cardinality certificate proves the
  frozen gate cannot pass.

If A and B are GREEN but C remains narrowly RED without a certificate, report
the shared-rank model as empirically unresolved rather than mathematically
refuted.  Given the already measured canonical-edge mismatch, an engineering
decision may still park it, but that must be labelled a risk/effort decision,
not a capacity theorem.

## 9. Correction to the canonical Candidate P record

Section 10.4's phrase "globally certified robust fit" is too broad to guide a
resume.  The concrete resume condition is now:

1. optimise the shared 64 by 64 phase-vertex coefficients jointly over the
   static interior-phase and translation truth;
2. solve the joint sigma4/sigma16 affine-two-flat problem with the block LP
   method above and deterministic multi-starts;
3. evaluate the constructed field with the unchanged exact frozen gate; and
4. treat GREEN as constructive proof, but treat RED as final only with a valid
   lower-bound/cardinality certificate.

This is the strongest decision available without new ray directions, extra
runtime reads, or extra resident memory.
