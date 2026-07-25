# Pairwise-plane finite-ray transfer: frozen offline gate

Status: **offline experiment only; no runtime authorization**.  The currently
served reconstruction remains rejected because the user still observes the
same widening/stretching and wrong-perspective behaviour.  This note freezes a
different representation before fitting it to the actual Calamagrostis source.
Failure may not be answered by silently adding ranks, reads, a march, a second
pass, or a hardware shell.

## 1. Exact field being approximated

For the canonical periodic botanical tile, let

\[
x=(u,v,y,\phi,\theta,q)\in
\mathbb T^2\times[0,1]\times\mathbb T\times[-1,1]\times[0,1],
\qquad
q={\log(1+10L)\over\log(1+10\cdot155)}.
\]

The source oracle traces the half-open pointed segment `[0,L)` and returns the
four-sample footprint moments

\[
F(x)=(A,m,C_r,C_g,C_b,N_x,N_y,N_z).
\]

`A` is fractional coverage, `M=Lm` is the premultiplied first-distance moment,
`C` is linear-premultiplied colour, and `N` is the face-forwarded
premultiplied normal moment.  The six coordinates specify the complete
canonical straight finite ray, modulo the harmless azimuth quotient at a
vertical direction.  They do **not** specify a curved terrain-space ray or
unmodelled wind deformation; any such map needs a separate line-preservation
proof.

## 2. Frozen representation

Number coordinates `0..5` in the order above and use all fifteen unordered
pairs

\[
\mathcal P=\{(i,j):0\leq i<j<6\}.
\]

Each pair owns one `512 x 512 x RGBA16F` feature plane.  Bilinear lookup gives
`p_ij(x_i,x_j) in R^4`.  The four-channel latent is

\[
z(x)=\mathop{\odot}_{(i,j)\in\mathcal P}p_{ij}(x_i,x_j).
\]

The head receives the fixed fourteen-term quadratic lift

\[
h(z)=(z_0,z_1,z_2,z_3,z_0^2,z_1^2,z_2^2,z_3^2,
z_0z_1,z_0z_2,z_0z_3,z_1z_2,z_1z_3,z_2z_3)
\]

and applies one affine `14 -> 8` map.  There is no candidate list and no
data-dependent control flow.  This is more expressive than the rejected
global `W(u,v,phi) * E(y,theta,L)` rank-eight split: every coordinate pair can
change the latent before the joint product, so no fixed 3+3 separation is
imposed.  It is still a severe rank-four hypothesis and is not presumed to
work until the actual-source gate passes.

The physical decode is fixed and cheap:

\[
\begin{aligned}
A &= e\,\operatorname{sat}(o_0),\\
m &= A\,\operatorname{sat}(o_1),\\
C &= A\,\operatorname{sat}(o_{2:5}),\\
N &= A\,{o_{5:8}\over\max(1,\|o_{5:8}\|_2)},
\end{aligned}
\qquad e=[L>0].
\]

The gate reports raw saturation counts; saturation is not allowed to hide a
bad approximation.

## 3. Frozen GPU-shaped storage and cost

All planes are layers of one `texture_2d_array`.  Periodic coordinates are
`u`, `v`, and `phi`; `y`, remapped elevation `(theta+1)/2`, and `q` clamp.  A
mixed plane is stored with its periodic coordinate on S.  Three samplers cover
repeat/repeat, repeat/clamp, and clamp/clamp without gutters.

A complete `512..1` mip chain contains `349,525` texels per layer, hence

\[
349525\cdot15\cdot 8 = 41,943,000\ \text{bytes}.
\]

The 120 head weights plus eight biases require 512 bytes as f32, for a total
of `41,943,512` resident bytes.  This is below the current `51,121,152`-byte
ceiling.  Runtime shape is exactly fifteen statically expanded filtered
samples.  The product costs fourteen `vec4` multiplies (56 scalar
multiplications), the quadratic lift ten scalar multiplications, and the head
112 scalar FMAs: 178 FMA-equivalent operations before the small physical
decode.  There is no loop, march, extra pass, barrier, dispatch, or candidate
search.  Base-level bilinear filtering touches sixty physical texels / 480
payload bytes.  Trilinear sampling would double that to 120 texels / 960 bytes
and is **not** smuggled into the acceptance cost.

The first gate samples level zero only.  Compute shaders have no implicit
fragment derivatives, and an unproved common LOD could blur endpoint and view
discontinuities into the same kind of sheets being removed.  A later mip rule
must be derived and costed before runtime integration; the presence of stored
mips is not permission to use a vague distance LOD.

## 4. Honest fit and sampling protocol

The continuous fitting model is only an offline table generator.  Fifteen
small pair functions produce the four factors; their weights never ship.
Training uses only `train.f32`.  All declared contiguous coordinate holdouts,
general validation, exact horizontal, exact vertical, endpoint sequences,
same-line censor pairs, and the preserved 17-frame 18-degree camera path stay
unseen.

The order of gates is fixed:

1. continuous model on every unseen split;
2. sample every plane at the frozen 512-square texel centres;
3. quantise the factors to f16 and the head to f32;
4. reconstruct exclusively through a software implementation of the declared
   hardware bilinear wrap/clamp lookup;
5. evaluate the same numerical gates again and inspect numbered camera-path
   truth/reconstruction/error images.

The continuous and quantised results separately require: `A p95 <= 0.10`,
coverage IoU `>= 0.90`, `M p95 <= 0.025 m`, conditional-distance p95
`<= 0.05 m`, premultiplied-colour RMSE `<= 0.06`, normal p95 `<= 20 degrees`,
and plume recall `>= 0.95`.  Endpoint A/M decreases, right-censor errors,
vertical-azimuth invariance, periodic seams, and raw physicality are separate
hard gates.  Any widening fan, wrong-view sheet, camera-distance band, or pale
panicle disappearance in the quantised camera path is an immediate rejection
even if aggregate metrics pass.

Only a clean quantised result authorises a measured runtime experiment.  It
does not authorise replacing the current path.  Overlapping species, litter,
and moss in one fixed ecological community require **no species coordinate**:
their geometry is unioned offline before tracing, and these same six ray
coordinates query the composite marked transfer.  An extra coordinate or
separate asset is needed only when the mixture/community state itself must vary
live across ecology.  Carrying enough material/species marks through the output
payload and resolve is a later explicit layout/quality gate; this single-species
fit cannot silently declare it solved.

## 5. Sources and boundary of novelty

Primary precedent:

- Fridovich-Keil et al., *K-Planes: Explicit Radiance Fields in Space, Time,
  and Appearance*, CVPR 2023, pp. 12479-12488,
  <https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf>.
  Equations 1-2 project a d-dimensional coordinate into all `d choose 2`
  bilinearly sampled planes and combine their features with a Hadamard
  product.  Their decoder and volume-rendering task are not our contract.
- Cao and Johnson, *HexPlane: A Fast Representation for Dynamic Scenes*, CVPR
  2023, <https://arxiv.org/abs/2301.09632> and the official project
  <https://caoang327.github.io/HexPlane/>.  This is adjacent six-plane dynamic
  radiance-field evidence, not evidence that the present botanical field will
  fit.

LAAS-specific, presently novel engineering hypotheses are: applying all
fifteen pair planes to the six coordinates of a pointed finite ray; the exact
`RGBA16F 512^2 x 15` array/sampler packing; the rank-four product plus
fourteen-term quadratic head; the premultiplied `(A,m,C,N)` physical decode;
and the actual-source endpoint/censor/vertical/panicle/oblique visual gates.
The papers must not be cited as validating those choices.  This experiment is
designed to falsify them.

## 6. Actual-source result: rejected before table sampling

The frozen continuous generator was trained for 80 epochs on the declared
60,000-record training split.  Its forward path used the exact cheap runtime
saturates (with straight-through gradients only in the offline optimiser), the
analytic `L=0` mask, periodic Fourier coordinates, and exact-pole azimuth
canonicalisation.  General validation, every contiguous coordinate holdout,
horizontal/vertical rays, endpoint and censor sequences, and the entire
18-degree camera path remained unseen.

The result fails even its training records: coverage IoU is `0.1933`, A p95
error is `0.5212`, conditional-distance p95 error is `5.379 m`, colour RMSE is
`0.2389`, normal p95 error is `134.87 degrees`, and plume recall is `0.6521`.
General validation is `0.1838` IoU / `0.5572` A p95 / `6.046 m` conditional
distance p95 / `139.03 degrees` normal p95 / `0.6047` plume recall.  Exact
horizontal rays reach only `0.6079` IoU and have `36.994 m` conditional-depth
p95; exact vertical rays have zero IoU and zero plume recall.  Endpoint A
decreases above 0.01 occur on `5.985%` of adjacent steps, and censor A/M p95
change errors are `0.3195 / 0.5877 m`.

The preserved 18-degree path is `0.4897` IoU with `2.718 m` conditional-depth
p95, `132.34 degrees` normal p95, and `0.1845` colour RMSE.  Plume recall alone
reaches `0.9626`, but direct inspection of the numbered QA shows that the
panicle and blades have collapsed into broad horizontal wrong-view sheets.
This is a hard visual rejection, not a threshold near miss.

Decision: **reject this rank-four all-pairs product plus quadratic head.**  The
failure happens in the continuous field, before f16 quantisation or bilinear
table lookup, so sampling the 512-square tables would only encode a known bad
function.  No table, shader, binding, pass, loop, march, or runtime change was
made.  Artifacts are preserved under
`data/work/groundcover-endpoint-pair-planes/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/6517639e8eda451e356c71dad6f1938795d13e3723de48650e5d600749efc21d/fit-width64-epochs80/`.
The report SHA-256 is
`cb437d30e1483f23c7fca1a440b436c6e6254d1e4fa9ec3cc17ce66b7244f756`;
the QA SHA-256 is
`e9f45f10312c38e01f7c737995ba9e7b8a0e34215d18e20cde988f8130533465`.
