# Resolved periodic-top-XZ validation track

## User-visible objective

Bake a repeating multi-cushion carpet into direction slices that all address the same canonical top-plane XZ tile, with deterministic GCRP/v2 output and CPU ray/triangle agreement.

## Effort and reusable result

Two real localhost Chromium/Apple Metal WebGPU attempts were used, the track's declared limit. Reusable code now includes:

- runtime-compatible top-plane XZ projection and reconstruction of ray parameter `t`;
- conservative integer periodic-copy derivation from mesh bounds, mesh height, ray slope, and tile size (not a fixed neighborhood);
- double-sided ray-facing normal orientation;
- GCRP/v2 periodic metadata, parser, deterministic RGBA16 packer, and wrapped one-pixel gutters;
- non-grid overlapping rounded-cushion validation mesh;
- periodic address, copy-bound, gutter, v1 regression, and v2 round-trip tests;
- selected-pixel CPU Moller-Trumbore oracle over every derived periodic copy.

## Exact blocker

Attempt 1 compared 389 GPU/CPU hit pixels and 3 misses, with zero hit/miss ownership mismatches. Exactly one hit selected a different depth/normal surface: maximum `t` error `0.13544333259743552`, minimum normal dot `0.7398475352224524`; the remaining samples account for a mean normal dot of `0.9993312212568604`.

Attempt 2 required nearest-hit barycentric edge margin `>= 0.03`, excluding 64 triangle-boundary samples. It reproduced the same outlier among 325 strict-interior hits and 3 misses, again with zero hit/miss ownership mismatches: RMS `t` error `0.0075130757564193015`, maximum `0.13544333259743552`, mean normal dot `0.9991995241309419`, minimum `0.7398475352224524`.

At that checkpoint no periodic binary or QA images were published as accepted. The accepted orthogonal ellipsoid artifact remained the active fallback.

## Why focus moved

The same unexplained surface-owner disagreement survived the one consolidated diagnosis/fix cycle, reaching the two-real-attempt go/no-go limit. Relaxing the numerical gate or silently dropping the outlier would make the artifact claim stronger than its evidence.

## Objective resume condition

### CPU-only ownership diagnosis (2026-07-21)

The requested identity instrumentation now enumerates the nearest, second-nearest, and every later hit for the exact deterministic `96x96`, step-14 pixel sequence. Every hit records slice/pixel, triangle id, periodic copy id, `t`, barycentric edge margin, and ray-facing normal. This follow-up launched no browser or GPU.

- 392 selected rays were enumerated; 354 had at least two hits.
- Applying the attempt-2 nearest-hit margin `>= 0.03` leaves the same 325 strict-interior rays.
- Those rays contain 1,066 CPU alternative-surface pairs.
- **No strict pair** has depth-separation error `<= 1e-4` against the recorded GPU discrepancy `0.13544333259743552`; therefore zero pairs can jointly match depth and normal.
- The closest-depth strict pair is slice 7, pixel `(87,45)`: nearest triangle 4057/copy `(0,0)`, `t=0.1878546121321925`; alternative triangle 3625/copy `(0,0)`, `t=0.3234424909577434`. Its separation is `0.1355878788255509` (error `0.0001445462281153831`), but its normal dot is `0.12707123337200904` (error `0.6127763018504433`). The alternative itself is nearly on an edge (`0.0020556489121024146`).
- The closest joint strict candidate is slice 3, pixel `(17,73)`: nearest triangle 5158/copy `(0,-1)`, `t=0.5611582257121187`, edge margin `0.15342932352556976`; second triangle 2426/copy `(0,-1)`, `t=0.6988188486334369`, edge margin `0.028645010071515986`. Separation `0.1376606229213182` misses by `0.0022172903238826813`; normal dot `0.6702351200929054` misses by `0.06961241512954697`.
- Before the strict filter, the nearest depth-only candidate is slice 0, pixel `(17,87)`, separation `0.13547631769308754` (error `0.00003298509565202723`), but normal dot `0.7919243838859868` still misses by `0.05207684866353446`, and its nearest edge margin `0.009041197029023569` means attempt 2 excluded it. It cannot be the reproduced strict-interior outlier.

**Conclusion:** CPU geometry alone does not explain the recorded outlier as the GPU choosing another valid overlapping surface at that sample center. The numerical gate must remain strict; an “overlap ambiguity” exclusion is unsupported.

### Exact next decision

Resume only with a diagnostic GPU attempt that first records the actual failing `(slice,pixel)`, decoded GPU `t`/normal, CPU nearest plus all alternatives, and the projected triangle vertices/copy uniforms for that one pixel. Also compare the four adjacent GPU texels to rule in/out a viewport/readback address offset. The outcome is binary:

1. GPU values match a different draw/adjacent texel: fix copy uniforms, viewport/readback addressing, or draw ownership.
2. GPU values match no submitted triangle: reduce to a one-pixel/one-slice raster capture and inspect interpolation/depth attachment semantics.

Do not spend another real attempt merely changing CPU acceptance; the CPU-only evidence has rejected that path.

## Resolution: fixed-function raster ownership

The authorized identity-capturing GPU attempt reproduced the outlier at slice 0, pixel `(3,45)` and stopped before a deterministic repeat. Its required identity summary is preserved at `/tmp/groundcover-gpu-bake-periodic/periodic-diagnostic-failure.json`; the later `/tmp/groundcover-gpu-bake-periodic/periodic-diagnostic-latest.json` records the corrected zero-mismatch run.

- GPU decoded `t=0.48297967318105767`, normal `(0.1112867917, 0.5511646442, 0.8269418269)`.
- Ideal double-precision Moller-Trumbore selected triangle 3551/copy `(0,0)`, `t=0.6184230057784932`, edge margin `0.25984086005452667`.
- Searching every submitted primitive found triangle 3032/copy `(0,0)` with interpolated `t=0.48309030291617505`, normal dot `0.999999911301934`, but ideal barycentric minimum `-0.0006230388193763925`: just outside the unsnapped analytic triangle.
- Transcribing the submitted-f32 fixed-function setup with 8 fractional window bits makes triangle 3032 the first covered owner with positive snapped barycentric margin `0.0017107420343573931`, `t=0.48297979832216587`, and matching normal `(0.1112867441, 0.5511645740, 0.8269418801)`. Depth differs from GPU by only `1.251411082e-7`.

**Root cause:** the CPU oracle modeled ideal continuous ray/triangle ownership, while the baker deliberately uses fixed-function rasterization whose window-space edge equations have finite subpixel precision. This was an oracle validity bug, not cross-direction contamination, a missing periodic copy, or incorrect GPU depth election.

The smallest fix retains the exact same 325 ideal-ray strict-interior sample selection and all original numerical thresholds, but determines the expected first owner/depth/normal using the submitted-f32, 8-bit-subpixel raster transcription. The consolidated corrected run then passed and performed its deterministic second bake:

- zero hit/miss mismatches;
- one sample where raster ownership legitimately differs from ideal ownership;
- maximum `t` error `7.170712652548872e-7`;
- RMS `t` error `6.136349957468802e-8`;
- minimum normal dot `0.9999999992655276`;
- mean normal dot `0.9999999999977324`;
- periodic-address maximum error `4.440892098500626e-16`;
- two identical packed hashes: `cc35db7ae4bc8f7107d389642116aef6855c0764dbb77f7b7ad04545ce905a2b`.

Accepted geometric-validation artifact:

- binary: `/tmp/groundcover-gpu-bake-periodic/cc35db7ae4bc8f71/multi-cushion-periodic-profile.gcrp`;
- QA/index: `/tmp/groundcover-gpu-bake-periodic/cc35db7ae4bc8f71/qa/`;
- stored atlas: `392x196`, eight `98x98` guarded slices with `96x96` canonical interiors;
- packed bytes: `615248` (`614656` texel payload plus v2 metadata).

This resolves the parked validation track. The fixture remains geometric-only and makes no botanical or Sphagnum species claim.
