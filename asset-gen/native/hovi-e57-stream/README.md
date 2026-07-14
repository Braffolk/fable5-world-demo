# Hovi E57 Stream Reader

This project-owned native boundary uses the checksum-pinned Rust `e57` 0.11.13
decoder. The current milestone exposes a bounded `dry-run` that reads the E57
header and XML metadata only, plus an authority-fixed `probe` over scan ordinal 0,
GUID `0000000000000001`, and source ordinals `[0,65536)`. `extract` fails closed
until a separate full point-read authorization and extractor are implemented.

All dry-run resource ceilings are mandatory and are additionally constrained by
compiled hard maxima. The HY_SPRUCE4 profile requires local ScaledInteger XYZ,
single-precision intensity, Integer row/column/Cartesian invalid state, unique scan
GUIDs, finite unit-quaternion poses, and absent source color.

The probe writes every valid, direction-only, and invalid source record as one
canonical fixed-width 96-byte record to a required inherited `--output-fd`. It
accepts no output path, hashes the exact bytes written, flushes and `fsync`s them,
and reports the exact count, byte length, and SHA-256 on stdout. It does not apply
the pose, claim EOF, validate the publisher's full point count, publish the output,
or create a surface claim.

The optional `e57/crc32c` feature is deliberately disabled. This boundary uses
the vendored crate's pure-Rust CRC implementation rather than adding an unsafe
SIMD/build-code dependency whose registry archive does not bundle its license.
The reduced dependency and audit surface outweighs CRC throughput for the bounded
probe; any future change requires explicit source, license, SBOM, and benchmark
evidence.

The vendored upstream archive and every LAAS-modified vendor file are bound by
`vendor/UPSTREAM.json` and the content-hashed `vendor/PATCH-MANIFEST.txt`. The
reported vendor patch-set identity is the SHA-256 of the manifest's exact bytes.
Do not replace the vendored source with an unpinned registry resolution.
