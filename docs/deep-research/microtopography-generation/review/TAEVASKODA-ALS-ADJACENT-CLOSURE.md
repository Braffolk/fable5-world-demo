# Taevaskoda ALS Adjacent-Tile Closure

**Date:** 2026-07-13
**Decision:** acquire exactly `445679_2019_tava.laz` and
`444680_2019_tava.laz`; do not acquire `445680`, and do not add a second epoch.

The frozen selection is
`asset-gen/config/evidence/taevaskoda-ahja-als-adjacent.json`. It has its own
selection identity so the already-retained eight-epoch `444679` manifest and its
original config snapshot remain byte-for-byte valid.

## Why adjacent evidence is required

The canonical fine publication is
`[679424,679936] x [6444544,6445056]` in EPSG:3301. Its deterministic `8 m`
repair collar is `[679416,679944] x [6444536,6445064]`. The collar therefore
crosses the north edge of primary ALS tile `444679`; tile `445679` is directly
required by the repair domain.

There is a second, independent closure issue. The previously qualified 2019 Ahja
profile was clipped to tile `444679`, so its first and last evidence stations were
tile crossings rather than physical reach endpoints. Frozen ETAK centerline
`etak_id=2356024` is a single `2288.237233708688 m` line. Exact intersections are:

| tile | EPSG:3301 bounds | centerline length |
|---|---|---:|
| `444679` | `[679000,680000) x [6444000,6445000)` | `492.827633307022 m` |
| `445679` | `[679000,680000) x [6445000,6446000)` | `247.116664369654 m` |
| `444680` | `[680000,681000) x [6444000,6445000)` | `1548.292936032012 m` |
| `445680` | `[680000,681000) x [6445000,6446000)` | exactly `0 m` |

Thus the smallest file-level acquisition that removes both artificial profile
endpoints is the same 2019 normal-mapping campaign in `445679` and `444680`.
Acquiring only narrow strips is not offered by the official file service.

The broad Ahja area polygon does intersect `445680`, but only from
`E 680722.37` eastward in that tile. The repair collar ends at `E 679944`, and
the selected centerline never enters the corner tile. That remote polygon piece
is neither profile support nor `8 m` repair support, so it does not authorize a
third file.

## Frozen official selection

The Maa- ja Ruumiamet public index pages were read on 2026-07-13 and list both
files as available `lidar_laz_tava` artifacts:

| tile | file | official index display | canonical endpoint |
|---|---|---:|---|
| `445679` | `445679_2019_tava.laz` | `39.6 MB` | `https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=445679&andmetyyp=lidar_laz_tava&dl=1&f=445679_2019_tava.laz&page_id=614` |
| `444680` | `444680_2019_tava.laz` | `44.3 MB` | `https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=444680&andmetyyp=lidar_laz_tava&dl=1&f=444680_2019_tava.laz&page_id=614` |

The portal's rounded display values are not exact byte counts. Until retention,
both exact byte counts, SHA-256 identities, and exact additional on-disk storage
remain unknown. The displayed sum is about `83.9 MB`, but must not be used as an
integrity or capacity assertion.

## Epoch decision and gate

No corroborating adjacent epoch is scientifically required. The accepted water
surface is a 2019 hydrological snapshot, and only the missing spatial continuation
of that same campaign closes it. Mixing epochs would conflate changed water level
with spatial support, while acquiring a second campaign would not repair a failed
2019 support test.

Retention alone does not predetermine acceptance. After acquisition and inventory,
the extended 2019 class-9 profile must independently pass the existing minimum
point count, Hampel rejection, maximum-gap, unambiguous orientation,
bank-consistency, topology, and abstention gates. If it fails, the affected reach
abstains; that result does not authorize another epoch automatically. Existing
epochs in `444679` remain corroboration of the anomaly diagnosis, not profile
values to average into 2019.

## Scope boundary

This selection closes the Taevaskoda Stage 1 evidence domain only. It does not
change packed formats, authorize bathymetry or morphology, or create a
coordinate-specific repair. National activation still requires the same typed
feature/topology gates and the Jägala waterfall protected-discontinuity regression.
