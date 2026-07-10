# DEEP-FOREST base decomposition (grass OFF) — 2026-07-05, honest gpuWall ablation

Pose: deep forest, embedded in trees, eye-level (x=-852,z=228,y=+2,yaw0.9,pitch-0.08). dpr2. measureFrames.
⚠️ Per-pass timestamps OVERLAP (Apple render||compute concurrency) → not additive; gpuWall is truth, ablation deltas are the attribution.

| config | query | gpuWall | c.nanVisClear | r.half.mrt | r.scene | compute | render | c.nanShadowHalf | r.nanHwPass |
|---|---|---|---|---|---|---|---|---|---|
| base | grass=0 | 17.4ms | 7.73 | 6.29 | 5.64 | 8.72 | 20.84 | 0.2 | 0.66 |
| no-shadow | grass=0&nanshadow=0 | 16.9ms | 7.27 | 5.77 | 5.24 | 8.91 | 19.99 | 0 | 0.66 |
| no-voxel(leaves+crowns) | grass=0&voxreg=0 | 71.4ms | 65.54 | 1.9 | 1.57 | 66.45 | 10.49 | 0.13 | 0.26 |
| no-crownlod0 | grass=0&crownlod0=0 | 13.5ms | 3.21 | 6.49 | 5.77 | 4.26 | 21.3 | 0.2 | 0.66 |
| no-leafmesh | grass=0&naniteleaf=0 | 13.3ms | 1.44 | 5.11 | 3.6 | 3.15 | 25.95 | 0.26 | 0.98 |
| no-shvox2(crown-shadows) | grass=0&shvox2=0 | 16.9ms | 7.01 | 6.16 | 5.51 | 8 | 20.77 | 0.13 | 0.59 |
| base2(drift) | grass=0 | 17.5ms | 7.27 | 6.42 | 5.77 | 8.26 | 20.97 | 0.2 | 0.66 |

## Subsystem marginal cost (base 17.4ms − ablated), grass already off
- **no-shadow**: ~0.5ms (frame drops to 16.9ms)
- **no-voxel(leaves+crowns)**: ~-54ms (frame drops to 71.4ms)
- **no-crownlod0**: ~3.9ms (frame drops to 13.5ms)
- **no-leafmesh**: ~4.1ms (frame drops to 13.3ms)
- **no-shvox2(crown-shadows)**: ~0.5ms (frame drops to 16.9ms)
- drift base2−base = 0.1ms (measurement noise floor)
