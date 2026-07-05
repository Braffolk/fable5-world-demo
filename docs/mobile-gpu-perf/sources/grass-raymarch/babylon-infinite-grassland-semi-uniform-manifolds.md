# Infinite Grassland: Ray-Traced "Rare Semi-Uniform Entity Manifolds" for Rendering Countless Discreta

Source: https://babylonjs.medium.com/infinite-grassland-ray-traced-rare-semi-uniform-entity-manifolds-for-rendering-countless-discreta-36630d72fc6

## Core Innovation

A screen-space volumetric technique for rendering "countless discreta" (individual grass
elements) with **screen-constant computational cost** rather than per-entity cost. The cost is
determined ONLY by per-pixel shader weight × number of pixels covered — there is NO per-entity
overhead because entities have no geometry.

## Key Concept: Rare Semi-Uniform Entity Manifolds

"A locally-Euclidean space where each point in the space is associated with an entity." Practical
example uses ℤ² (integer coordinates in 2D Cartesian space):

- Entities distributed at discrete grid points.
- From any world position, finding nearby entities is trivial: `f(x, y) = (round(x), round(y))`.
- All entities are mathematically similar but carry different parameters (rotation, curvature,
  offset), derived from the entity origin via texture sampling or noise.

## Rendering Pipeline

**1. Ray Origin Setup** — Vertex shader computes world-space position + view direction. The
geometry position is the ray origin; it's a "hull" containing the manifold (may not itself contain
grass).

**2. Corridor Computation** — Instead of checking infinite entities, compute a bounded "corridor":
project the ray onto 2D manifold space and calculate a bounded region containing potentially
intersectable entities. Corridor dimensions depend on entity width and transparency.

**3. Surface Derivation** — For each entity in the corridor, derive a mathematical surface
(demos use a custom paraboloid). Parameters vary per-entity but derive from entity origin. Must
stay locally-bound to prevent clipping at corridor edges.

**4. Ray-Surface Intersection & Shading** — Compute line-surface intersections for all relevant
entities, convert intersection points to UV, light and blend.

## Cost Model (the load-bearing claim)

> "the cost of the effect is exclusively determined by the per-pixel weight of the shader and the
> number of pixels to which it's being applied."

No per-entity overhead — entities are analytic surfaces evaluated only along the ray inside the
corridor. This is the "screen-constant-cost grass" property.

## Limitations Acknowledged (proof-of-concept)

- Aliasing and depth problems.
- "Corridor"-related clipping artifacts when boundaries viewed edge-on.
- Over-rendering where grass appears outside intended volumes.
- Visible periodicity from cosine-based parameter derivation.

## Potential Extensions

- Higher fidelity via manual mip-mapping; control textures replacing procedural params.
- Extending to forests, clouds, hair/fur (narrower discreta).
- Hexagonal grid to reduce visual regularity.
- Direct intersection calculation eliminating the corridor requirement.

## Performance Claims

No concrete FPS/ms numbers given. Renders "hundreds of thousands of individual grass cards."
Author flags this as exploratory research, not production-validated.

---

### Map to our kGrassRay march
- The **corridor** = bounding the ℤ² cells the ray crosses so you only test O(cells-crossed)
  entities, NOT march fixed small steps. This is the analytic version of our per-pixel march:
  replace fixed-step raymarch with a **cell-DDA over the grass grid** + **analytic ray-vs-paraboloid**
  per cell (closed-form intersection, no iterative stepping).
- Each blade is an analytic surface → **zero dependent texture fetches per step**; params come from
  a single hash/noise of the cell origin.
- Screen-constant cost = the mobile-friendly property we want: cost scales with covered pixels, not
  blade count.
