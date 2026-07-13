# Estonia Microtopography Regime And Phenomenon Matrix

Status: architecture input, 2026-07-13. This document defines the physical problem space that model selection must cover. It is not an implementation authorization and does not claim that every row already has adequate training data.

## Purpose

The existing game land-cover palette is a rendering classification, not a sufficient terrain generator taxonomy. `forest`, `field`, `bog`, or `sand` does not uniquely determine the geometry at 6.25 cm sampling. Surface form is produced jointly by substrate, surficial material, soil profile, hydrology, slope position, vegetation and roots, management, disturbance history, exposure, and observation quality.

The replacement architecture must therefore classify overlapping physical regimes and events, not select one texture per land-cover polygon. Every generated point must have:

- a typed structural state;
- a mixture of eligible morphology regimes with confidence and spatial transition support;
- an observation-support state describing what the DTM and other sensors actually measured;
- a release status for each requested scale band;
- an abstention result when the evidence or target prior is inadequate.

`Unsupported` means retain the corrected structural authority without invented residual detail. It never means fill the area with generic noise.

## Representation Classes

| Class | Meaning | Heightfield treatment |
|---|---|---|
| measured ground | directly observed single-valued ground with adequate returns | preserve within calibrated observation uncertainty |
| reconstructed ground | single-valued ground corrected from direct vector, hydrographic, optical, or neighboring evidence | publish through corrected ordinary LOD 0 and derive finer rungs from it |
| unresolved surface | physically real single-valued morphology below source support | synthesize only with a validated regime prior |
| forbidden residual | open-water surface, hard structure, or other domain where relief must not be invented | reapply zero/typed residual constraint after every synthesis stage |
| occluded unknown | ground hidden by canopy, objects, water, shadow, or sensor geometry | use uncertainty and a supported conditional prior or abstain |
| non-heightfield structure | vertical, undercut, overhanging, cave, detached rock, root plate, or object geometry | outside the fine-height generator; do not flatten it into a false heightfield claim |

## Mandatory Typed Structural Domains

These domains are reconstructed before unresolved-detail synthesis. They may overlap a morphology regime, but the generator may not move their constrained structure silently.

| Domain | Required distinction | Direct evidence | Required behavior |
|---|---|---|---|
| open-water optical surface | visible water plane or sloped river surface | ETAK water polygons/centerlines, water level evidence, orthophoto, DTM confidence | no stochastic terrain relief; model as a connected hydrologic surface when this height field represents water |
| submerged terrain bed | terrain below a separately rendered water surface | bathymetry where available, bank continuation, channel priors, explicit uncertainty | never confuse with the optical water surface; synthesize only from a separately validated bed regime |
| shoreline | water/land intersection | ETAK shoreline and water polygons, dated orthophoto, local elevation | reconstruct a smooth world-space curve; do not retain 1 m raster stair steps |
| bank | terrestrial surface adjacent to a channel or water body | DTM, shoreline, channel side, slope, orthophoto | preserve bank-side topology and signed distance; do not blur across the shoreline |
| escarpment | abrupt natural slope break | ETAK escarpment lines, EGT features, DTM derivatives, imagery | preserve the break in the single-valued portion and carry a non-heightfield warning for vertical/undercut faces |
| ditch or small channel | connected drainage structure | ETAK, inferred drainage, orthophoto, local relief | preserve connectivity, downstream direction, cross-section family, and junction behavior |
| road or track | engineered corridor | ETAK roads, orthophoto, management evidence | distinguish paved, compacted unpaved, forest track, wheel ruts, verge, and drainage; do not apply a generic hard mask |
| quarry, spoil, excavation | engineered removal or deposition | ETAK, soil technogenic classes, orthophoto, DTM change | preserve benches, cuts, piles, and temporal state as typed structure; unsupported sites abstain |
| building/object contamination | non-ground leakage into DTM | nDSM/DTM disagreement, ETAK buildings, imagery | remove only with calibrated evidence; do not destroy real cliffs or boulders |
| mapped boulder or exposed block | discrete rock evidence | ETAK points/polygons, imagery, geology | low single-valued portions may affect terrain; detached/overhanging form belongs to object or structural geometry |

## Morphology Regimes

The `candidate families` column defines a fair bakeoff, not a preselected winner. Every candidate returns an ordinary metric height surface to the existing cook contract.

| Regime | Constituent phenomena and scale breaks | Required conditions | Target-scale evidence needed | Candidate families | Principal failure or representation limit |
|---|---|---|---|---|---|
| raised bog | hummock, lawn, hollow, pool margin, ridge/string organization, local collapse; centimeter surface texture over meter-scale microforms | peat type/depth, water-table proxy, flow direction, vegetation state, drainage disturbance | Estonian or transferable 1-5 cm DEMs spanning sites, seasons, intact/drained states and larger contextual windows | measured/process mixture, conditional simulation, contextual exemplar, GAN, diffusion, hybrids | foreign small plots do not establish national layout; open pools remain forbidden residual domains |
| aapa/fen and transitional mire | strings, flarks, sedge tussocks, saturated lawns, channels, floating-mat margins | mire type, minerotrophic state, flow direction, slope, peat/mineral transition, vegetation | site-separated target surfaces with hydrologic context and class labels | orientation-aware process/event model, conditional simulation, learned expert | raised-bog priors cannot be reused blindly; floating or deforming vegetation is not stable bare ground |
| drained or cut peatland | extraction ridges, drainage ditches, cut faces, subsidence, machinery tracks, regrowth | peatfield boundaries, ditch network, extraction state/date, orthophoto | dated target surfaces across active, abandoned, and restored sites | typed deterministic reconstruction plus event-conditioned learned/exemplar expert | highly temporal engineered morphology; generic peat detail is wrong |
| forest pit-and-mound | windthrow pit, mound, decayed root plate footprint, age sequence, overlap, slope alignment | stand type/age, soil depth/texture, wetness, wind exposure, management and disturbance history | 2-5 cm ground DTMs from multiple Estonia/analogue sites with ground visibility and event age | calibrated stochastic event process, contextual exemplar, GAN/diffusion expert, hybrid | canopy occlusion and root plates; a single ideal pit/mound primitive collapses real variance |
| forest floor without major disturbance | root heave, shallow hollows, decomposed logs, animal paths, moss/organic mat, drainage microchannels | species/stand age, soil, wetness, slope, management, deadwood context | target surfaces with vegetation/organic-layer treatment explicitly documented | learned/exemplar expert or calibrated event mixture | sensor often measures vegetation rather than mineral ground; semantics alone do not determine form |
| managed forest and clear-cut | harvester ruts, skid trails, drainage, stump/root disturbance, slash piles, recovery | forestry operations/date, wetness, soil bearing capacity, orthophoto | dated pre/post-operation surfaces and management labels | deterministic corridor reconstruction plus event-conditioned generator | rapidly changing and often absent from static registry data |
| cultivated field: ploughed | furrow/ridge geometry, headlands, wheel tracks, clods/aggregates, erosion, cross-slope drainage | crop/operation/date, implement, direction, soil texture/moisture, slope, orthophoto | millimeter-centimeter surfaces across tools, soils, moisture, directions and weathering states | directional process model, contextual exemplar, learned expert, hybrid | land-cover alone cannot identify current operation; optical date mismatch is decisive |
| cultivated field: seedbed/harrowed/rolled | fine aggregates, weak rows, compaction, wheel tracks, rain crust | operation sequence, soil texture, moisture/rain history, direction | target surfaces resolving both millimeter aggregate and decimeter track bands | multi-band event/process or learned expert | nominal 6.25 cm output cannot represent all millimeter roughness; retain only resolvable geometry |
| pasture and meadow | tussocks, hoof prints, mole/ant mounds, shallow drains, trampling paths, erosion around gates | grazing/mowing state, soil, wetness, slope, livestock evidence | representative ground surfaces with event frequency and management state | stochastic event mixture, conditional simulation, learned expert | rare-event density and state cannot be guessed from `grassland` alone |
| yard and recreational turf | grading, mowing/compaction, paths, drainage, local disturbance | ETAK yard/open class, orthophoto, object/road context | target surfaces for supported subtypes | typed reconstruction plus subtype expert | current `yard = grass` palette is not morphology evidence |
| exposed sandy soil | grains below representable scale, ripples, deflation hollows, tracks, rain and rill marks | grain family, moisture, wind/water exposure, land use | centimeter surfaces across aeolian, fluvial, coastal, quarry and disturbed sand | process specialist, exemplar, GAN/diffusion expert | one `sand` class conflates distinct processes; grain geometry is below the height grid |
| dune and aeolian sand | ripple fields, slip-face breaks, blowouts, vegetation anchoring, storm/track disturbance | dune form, wind regime/orientation, exposure, vegetation, orthophoto | contextual target surfaces covering ripple-to-dune hierarchy | orientation-aware process model, learned expert, hybrid | whole forms may already be measured at 1 m; do not regenerate macrorelief |
| beach sand and shoreface | swash ridges, runnels, wrack/ice disturbance, berms, wet/dry zones | shore orientation, water level/wave exposure, sediment, date | dated coastal target surfaces across Baltic exposure states | coastal process/event expert, exemplar, learned expert | temporal water level and shoreline movement; optical water edges are not timeless truth |
| shingle, gravel, and cobble shore | imbrication-scale relief, storm ridges, sorting patches, drift/ice push | sediment class/size, wave/ice exposure, beach orientation | target surfaces plus particle-size distributions | object-aware/event process or learned/exemplar expert | discrete blocks and overhangs exceed pure heightfield fidelity |
| till and glacial sediment plain | matrix roughness, embedded coarse fragments, frost/rain disturbance, local drainage, inherited weak lineation | surficial deposit, soil texture/skeleton/stoniness, wetness, land use | Estonia target surfaces stratified by till/deposit and surface state | conditional simulation, exemplar, learned expert, narrow calibrated process | macro drumlins/eskers belong to measured base; unsupported patterned-ground analogues must not be imported |
| gravel, esker, and outwash exposure | sorting, cobble/stone patches, shallow channels, extraction/track disturbance | geomorphology coverage, deposit class, exposure/land use, orthophoto | target surfaces across intact and worked sites | exemplar/learned expert, event process | EGT 1:50k coverage is incomplete and cannot locate each stone |
| carbonate alvar: thin soil | exposed pavement/soil islands, shallow solution hollows, vegetation-edged microrelief | limestone/dolomite, soil thickness/type, exposure, joint orientation evidence, wetness | Estonian alvar surfaces with exposed/covered fractions and joint mapping | fracture/process model, contextual exemplar, learned expert, hybrid | coarse geology boundaries cannot place grikes; vegetation surface must not become bedrock height |
| carbonate pavement and karst | joints/grikes, karren, solution pits, clint edges, sediment fills | exposed bedrock, lithology, fracture/bedding evidence, drainage | high-resolution local rock surfaces and contextual joint networks | explicit fracture/process specialist, implicit-feature method, learned expert | narrow/deep cracks and overhangs alias or violate heightfield; must clip claim to resolvable single-valued top |
| sandstone outcrop top and slope | bedding-controlled ledges, joints, weathering pits, blocks, runoff grooves, colluvium | bedrock formation, exposure, bedding/joint evidence, moisture, slope, imagery | target surfaces from relevant Estonian sandstone, not unrelated lithologies alone | implicit/process features, contextual exemplar, learned expert, hybrid | vertical Suur Taevaskoda wall, caves, and undercuts are not representable by this heightfield |
| limestone/dolomite outcrop top | bedding steps, fractures, karren, frost blocks, soil pockets | lithology, exposure, structure, slope | lithology-matched target surfaces | process/exemplar/learned expert | rock RGB-to-TRI papers do not supply height targets or Estonia transfer |
| colluvial slope and talus | lobes, rills, creep steps, block fields, deposition toes | slope position, substrate, soil depth, drainage, exposed-cliff source | contextual target surfaces spanning source-to-toe | event/process simulation, learned expert | detached blocks and overlap; local patches without upslope context are invalid |
| fluvial floodplain | scrolls/depressions already partly measured, silt microrelief, levees, abandoned channels, flood deposits, animal/vehicle disturbance | channel topology, flood frequency, soil, land use, wetness | dated pre/post-flood target surfaces and contextual hydrology | typed base reconstruction, event/process expert, learned expert | do not let a fine generator reroute mapped drainage or invent current flood state |
| active channel bar and exposed bed | ripples/dunes, gravel bars, scour, deposits, wet/dry transition | flow regime, sediment, water level/date, channel order | bathymetric/exposed-bar surfaces under known flow state | fluvial process specialist, learned/exemplar expert | submerged bed is poorly observed; open-water surface is a different domain |
| small rill, gully, seep, and spring | connected incision, headcut, depositional fan, saturation mound/hollow | local flow accumulation, soil erodibility, slope, imagery, mapped drains | target surfaces including network junctions and temporal state | hydrology-aware process specialist, learned expert, hybrid | patch-local synthesis can break connectivity; generic erosion everywhere is unacceptable |
| saline/coastal wetland | pans, shallow channels, hummocks, wrack/ice disturbance, inundation boundaries | coastal soil classes, elevation/water level, salinity proxy, vegetation | Baltic target surfaces with inundation state | hydrologic/process expert, conditional simulation, learned expert | borrowed tidal-marsh morphology may not transfer to weak-tide Baltic conditions |
| technogenic and disturbed ground | cuts, fills, spoil, compaction, tracks, demolition/extraction residue | full technogenic soil class, ETAK, dated imagery, history | subtype-specific dated surfaces | typed reconstruction plus event/subtype expert | `barren` is not a physical class; stale imagery creates false state |

## Cross-Cutting State Axes

No regime row may be implemented as a single immutable template. At minimum, candidate models must represent or explicitly abstain across these axes:

| Axis | Required distinctions |
|---|---|
| substrate | bedrock family, surficial deposit, coarse skeleton/stoniness, soil depth and vertical profile |
| hydrology | dry, seasonally wet, saturated, inundated, channelized; flow direction and shore side where relevant |
| relief | slope, aspect, curvature, topographic position, contributing area, distance to structural break at multiple radii |
| biological state | vegetation cover/occlusion, forest species and age, root/disturbance state, organic layer |
| land management | cultivation operation, implement/direction, grazing/mowing, forestry operation, drainage, extraction |
| disturbance age | fresh event, weathered/recovered state, overlap with older events |
| observation | DTM vintage/source, point support/interpolation, nDSM/CHM disagreement, image date/visibility, map scale/coverage |
| representation | single-valued, near-vertical, undercut/overhang, discrete object, below-grid feature |

## Scale Contract

The packed rungs define sample supports, not two semantic feature buckets. Candidate methods operate on physical bands established by measured target modulation transfer and antialiased analysis filters:

- corrected structural authority includes real structures at and above the reliable 1 m observation support, even when reconstructed on a finer internal grid;
- the 0.25-1 m band may contain unresolved event shape, edges, and material forms only when target evidence resolves them;
- the 0.0625-0.25 m band may contain target-supported geometry, but not millimeter grains, leaf litter, grass blades, optical texture, or features whose acquisition error exceeds their signal;
- features crossing a band boundary are generated as coherent physical forms and then decomposed by the canonical analysis pyramid; they are not independently painted into each rung;
- the final accepted master surface is generated in a world-aligned physical domain and all packed rungs are derived from that one surface and corrected authority.

## Release Matrix Required From Implementation

Before any national cook, implementation must materialize a machine-readable row for every claimed regime containing:

1. regime and subtype identifier;
2. eligibility predicates and transition model;
3. direct structural constraints and forbidden residual domains;
4. target datasets, sites, acquisition error, effective resolution, ground visibility, license, and hashes;
5. train/validation/blind-test geographic split;
6. candidate methods and frozen configurations;
7. per-band signal-to-measurement-error evidence;
8. blind visual and measured bakeoff result;
9. packed entropy and cook-cost result;
10. failure, abstention, and representation-limit behavior;
11. release status: `unsupported`, `research`, `pilot`, or `national`.

No implicit fallback from an unsupported row to another regime is permitted.
