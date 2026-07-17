# Morning review — 2026-07-17 overnight run

Orchestrator: Fable 5. All work autonomous per the AFK mandate. Every terrain link
carries `grass=0` (your law — grass hides the mesh). This file is the single place to
review the night's results.

## How to bring up the links

The four data servers + vite were started with `nohup` on this machine. If a link 404s
or the page is blank, the server died (they don't survive a session stop) — restart it
by pasting the matching command into the prompt with a leading `!`:

```
! cd /Users/sebastian/IdeaProjects/fable-demo2 && npx vite --port 5195 --strictPort &
! cd /Users/sebastian/IdeaProjects/fable-demo2 && PORT=8804 LAAS_PREVIEW_MANIFEST=builds/af31c169b6e41b970ec8f5d5ae26908ab38336314aeac7aca98231dc7eaae73c/m/3a68ba63f0f8231e/manifest.json node tools/serve-data.mjs &
! cd /Users/sebastian/IdeaProjects/fable-demo2 && PORT=8805 LAAS_PREVIEW_MANIFEST=builds/6006c7429a87881e58767c62385dfd99b977ee143cfd330b5c177cf2438608bb/m/585887fdf41b2441/manifest.json node tools/serve-data.mjs &
! cd /Users/sebastian/IdeaProjects/fable-demo2 && PORT=8806 LAAS_PREVIEW_MANIFEST=builds/23bdc31a7af3a8164f3e68805429de138941d1119bbe0b2b7d2580f3e302f9b0/m/f79887ee9b3da18b/manifest.json node tools/serve-data.mjs &
```

## 1. Forest — second full-LOD0 tranche (the Taevaskoja HERO chunk)

**Status: PACKED + VERIFIED + BOOTS CLEAN. New live regime.** Extends the accepted
mesic-mineral forest generator (unretuned) across complete LOD0 `(0,151,93)` — the
hero chunk containing the Ahja river/escarpment. All seven pack gates pass (zero seams,
byte-exact parents + inherited layers). Heavy, correct hard-exclusion from the river.

- Hero-chunk center (open field / forest edge — fly toward the tree stands):
  `http://localhost:5195/?scene=world&src=estonia&dataurl=http%3A%2F%2Flocalhost%3A8804&alt=12&x=310272&z=191488&yaw=0&pitch=-0.22&grass=0&shadowclipres=896`

What to check: same natural forest-floor microdetail you accepted before, now over a
different 2 km chunk with a real river transition; no seams or repetition.

## 2. Raised bog — self-organized string/pool network (NEW regime, first bootable)

**Status: PACKED + VERIFIED + BOOTS CLEAN. This is the night's main new result.**
Three prior bog families were parked for making even corrugations with isolated pools.
This one *generates* a real branching/merging string network with margin-coupled pools
via anisotropic ecohydrological self-organization (Rietkerk/Eppinga), with the string
spacing modulated by the mire's own slope so it varies naturally instead of locking to
one wavelength. All gates pass. Relief is honestly subtle (~5-6 cm — that is the real,
evidence-bounded scale of bog microtopography; it is not cranked).

- Mire center:
  `http://localhost:5195/?scene=world&src=estonia&dataurl=http%3A%2F%2Flocalhost%3A8805&alt=8&x=167808&z=183040&yaw=0&pitch=-0.25&grass=0&shadowclipres=896`

What to check: the peat surface should show subtle linear string/hollow undulations
(fly low and look across the surface at a shallow angle). **Known cosmetic flag:** the
carved 128 m patch has no boundary taper, so there is a ~5-6 cm step ringing the patch
edge against the surrounding baseline terrain. If you like the bog morphology, adding a
taper is a cheap fix — I left it so you can judge the morphology itself first.

## 3. Aeolian dunes — real coastal dune landform (NEW landscape, honest, no synthesis)

**Status: BASE COOKED + BOOTS CLEAN.** Investigation this night showed the Estonian
1 m DTM already contains the full 1-8 m dune relief (the parked attempt had wrongly
concluded the parent was flat), and that flat dry bare sand is *effectively smooth* at
the sub-metre band (Kijkduin evidence) — so inventing ripples would be faking. The
honest deliverable is therefore the real dune landform straight from the DTM: no
synthesis at all. Cooking a west-coast AOI base makes it renderable.

- Wide vantage (bare-sand blowout + rolling vegetated dunes + beach/sea):
  `http://localhost:5195/?scene=world&src=estonia&dataurl=http%3A%2F%2Flocalhost%3A8806&alt=140&x=25728&z=102400&yaw=0&pitch=-0.5&grass=0&shadowclipres=896`
- Ground level near the dunes:
  `http://localhost:5195/?scene=world&src=estonia&dataurl=http%3A%2F%2Flocalhost%3A8806&alt=12&x=25728&z=101888&yaw=0&pitch=-0.22&grass=0&shadowclipres=896`

What to check: rolling dune ridges/swales, a bare-sand blowout (rendered as sand), and
the beach→sea transition. These are stabilized *forested* dunes, so much of the relief
sits under tree cover — that is real, not a defect.

## Regimes NOT given a live link tonight (honest dispositions)

- **Cliffs / erodible banks (rank 1):** a new anchor+organization mechanism SOLVED the
  connectedness the prior attempt lacked (no more sparse islands), but at Dev A the
  *measured* ALS evidence only supports ~0.1 m of added relief beyond the base bank, so
  the visible result is near-identical to the base — the same character you rejected
  before. Root cause is an evidence ceiling (the macroform is already in the DTM; more
  relief needs better measured data or banned foreign-amplitude transfer), not a
  mechanism failure. Not packed. Cliffs stay evidence-blocked.
- **Dune sub-metre texture:** deliberately none — flat dry sand is evidenced-smooth;
  the only honest future target is discrete blowout-rim forms, needing a sub-metre prior
  we do not yet have (Curonian data is unavailable; Kijkduin's noise floor is too high).

## Decisions for you

1. **Bog morphology** — good enough to keep? If yes, I add the boundary taper and any
   amplitude recalibration within the evidence envelope.
2. **National base cook** — tonight proved any region can be cooked per-AOI on demand.
   Rolling this out widely (or nationally) is a bytes/time decision that is yours.

*(Agriculture v2 and an evidence-hunt sweep for the remaining starved regimes were
queued; their status is appended below if they completed before you returned.)*
