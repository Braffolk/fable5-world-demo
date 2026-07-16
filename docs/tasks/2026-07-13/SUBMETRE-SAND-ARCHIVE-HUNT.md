# Sub-metre bare-sand dune surface archive hunt (2026-07-17)

Scope: one capped (~25 min) public-archive hunt for a downloadable, licensed <=0.25 m (ideally
<=0.10 m) bare-sand dune surface-geometry archive from the Baltic or a physically analogous
temperate sandy coast, with independent error/control, to feed a 0.0625-1 m dune-sand morphology
prior. Facts only, no recommendations. Note downloads themselves capped at 5 MB (not attempted).

## PRIORITY 1 — Curonian Spit (Pilkosios / Nagliai) multi-epoch UAV-SfM DEMs — VERDICT

The study is REAL and confirmed via indexed abstract text, but NO public downloadable archive of
its DEMs/point clouds was locatable, and NO data-availability statement could be read.

- Confirmed study (abstract paraphrased consistently across indexes): 6 DEMs of the **Pilkosios
  Dunes** (Nagliai Nature Reserve, Curonian Spit) for **2010, Jan/May/Oct 2018, May 2019, Nov
  2022**; differenced to elevation-change rasters; highest-intensity change over the **bare-sand
  surface class** at active blowouts, avg negative change ~**2 cm/yr**; strongest predictor =
  distance to grassland/bare-sand edge. This matches the "Tijūnaitė et al. 2018-2022, ~4-6.6 cm"
  named lead.
- Exact peer-reviewed paper by title + DOI: **NOT located** in the time budget. Could not resolve a
  journal/DOI for the Pilkosios 6-DEM paper. Associated authors surfaced: Simonas Danielius,
  Donatas Pupienis (VU), likely Jarmalavičius/Žilinskas circle; the surname "Tijūnaitė" did not
  attach to a resolvable DOI (a "Gabriele Tijunaityte" hit is an unrelated astronomer).
- Related accuracy paper **"Comparison of Accuracy of UAV Aerials and Ground Measurements in the
  Curonian Spit Dunes"** — only found on academia.edu (HTTP 403, unreadable); no journal DOI, no
  data deposit found. Describes the earlier Sep 2015 / Apr 2016 marks+GPS+UAV 3D-model-diff method.
- Repositories checked for a deposit: Zenodo (search unreadable via fetch; keyword web-search found
  nothing Curonian), Crossref bibliographic query (returned only older Curonian dune papers, none
  the Pilkosios DEM paper, none with data DOIs), Figshare/Dryad/ScienceBase/PANGAEA — no Curonian
  dune DEM deposit surfaced. Klaipėda University env-remote-sensing group page seen but no archive.

**Data-availability statement (verbatim): UNAVAILABLE — the source paper could not be accessed
(academia.edu 403; no open full text / DOI resolved), so no verbatim wording could be recorded.**
Bottom line: no public, licensed, downloadable Curonian-Spit sub-metre dune DEM archive found.

## PRIORITY 2 — adjacent temperate-coast open geodata sweep

| Source / dataset | Coast | Sensor | Res / spacing | Error / control | Coverage / epochs | Raw vs DEM | License | Size | Direct DL |
|---|---|---|---|---|---|---|---|---|---|
| **Kijkduin 4D beach-dune** — Vos et al., Sci Data 2022; PANGAEA **10.1594/PANGAEA.934058** | NL (North Sea) | Riegl VZ-2000 TLS, 1550 nm | ~50 pts/m² (0.05°); daily hi-res ~4500 pts/m² (0.013°); 1 cm min spacing | median align 0.4 cm (sd 1.9 cm); validated vs **3 RTK-GNSS surveys + ALS**; +1-2 cm/100 m range | ~1 km coast; **4082 hourly** clouds, 11 Nov 2016–26 May 2017 (190 d) | **Raw LAZ point clouds**, mm precision (no gridded DEM) | **CC BY 4.0** | very large (thousands of ~1.06M-pt clouds); total not stated | Yes (PANGAEA) |
| **Noordwijk 4D beach-dune** — Vos/Kuschnerus/Lindenbergh/de Vries, 4TU 2023 (`data.4tu.nl/datasets/1aac46fb…`) | NL (North Sea) | Riegl VZ-2000 TLS (fixed, hotel balcony) | ~4M pts/scan; spacing varies w/ tide/atmos | metrics in metadata files (not read); no independent-checkpoint figure captured | ~1000×350 m; **21812 hourly** scans, 11 Jul 2019–21 Jun 2022 | **Raw point clouds** (+intensity) | **CC BY-NC 4.0** (non-commercial) | **~598 GB** | Yes (4TU; also PANGAEA link) |
| **Mrzeżyno beach-dune** — Śledziowski, Landform Analysis 44 (2025), DOI 10.12657/landfana-044-003 | PL (S Baltic) | UAV RGB + UAV-LiDAR | "high-res DTM" (value not stated) | **not stated** (no GCP/checkpoint metrics) | ~1 km, Rega R. mouth; **10 campaigns** Sep 2020–Nov 2022 | DTM only in paper | **CC BY-NC-ND 4.0** | n/a | **No** — no repository/DOI; data not archived |

Notes on the table:
- Kijkduin = the strongest match to the brief: sandy beach-dune, raw sub-cm point clouds, explicit
  independent RTK-GNSS + ALS control, CC BY (commercial-OK), on PANGAEA. Caveat: bare-sand vs
  vegetated is not explicitly partitioned; it is a TLS time-series (not UAV), single 190-day window.
- Noordwijk = same team/sensor, 3-yr hourly, but CC BY-**NC** and ~598 GB, and I did not capture an
  independent-checkpoint error figure. Both NL sets are TLS from a fixed high mount, oblique geometry.

## Misses / dead ends (do not repeat)

- Curonian Pilkosios/Nagliai DEM paper: no DOI resolved; no Zenodo/Dryad/Figshare/PANGAEA/ScienceBase
  deposit found; academia.edu accuracy paper 403; Klaipėda Univ. group page has no archive link.
- Zenodo web UI not readable via WebFetch (JS shell); keyword search for "Curonian/Nagliai/Pilkosios
  dune DEM" returned only unrelated dune datasets (Iceland kettle holes, Sardinia robotic dunes,
  Tottori UAV-LiDAR, Po Delta embryo dunes, Sandhills ASCII).
- OpenTopography data catalog fetch returned 0 datasets (search endpoint needs the interactive map /
  filters; USGS-3DEP & NOAA-coastal-lidar access is gated to academic community). Not exhausted —
  a manual filtered OpenTopography search remains untried.
- Crossref bibliographic query surfaced only older Curonian dune papers (Jarmalavičius et al. 2019
  Aeolian Research 10.1016/j.aeolia.2019.100542; Bitinas et al. 2018 Geol. Quarterly 10.7306/gq.1435;
  Morkunaite 2016; etc.) — none is the multi-epoch UAV DEM paper, none carries a data DOI.
- LRT news "Uncovering the secrets of the Curonian Spit dunes" (2022) = GPR marl-layer story
  (Pupienis + Jol), no DEM dataset, no repository.

## Untried within budget (leads for a follow-up)
- INQUA Peribaltic Working Group 2024 abstract volume (sisu.ut.ee PWG2024 PDF) — likely contains the
  Pilkosios DEM abstract with author/affiliation to chase the DOI + any deposit.
- Direct OpenTopography community-contributed filtered search for coastal-dune UAV/TLS.
- Semantic Scholar / VU (talpykla.elaba.lt) institutional repository for the Danielius/Pupienis
  Pilkosios paper and its data-availability statement.
