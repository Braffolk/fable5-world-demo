# Ground-cover asset status

## Runtime/published

- `estonia-native-groundcover.gcar` and the ordinary small `*.gcrp` profiles feed the deprecated
  but still-live default multi-species compatibility renderer.
- `calamagrostis-canescens.gcrp` is a generated high-resolution input for the isolated
  `grassprofile=2` preview. It is about 114 MiB, remains at this local URL for development, and is
  intentionally ignored/untracked. A clean checkout must reproduce or supply it before using that
  preview; it must never be committed as a regular Git blob.

## Validation only

- `calamagrostis-canescens.gcc1`
- `calamagrostis-canescens.reference-v4.gbr4`

These validate historical carrier/GBR4-v4 records. They do not enable a boundary-transfer renderer.

## Rejected/inactive outputs

The non-reference `calamagrostis-canescens.gbr4` and 131 MiB
`calamagrostis-canescens.gbc2` were moved to ignored
`data/work/groundcover-rejected-inactive/assets/`. Their production representations are RED and
they must not be copied back into this directory or committed.
