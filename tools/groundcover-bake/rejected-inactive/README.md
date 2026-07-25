# Rejected and inactive ground-cover research tools

**Nothing in this directory is part of the active bake or runtime toolchain.** These files are
preserved to reproduce failed gates and avoid repeating rejected work. They are excluded from the
application TypeScript build so historical experiments cannot break production typechecking.

- `class-e/` — rejected concrete Class-E compilation and CPU gate.
- `gbc2-k6/` — rejected GBC2/K6 codec cooks and training scripts.
- `gbr4-production/` — rejected GBR4 production cooks. The small v4 runtime validator outside this
  directory remains validation-only; it does not make this renderer active.
- `candidates-h-q/` — rejected GrassProfile2 Candidate H–Q gates and fits.
- `boundary-transfer/` — parked/rejected box-boundary transfer analysis.
- `representation-gates/` — older ray/light-field/event/relay representation experiments.

The scripts retain their imports and repository-relative output paths for reproducibility. Resume
one only after an explicit active task invalidates its recorded blocker.
