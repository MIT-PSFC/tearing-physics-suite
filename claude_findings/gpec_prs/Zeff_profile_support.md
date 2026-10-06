# GPEC PR report: `Zeff_profile_support` → `develop`

**Branch tip:** `99378252` (local, not pushed). It is `origin/Zeff_profile_support` (`57246273`) plus two merge commits: `develop` (`0b4a7720`) and `bugfix/dcon-vacuum-theta-frame` (`99378252`).

## What the PR adds
- **RDCON reads a Zeff profile.** The new `&RDCON_CONTROL` arrays are `psi_N_Zeff` and `Zeff`, read with the new `read_var_len` routine (at most 1000 points). RDCON fits them with a spline in `mercier.f` and writes them to the rdcon netcdf. RDCON stops if `MRE_flag` is set but no Zeff profile was given.
- **`MRE_flag` and `geom_flag` now default to off** (`57246273`).

## Merge of `develop` (2026-10-06)
- `git merge origin/develop` (`5be646f2`, 47 commits) merged cleanly with no conflicts. Nothing under `equil/` changed.
- `develop` replaces the g147131 inputs in the DIIID examples with `TkMkr_D3Dlike_Hmode.geqdsk` and `.gpeckf`.

## Merge of `bugfix/dcon-vacuum-theta-frame` (2026-10-06, user request)
- `origin/bugfix/dcon-vacuum-theta-frame` (J. Halpern, 3 commits) puts the vacuum matrix and its norm in the plasma's Fourier frame in DCON, RDCON and STRIDE (`free.f`). It merged cleanly.
- Rebuilt into `submodules/GPEC/bin`; TPS `run_tests.py` passes 7/7.

## Verification
- **GPEC build** (TPS build script, `--remake-gpec`): passes. The install tests pass, apart from `a5_tearing` STRIDE, which is a known failure that does not block.
- **TPS `tests/run_tests.py`: 7/7.**
  - The TPS tests used to read g147131 from `submodules/GPEC/docs/examples`. After the merge they read it from `tests/data`; this is TPS commit `a845f72`, which also adds the `.kin` file.
  - The PEST3 no-wall Δ′ check still prints `False`. That failure predates this work.
- **Build quirk, not caused by this PR.** `install/TARGETS.inc` regenerates `*/version.inc` only when the file is missing, so a rebuild in an existing tree reports the old `git describe`. Delete `*/version.inc` before rebuilding.
