# tearing-physics-suite (TPS): notes for Claude

## Layout (keep it)
- `tearing_physics_suite/physics/`: analysis and physics on datasets (Delta' coupling, MRE terms, rotation,
  global quantities, combining codes). Never runs a code. Imports only `utils` and other `physics` modules.
- `tearing_physics_suite/wrappers/`: write inputs for, run, and read RDCON (`rdcon.py`), STRIDE (`stride.py`),
  PEST3 (`pest3.py`); `gpec_inputs.py` writes the GPEC namelists; `run_codes.py` runs them all;
  `build/` builds the codes and libraries. Imports only `utils` and other `wrappers` modules.
- `tearing_physics_suite/drivers/`: the run pipeline (`pipeline.py`), parallel runs and zarr compilation
  (`multi_run.py`, `zarr_store.py`), profile readers, input scans and the input test suite. May import anything.
- `tearing_physics_suite/utils.py`: small shared helpers (`tps_home()`, `eq_stem`, ...).
- `tests/unit/test_layering.py` enforces the import direction; don't work around it.

## The user's preferences
- Minimal, succinct docs and comments. Keep the user's existing wording; you may add missing parameters.
- No changes to physics methods. Any change to computation or data structure needs: a unit test, a golden
  comparison on real data, and a line in `docs/refactor_changes.md`.
- New code reads like the surrounding code.

## Tests
- `pytest` runs the fast unit tests (no Fortran). Markers: `fortran`, `julia`, `parallel`, `slow`.
- Golden reference: `tests/golden/cases.py` writes outputs; `tests/golden/compare.py` diffs them (exact by
  default; run timings ignored). `tests/integration/test_golden.py` reruns the cases against `TPS_GOLDEN_DIR`.
- Marked tests run on compute nodes as background `sbatch` jobs, only on the user's go. Size the request to
  the job: tests 1-2 CPUs (multi_run golden case: 2), about 8-16 GB.
- On omega, multi_run worker dirs must be on local disk (`.nfs*` files make `_clean_working_dir` fail on NFS).
- Namelist writers: `tests/unit/wrappers/test_gpec_inputs.py` checks byte-identical output against fixtures.
  After changing a writer or a code's branch, run
  `python -m tearing_physics_suite.wrappers.check_namelists --all-paths`.

## xarray conventions
- Put arrays on a template's dims with `physics.xr_utils.like(values, template)`, not `values + 0.0*template`.
- Interpolate profiles onto the rational surfaces with `interp_to_surfaces`.
- Prefer one `assign(**dict)` and vectorised `interp`/`apply_ufunc` over loops of scalar `.sel`/`.interp`,
  positional fill buffers, and `expand_dims`+`concat` chains.
- `r`/`r_prime` are positional indices; real values are `r_value`/`r_prime_value` (`sel_rational`,
  `collapse_to_primary` in `physics/combine.py`).

## Adding a code (e.g. jGPEC)
- Per-code datasets go through `add_code_dim` and `combine_codes` (`physics/combine.py`); the order of the
  dict is the order on the `code` dim. Grep for hard-coded code names ('rdcon', 'stride', 'pest3').

## Environment and git
- Python: `/fusion/projects/tmdb/scripts/eq_stab/venvs/genvenv1/.venv`. `TPSHOME` (from
  `tearing_physics_suite_env.sh`) is read lazily via `utils.tps_home()`; never read it at import time.
- Fortran GPEC default branch: `OFT_interface` (`submodules/GPEC`); PEST3: `master`.
- Make changes on a new branch cut from the checked-out branch, in a worktree if others are working;
  merge or push only on the user's go.
- Open work: `ongoing_handoff.md`. Change log: `docs/refactor_changes.md`. Cross-project plan:
  `/fusion/projects/tmdb/src/tps_meta_plan/META_PLAN.md`.
