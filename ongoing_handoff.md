# Ongoing handoff: `bouquet_interface_v2`

Tracking for open work on this branch. Update this file when an item is done (tick it and add the commit) or when a new item appears.

## Branch state
- `bouquet_interface_v2` = local `develop` + the useful parts of `origin/bouquet_interface_WIP` + a reader for bouquet `bouquet_unified` archives.
- Left out as deprecated:
  - `sampling_DB.py` and in-TPS GPR rotation resampling (bouquet now perturbs rotation itself);
  - the old worker-dir/`map_object.pkl` bouquet reader;
  - the hand-copied bouquet h5 loaders.
- Full `tests/run_tests.py` suite passes (7/7, exit 0) on Perlmutter, 2026-10-06, with GPEC built from `Zeff_profile_support`. Remaining value issue: `fortran_wrappers_tests` prints `PEST3 Delta prime 21 no-wall value correct: False` (PEST3 2.57 vs reference 7.3; rdcon 7.29 and stride 7.31 match). The script does not fail on it. Not yet diagnosed.

## GPEC branch dependencies
TPS `build_GPEC(branch="develop")` is still the default. Use `--gpec-branch Zeff_profile_support` when cloning GPEC for the TPS tests. The branch is used only when `submodules/GPEC` does not exist yet.

| TPS feature | Requires | Status (2026-10-05) |
|---|---|---|
| Zeff profile: `psi_N_Zeff=`/`Zeff=` arrays in `&RDCON_CONTROL` (`wrappers/gpec_inputs.py`), plus `psi_N_Zeff`/`Zeff` read back from the rdcon netcdf (`physics/surface_terms.mre_raw_interp`) | GPEC `Zeff_profile_support` (tip `57246273`) | **Not merged into GPEC develop.** rdcon built from develop rejects the namelist. |
| At most 998 Zeff points (`ZEFF_MAX_PTS` in `wrappers/gpec_inputs.py`) | `read_var_len` buffer size, same branch (`4e1c8e23`, `4584a7af`) | Update TPS if GPEC changes it |
| `Zeff_surf` uses CubicSpline to match `rdcon/mercier.f` | Same branch | Update TPS if GPEC's interpolation changes |
| `out_ahg2msc` written to `equil.in` | GPEC `9e107194` | In develop and in `Zeff_profile_support` |
| `MRE_flag`, `geom_flag` in rdcon | `adding_MRE_terms` | In develop |
| `ishape` in `&VACDAT` | GPEC vacuum | In develop |

## Future work
- [ ] **GPEC `Zeff_profile_support` → develop.** Until it merges, build GPEC with `branch="Zeff_profile_support"` for TPS tests. When it merges, check that these are unchanged:
  - the namelist and netcdf variable names (`psi_N_Zeff`, `Zeff`);
  - the `read_var_len` length limit (TPS assumes 998 or fewer points);
  - `mercier.f`'s Zeff interpolation (TPS assumes a cubic spline);
  - the `MRE_flag`/`geom_flag` defaults. `57246273` turned them off, so TPS must keep writing them explicitly.
- [ ] **Re-run the TPS tests** against the new GPEC and refresh the reference values (last set in `ebf8ce2`). The Akima spline switch is expected to shift them slightly.
- [ ] **Keep `wrappers/gpec_inputs.py` in sync** with any GPEC change to the `&RDCON_CONTROL`, `&EQUIL_CONTROL` or `&VACDAT` namelists.
- [ ] **Φ_N bouquet archives** (`profile_coord='phi_n'`). `read_bouquet_archive` currently raises `NotImplementedError`. Map Φ_N → ψ_N per draw using that draw's eqdsk q or `eq_fsa/q`.
- [ ] **Drop the 998-point Zeff check** once GPEC allocates the Zeff arrays dynamically.
- [ ] **Bouquet in the TPS environment.** `read_bouquet_archive` lazy-imports `bouquet` (branch `bouquet_unified`, schema v2). bouquet is not on PyPI, so it is not listed in `pyproject.toml`. genvenv1 currently has bouquet installed editable from a working tree on `parallel_extensions` (1.3.0); the reader was tested against an exported `bouquet_unified`.
- [ ] **End-to-end bouquet run.** Do a 2-case `multi_run_` from bouquet's `tests/golden/D3Dlike_Hmode_golden_slim.h5`, then `multi_compile_zarr`, with `meta_data_dicts=profile_list`.

## Behaviour changes to be aware of
- `combined_xr` `r`/`r_prime` are now positional indices; the real values are in `r_value`/`r_prime_value`. Use `sel_rational(ds, value)` or `collapse_to_primary(ds)` from `physics/combine.py` to select by value.
- `extract_mre_factors` no-root sentinels:
  - always decaying → w_marg=1, w_sat=0;
  - always growing → w_marg=0, w_sat=1;
  - peak at w=1 → w_max=NaN.
- `multi_run_` defaults to `warm_start=True`, so cases with an existing `done_{idx}.marker` are skipped.
- Output file names strip only `.geqdsk`, `.eqdsk` and `.gfile` (`utils.eq_stem`).

## Building on NERSC Perlmutter
Default modules: PrgEnv-gnu, gcc-native/14, cray-mpich. Build from a compute node:
```bash
export UV_CACHE_DIR=$PSCRATCH/uv-cache UV_LINK_MODE=copy HDF5_USE_FILE_LOCKING=FALSE
uv run tearing_physics_suite/wrappers/build/build_tearing_physics_suite.py --skip-libs \
    --work-dir $PSCRATCH/tmdb/build/tps --gpec-branch Zeff_profile_support
```
- **CFS is too slow for compiling.** `--work-dir` (or `TPS_BUILD_DIR`) compiles PEST3 and GPEC there and copies the executables back into `submodules/`. A full build takes about 15 min: PEST3 about 5 min with `-j1`, GPEC about 4 min, plus the install tests.
- **GPEC compilers.** GPEC gets plain `gfortran`/`gcc`, because its makefile rejects `mpif90`/`mpicc`. PEST3 still uses the MPI wrappers. The env file now exports the plain compilers.
- **GCC 14.** PEST3's C is configured with `-fpermissive` for GCC >= 14 (implicit declarations in portlib).
- **GPEC install test.** On `Zeff_profile_support` the DIIID examples still have the legacy g147131 inputs, so the "current" test checks them against the legacy references. It prints a NOTE when it does this.
