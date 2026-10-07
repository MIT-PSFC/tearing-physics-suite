# Refactor change log (meta-plan item A)

Every change that alters computation, structure or behaviour, with its test and the golden-reference result.
Golden reference: `tests/golden/cases.py` run on `bouquet_interface_v2` 640310f; diff with `tests/golden/compare.py`
(exact match; run timings `cpu_time`/`wall_time` are ignored).

| # | Item | Change | Why | Test | Golden diff |
|---|---|---|---|---|---|
| 1 | A1 | Modules moved into `physics/`, `wrappers/`, `drivers/` (path map in the TPS meta-plan, `tps_meta_plan/META_PLAN.md` §5). Function bodies unchanged. | Layout | `tests/unit/test_layering.py`, all modules import | Identical (4 cases + zarr) |
| 2 | A3 | `input_test_suite.py`: 28 scan functions replaced by the `SCANS` registry and `run_scan(name, ...)`; the runner uses it. | Readability, one place to add a scan (item E) | `tests/unit/drivers/test_input_test_suite.py` (same `scan_1D_input` calls as the recorded old functions) | n/a (no physics) |
| 3 | A3 | Scans now forward the caller's `output_prefix` (before any `no_wall_` prefix). | Bug: it was accepted and silently dropped | same test, `prefix` cases | n/a |
| 4 | A3 | Full `matching_point_scan`/`psilow_truncation_scan` use `create_dense_log_paramvals`. | Bug: `tps.` was undefined (NameError) | same test, `full` cases | n/a |
| 5 | A3 | `tests/run_tests.py` and the script tests replaced by pytest (`tests/unit`, `tests/integration`). | Test suite | `pytest` | n/a |
| 6 | A2.1 | `combine_codes` (physics/combine.py) replaces the duplicated per-code combine blocks in `compile_xarrays` and `analyse_with_mre`. On a concat failure, codes are added one at a time and the code that fails is left out by name (was: always drop the last dataset, assumed PEST3; a failing non-PEST3 code then raised or gave None). | Robust to code order; needed for jGPEC (item B) | `tests/unit/physics/test_combine.py::test_combine_codes_*` | expected identical (PEST3-last behaviour unchanged) |
| 7 | A2.1 | `compile_xarrays`/`analyse_with_mre` no longer modify the caller's datasets (attrs copied into variables) or input dicts (merged into a new dict). | Side effects | `test_add_code_dim_does_not_mutate`, `test_merge_input_dicts` | expected identical |
| 8 | A2 lint | ruff config; automatic fixes (import order, `is not`, raw regex strings, whitespace) and hand fixes (unused variables, semicolons, lambda assignment, `stacklevel`, `raise ... from None`). | Lint | unit tests, namelist fixtures byte-identical | expected identical |
| 9 | A2.4 | `TPSHOME` is read lazily by `utils.tps_home()`; default paths (`working_dir`, `gpec_dir`, `pest3_dir`, `results_dir`) are `None` and resolved at call time. Modules now import without `TPSHOME`, and a `TPSHOME` change after import is respected. | Import-time side effects (B008) | `test_layering.py::test_package_imports_without_tpshome` | expected identical |
| 10 | A2.4 | List defaults (`nvec`, `q_surfs_of_interest`, `psi_surfs_of_interest`) are `None` and set to the same lists inside the function. | Shared mutable defaults (B006) | golden | expected identical |
| 11 | A2.2 | `like(values, template)` (physics/xr_utils.py) replaces the `values + 0.0*template` idiom (~70 sites in surface_terms, rotation, cross_field_transport, delta_prime_extraction). Same values, dims, coords, NaN pattern and float64 dtype. | Readability | `tests/unit/physics/test_xr_utils.py`; A/B of the full surface-term chain on stored RDCON output: identical (values, dtypes, variable order) | expected identical |
| 12 | A2.2 | `mre_raw_interp`: 24 copy-pasted `interp` calls -> one `_RAW_TO_SURF` table and `interp_to_surfaces`. | Readability | A/B identical | expected identical |
| 13 | A2.2 | rotation: one surface-interp call and one Akima-derivative helper in `put_drift_rotation_on_surfaces`; one `assign` per key in `add_rotation`/`decorrelation_ratios`; `decorrelation_timescales` interpolates all reference surfaces at once (was a nested loop of scalar interps + concat). | Speed/readability | A/B identical, including the no-q-surface (NaN) path | expected identical |
| 14 | A2.2 | `extract_delta_primes`: the expand/concat/assign chains for real/imag/err/perr -> one `_by_type` helper. | Readability | A/B identical on rdcon/stride/pest3 fixtures, with and without a `code` dim | expected identical |
