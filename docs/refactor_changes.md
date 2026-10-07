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
