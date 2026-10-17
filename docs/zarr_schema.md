# Compiled zarr: changes from item F

Old name / dims -> new, for analysis scripts. Per-run files (`result_{idx}.pkl`, `combined_xr_{idx}.nc`) keep
everything except the renames in rows 1-3.

| # | Old | New | Notes |
|---|---|---|---|
| 1 | `di`, `dr` (PEST3 slice) `(…, code, psi_n, qprof_dim)` | `di_pest3`, `dr_pest3` `(…, code, qprof_dim)` | RDCON's `di`, `dr` are now `(…, code, psi_n)` |
| 2 | `qa` (PEST3 slice) `(…, code, qprof_dim)` | `q_pest3` `(…, code, qprof_dim)` | RDCON's `qa` (edge q) is now a scalar per (nn, code) |
| 3 | `Delta_prime` attr `long_name='PEST3 Delta Prime Matrix'` | `'Delta prime matrix'` | inherited `long_name`s removed from derived variables |
| 4 | `dwdt_surf` | not stored | `drivers.mre_recompute.dwdt_curves(ds_small, k0, k1, C0)` |
| 5 | `r_prime_value`, `r_prime_occ`, `r_prime_unique` | not stored | equal to `r_value`, `r_occ`, `r_unique`; `collapse_to_primary` handles it |
| 6 | `Delta_prime_(rel)diff_across_GPECvsPEST(_2)` | not stored when the codes are only rdcon/stride/pest3 | equal to `*_across_code*` then |
| 7 | STRIDE slice of GPEC `equil` outputs (`f`, `mu0p`, `dvdpsi`, `ca1`, `di`, `dr`, betas, `li*`, …; `store_schema.GPEC_EQUIL_VARS`) | NaN where equal to RDCON's | `dup_maxreldiff(dup_code, dup_var)` records the check; q quantities are never blanked |
| 8 | float64 results | float32 | except `mre_recompute.INPUTS` (k0/k1/C0 recompute inputs) and coordinates; `store_float32=False` keeps float64 |

Store attributes `store_float32`, `drop_duplicates` record the options used.
