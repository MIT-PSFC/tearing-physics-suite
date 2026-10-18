# Compiled zarr: changes from item F

Old name / dims -> new, for analysis scripts. Per-run files (`result_{idx}.pkl`, `combined_xr_{idx}.nc`) keep
everything except the renames in rows 1-3 and the dims in row 9.

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
| 9 | RDCON-only variables `(…, code, …)` with NaN in every other code: RDCON's surface terms (what `analyse_with_mre` adds to RDCON's output: `X0_surf`, `Dnc_surf`, χ terms, rotation and decorrelation terms, …), `avg_*` and `combine.RDCON_ONLY` (`Dnc`, `Wc`, `W_t`, `tau_a`, …) | the same names without `code` | `ds.sel(code=...)` on the Dataset still works; `ds[var].sel(code='rdcon')` on one of these raises, use `ds[var]`. The set is fixed by name, so a run where STRIDE fails has the same dims |
| 10 | full-grid input profiles `ne_m3`, `ni_m3`, `te_keV`, `ti_keV`, `Er`, `omega_*`, `dq_dpsi_n`, `dmu0p_dpsi_n`, `eta_spitz`, `lnLamb_e*` `(…, psi_n)` | not stored | `store_schema.INPUT_PROFILES`; rerun `mre_terms_on_modes` on the run's input profiles; the `*_surf` values stay |
| 11 | float32 profiles on `psi_n` | bit-rounded to 12 mantissa bits | relative error <= 1.2e-4 (`store_schema.PROFILE_KEEPBITS`); not with `store_float32=False` |

Store attributes `store_float32`, `drop_duplicates` record the options used.
