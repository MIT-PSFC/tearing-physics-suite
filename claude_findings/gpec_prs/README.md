# GPEC/OFT/bouquet/TPS equilibrium-interface work: handoff

The overarching plan is `~/.claude/plans/warm-gliding-hartmanis.md`. It covers Issue 1 (profiles from FF′ and p′), Issue 2 (splines) and Issue 3 (the OFT inverse interface).

## State (2026-10-06, end of session 1)
| Step | Branch (tip) | Status |
|---|---|---|
| Phase 0: merge `develop` and `bugfix/dcon-vacuum-theta-frame` | GPEC `Zeff_profile_support` `99378252` | Done. Built into `submodules/GPEC/bin`; TPS 7/7. [report](Zeff_profile_support.md) |
| Phase A + B.0: profiles from FF′/p′, Hermite and pchip fits, dump fix | GPEC `spline_improvements` `7c44ec05` | Done. [report](spline_improvements.md) |
| B.1: OFT i-file on exact surfaces, FF′/p′ records | OFT `GPECf_interface` `7deb6f8` | Done; OFT tests 114/114. **Merge into PSFC_dev blocked by a permission check; needs the user.** [report](GPECf_interface.md) |
| B.2: `ldp_i` takes \|F\| | GPEC `OFT_interface` `eb9e9fc1` | Done. [report](OFT_interface.md) |
| B.3: `write_ifile` option | bouquet `OFT_inverse` `7df036f` | Done; full bouquet suite passes |
| B.4: TPS `eq_source='ifile'`, `ldp_i` from extension, pchip kinetics and Zeff | TPS `bouquet_interface_v2` `a522e96` | Done; TPS 7/7 |
| B.5: end-to-end and kinetics checks | — | Done (below) |

- **Code review (session 2):** one fix commit per repo, verified as below.
  - GPEC `7c44ec05`: profile-derivative sign from f and p together (a pressureless or flat-f equilibrium no longer falls back to `values`); dump writes `eqfun` only if allocated; doc and test tidy. Unit test passes; TPS end-to-end Δ′ identical on `spline_improvements` and on `OFT_interface` (rebased, `eb9e9fc1`).
  - OFT `7deb6f8`: `ifile_snap` keeps the traced point if Newton leaves the mesh or does not converge. Full `test_TokaMaker.py` 114/114.
  - bouquet `7df036f`: `try_save_ifile`, so a failed baseline i-file no longer aborts the warmstart eqdsk re-save. `test_ifile.py` 4/4; a fresh 2-draw run archived the baseline and both draw i-files and ran through TPS (draws are unseeded, so Δ′ differs from the earlier run: g-file 17.36/17.37, i-file 17.08/17.10).
- Nothing has been pushed.
- Worktrees:
  - GPEC: `$PSCRATCH/tmdb/build/gpec_wt/{spline_improvements,OFT_interface}`
  - OFT: `$TMDB_SRC/.oft_wt/GPECf_interface`, installed at `$PSCRATCH/tmdb/soft/oft/install_GPECf_interface`
  - bouquet: `$TMDB_SRC/.bouquet_wt/OFT_inverse`
- TPS still runs the `Zeff_profile_support` GPEC binaries and OFT `install_release`. Switching either is the user's call.

## Key results
1. **`profile_source = integrate` (PR #506's method) is the most accurate and is now the default.**
   - Measured against exact TokaMaker profiles, its F″ error is 0.1–0.8%. `values` gives 2–29% and `hermite` (values + slopes) gives 5–47%, and both get worse as the grid is refined.
2. **Why the `ldp_i` GSE was high: the OFT i-file, not GPEC.**
   - `gs_save_ifile` puts R,Z about 2e-7 m off the true ψ crossings (median), with occasional points up to 1e-3 m off.
   - Second derivatives amplify those errors, so the GSE and Δ′ get *worse* with npsi. At 257×513, Δ′(2/1) comes out as −2.5 where it should be about 8.5.
   - With the points moved onto the exact crossings, the i-file path beats the g-file path:
     - integrated GSE is 3e-5, against 7e-5 to 1.2e-4 for the g-file;
     - Δ′ is 8.4–8.6 at every mpsi;
     - file size is 0.54 MB at 129×257, against 1.1 MB for g257.
3. **The g-file path is still noisy at mpsi ≥ 256 after the profile fix.** The single-precision ψ(R,Z) table limits it.

## Phase B plan (approved 2026-10-06)
User decisions: GPECf_interface gets a PR report kept current ([GPECf_interface.md](GPECf_interface.md)) and is
merged into PSFC_dev when done; TPS uses pchip like bouquet; upgrade the RDCON Zeff and other derivative-free
profile spline pathways; TPS reference values may move; dump keeps the slopes; PEST3 no-wall Delta' later.

**B.00 GPEC bugfix merge** (user request): `origin/bugfix/dcon-vacuum-theta-frame` (3 commits, DCON/RDCON/STRIDE
`free.f`: vacuum matrix in the plasma's Fourier frame) merged into `Zeff_profile_support` (`99378252`) and forward
into `spline_improvements` (`0bb9a111`); `OFT_interface` branches from there. Clean merges.

**B.0 GPEC `spline_improvements` (additions)**
- `equil_out_dump` appends `sq%fs1` after the existing records; `read_eq_dump` reads it if present and keeps the
  f, p slopes (old dumps still read).
- `spline_fit_pchip`: monotone cubic Hermite (Fritsch-Carlson) for tabulated data without derivatives. Used for
  the RDCON Zeff profile and the PENTRC kinetic profile table, replacing cubic splines that ring at the pedestal.

**B.1 OFT `GPECf_interface`** off `PSFC_dev`, worktree `$TMDB_SRC/.oft_wt/GPECf_interface`, own build and install
(the main tree and install_release are untouched until the final merge).
1. `gs_save_ifile`: refine every traced point to the exact psi = psi_k crossing along its ray (Newton on the FEM
   psi). Acceptance: the file's own GS residual stays flat with npsi.
2. Append FF' and p' records (real*8, vs psi in Wb/rad); `read_ifile` reads them when present.
3. Fixes: `IF(pack_lcfs)` -> `do_pack`; q min/max index; abort label.
4. Defaults npsi = 129, ntheta = 257.
5. pytest: i-file round trip, GS residual check.
6. Merge into PSFC_dev when done.

**B.2 GPEC `OFT_interface`** off `spline_improvements`: `read_eq_ldp_i` takes |F| like `read_eq_efit` (inverse.f's
q is proportional to F); format documented.

**B.3 bouquet `OFT_inverse`** off `bouquet_unified`: `GenerationConfig.write_ifile` (+ `ifile_npsi`, `ifile_ntheta`),
`save_ifile` beside `safe_save_eqdsk`, `ifile` bytes dataset, `ifile_bytes` accessor, pytests.

**B.4 TPS `bouquet_interface_v2`**: `read_bouquet_archive(eq_source='geqdsk'|'ifile')`; `eq_type="'ldp_i'"` with
`psihigh` converted to the i-file's padded edge; kinetic interpolants `PchipInterpolator` (as bouquet);
`Zeff_surf` matches GPEC's pchip.

**B.5 Checks**: bouquet draw -> ifile -> TPS -> rdcon/stride end to end; sum n*T against OFT p(psi).

## Phase B results
- **OFT i-file:** the file's own GS residual is now 2–4e-4 (the FE limit) at every resolution; it was 2e-3 to 1.8e-2 before. GPEC's GSE is at or below the g-file path's.
  Δ′(2/1) is 8.51 for the 129×257 i-file at mpsi 128, and 8.53 at mpsi 512. Details in [GPECf_interface.md](GPECf_interface.md).
- **TPS end to end** (`scripts/tps_ifile_e2e.py`, nn = 1, mpsi = 257, TPS defaults):

  | file | RDCON Δ′(2/1) | STRIDE Δ′(2/1) |
  |---|---|---|
  | g257 g-file | 8.400 | 8.415 |
  | i129x257 i-file (picked up as `ldp_i`) | 8.246 | 8.261 |

- **Real bouquet run** (`scripts/bouquet_ifile_run.py`, D3D-like example, 2 draws, `write_ifile=True`, 2.4 min):
  - Every draw stored its eqdsk (1.1 MB) and its i-file (0.54 MB).
  - TPS `read_bouquet_archive(eq_source=...)` and RDCON/STRIDE ran on draw 0:

    | source | RDCON Δ′(2/1) | STRIDE Δ′(2/1) |
    |---|---|---|
    | g-file | 7.88 | 7.89 |
    | i-file | 8.07 | 8.08 |

  - The i-file result is the one that stays put as mpsi changes (see the reports).
- **Kinetics (Issue 3.3)** (`scripts/kinetic_pressure_check.py`, golden archive, 3 draws):
  - TPS's pchip splines give e(n_e T_e + n_i T_i), which matches bouquet's `pressure_thermal` (the solve's pressure without impurity and fast ions) to 1.5e-5 of the peak.
  - pchip and Akima differ by only 2e-6 on the dense kinetic grid.
  - pchip is used everywhere now, to match bouquet's `pp_prof` and GPEC's Zeff fit.
- **Dump path:** fixed and verified exact for DCON (see the spline report).

## Open items
- **Merge `GPECf_interface` into `PSFC_dev`.** This was blocked by a permission check, so it needs the user. Rebuilding `install_release` afterwards is also the user's call.
- **TPS on the new GPEC:** point `submodules/GPEC` at `OFT_interface` (or rebuild it), then refresh the TPS reference values. The user has accepted that these will move.
- PEST3 no-wall Δ′: 2.57 against 7.3 (deferred by the user).
