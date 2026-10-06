# GPEC/OFT/bouquet/TPS equilibrium-interface work: handoff

The overarching plan is `~/.claude/plans/warm-gliding-hartmanis.md`. It covers Issue 1 (profiles from FF′ and p′), Issue 2 (splines) and Issue 3 (the OFT inverse interface).

## State (2026-10-06)
| Step | Branch | Status |
|---|---|---|
| Phase 0: merge `develop` into `Zeff_profile_support` | GPEC `Zeff_profile_support` `0b4a7720` | Done. Built into `submodules/GPEC/bin`; TPS 7/7 passes. Report: [Zeff_profile_support.md](Zeff_profile_support.md) |
| TPS test data fix | TPS `bouquet_interface_v2` `a845f72` | Done |
| Phase A: Issues 1 and 2 | GPEC `spline_improvements` `414877b2` (6 commits) | Done. Report: [spline_improvements.md](spline_improvements.md) |
| Phase B: Issue 3 | GPEC `OFT_interface`, OFT `GPECf_interface`, bouquet `OFT_inverse`, TPS `bouquet_interface_v2` | Not started. Detailed subplan below; needs approval |

- Nothing has been pushed.
- The worktree for `spline_improvements` is `$PSCRATCH/tmdb/build/gpec_wt/spline_improvements`, with its own binaries in its `bin/`.
- TPS still runs the `Zeff_profile_support` binaries.

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

## Phase B detailed subplan (proposed)
**B.1 OFT `GPECf_interface`**, off `PSFC_dev`, in a worktree with its own build and its own `BUILD_TAG` install. The main tree stays untouched.
1. **Exact points in `gs_save_ifile`.** After tracing, refine each (ψ_k, θ_j) point by Newton iteration on ψ along its ray, in the way `diag_ifile_noise.py` does, to |δψ| ≈ 1e-14.
   - Alternatively, replace the per-surface ODE with ray-wise root finding.
   - Acceptance: the file's own GS residual stays flat with npsi.
2. **Append FF′ and p′ records** (real*8, against ψ in Wb/rad). This stays compatible with old readers, which stop after Z. Extend Python `read_ifile` to read them.
3. **Small fixes** in `gs_save_ifile`:
   - line 1041 should test `do_pack`, not the optional argument;
   - the q min/max print should index `cout(:,4)`;
   - the error label should read `gs_save_ifile`.
4. **Defaults:** `npsi = 129`, `ntheta = 257`. Packing: keep the edge packing; A6 showed no gain from removing it.
5. **Test:** an OFT pytest that round-trips an i-file and checks the GS residual.

**B.2 GPEC `OFT_interface`**, off `spline_improvements`.
- `read_eq_ldp_i` already reads FF′ and p′, from Phase A.
- Take `|F|`, as `read_eq_efit` does. The reader currently keeps the sign of F, and inverse.f computes q in proportion to F, so a file with F < 0 would give negative q.
- Document the file format.

**B.3 bouquet `OFT_inverse`**, off `bouquet_unified`.
- Add `GenerationConfig.write_ifile` (default False), plus `ifile_npsi` and `ifile_ntheta`, following the `capture_live_eq` pattern.
- Call `save_ifile` next to `safe_save_eqdsk` (`TokaMaker_interface.py` around lines 6281 and 4863).
- Store the result as an `ifile` bytes dataset (a `schema.py` constant and a `store_equilibrium` keyword). Expose it as `d.ifile_bytes`.
- Add pytests.

**B.4 TPS `bouquet_interface_v2`**
- `read_bouquet_archive(eq_source='geqdsk'|'ifile')` writes `TPS_eqdsks/*.ifile`.
- `write_equil_in` / `write_rdcon_stride_inputs` set `eq_type="'ldp_i'"` and convert `psihigh` for the i-file's padded edge (ψ_N_file = ψ_N / (1 − lcfs_pad)).

**B.5 Kinetic profiles (Issue 3.3)**
- TPS interpolates kinetic profiles with Akima, while bouquet builds p′ with pchip.
- Check Σ n·T against OFT's p(ψ) on a bouquet draw, and align the interpolants if they disagree.
- Zeff reaches RDCON through a cubic `extrap` fit. Check it for ringing at the pedestal.

## Open items
- TPS reference values will move once TPS builds against `spline_improvements` (g147131 Δ′ changes from 8.00 to 8.14).
- The `eq_type = "dump"` round trip drops the supplied slopes (the dump format stores only `sq%fs`).
- PEST3 no-wall Δ′: 2.57 against 7.3 (older issue, not yet diagnosed).
