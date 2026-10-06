# OFT PR report: `ifile_fixes_for_GPEC_interface` → `main`

- **Branch:** `ifile_fixes_for_GPEC_interface`, off `main` `025debf`: the four `GPECf_interface` commits cherry-picked cleanly, with nothing else from `PSFC_dev`.

## Why
The i-file that `save_ifile` writes is read by GPEC's `eq_type='ldp_i'` (`inverse.f`), and that path gave a high Grad–Shafranov error (GSE) and nonsense Δ′. The diagnosis below traced this to the file itself:
- `gs_save_ifile` places R,Z about 2e-7 m off the ψ = ψ_k surfaces (median), with occasional points up to 1e-3 m off. This is surface-tracing error, which is resampled through a spline.
- Second derivatives in ψ amplify these errors, so the file's own GS residual grows with resolution: 2e-3 at npsi = 65, 1.8e-2 at npsi = 257.

## Changes
Four commits:
1. **`gs_save_ifile`** (`7110b88`)
   - After tracing, each point is moved onto its exact surface by Newton iteration on the FE ψ. The search runs along a ray from the axis at angle exactly 2πk/(ntheta−1).
   - F·dF/dψ and dP/dψ (ψ in Wb/rad, P in Pa) are written as two extra records **after** the R and Z records, so older readers that stop after Z are unaffected.
   - Fixes:
     - the `pack_lcfs` optional argument is now read through `do_pack`;
     - the Qmin/Qmax debug print uses the correct column;
     - the abort label now names `gs_save_ifile`.
2. **Python** (`35a7dd2`)
   - `util.read_ifile` returns `ffp` and `pp` when those records are present.
   - `save_ifile` defaults change to `npsi=129, ntheta=257`, the best accuracy for the file size in the diagnosis below.
3. **Test** (`7f6aaf4`): `test_ITER_eq` now also checks that ψ at the i-file's R,Z points matches its ψ grid, and that the FF′ and P′ records match `get_profiles`, each to 1e-10. The test fails against the old install, as it should.
4. **Review fix** (`9243fbd`): if the Newton snap leaves the mesh or does not converge, `ifile_snap` keeps the traced point instead of using an unconverged one.

## Results
Same TokaMaker reference equilibrium as in the diagnosis. The file's own GS residual (median, 0.1 < ψ_N < 0.9):

| npsi \ ntheta | 65 | 129 | 257 | 513 |
|---|---|---|---|---|
| 65 | 2.9e-4 (was 2.0e-3) | 2.3e-4 | 3.3e-4 | 4.3e-4 |
| 129 | 3.2e-4 (was 5.1e-3) | 2.7e-4 | 3.4e-4 | 4.1e-4 |
| 257 | 3.6e-4 (was 1.8e-2) | 3.3e-4 | 3.9e-4 | 4.4e-4 |

The residual is now flat with resolution, at the FE limit. The FF′ and P′ records agree with the truth profiles to 1e-6, which is the accuracy of the reference interpolation.

GPEC `spline_improvements`, reading `eq_type='ldp_i'` with `profile_source='integrate'`:

| i-file | size | Δ′(2/1), mpsi = 128 | Δ′(2/1), mpsi = 512 | GSE local / integrated median (mpsi = 128) |
|---|---|---|---|---|
| 129×257 (new default) | 0.54 MB | 8.51 | 8.53 | 1.6e-4 / 2.9e-5 |
| 257×513 | 2.1 MB | 8.51 | 8.60 | 1.8e-4 / 2.9e-5 |
| 65×257 | 0.27 MB | 8.38 | 8.53 | 1.5e-4 / 3.6e-5 |
| *before the fix:* 129×257 | 0.53 MB | 0.35 | — | 2.3e-3 / 7.6e-3 |
| *g-file 257×257 (efit, integrate)* | 1.1 MB | 8.59 | 8.48 (9.19 at mpsi = 256) | 1.9e-4 / 7.4e-5 |

## Diagnosis: i-file vs g-file, GSE and file size (`run_ifile_comparison.py`)
All files come from the same TokaMaker reference equilibrium (GPEC `spline_improvements`, `profile_source = integrate`), with the last surface at the same true ψ_N = 0.985 and mpsi = 128. Integrated GSE is the θ-integrated residual divided by the θ-integrated source, taking the median over 0.05 < ψ_N < 0.95.

| file | size | Δ′(2/1) | GSE local, median | GSE integrated, median |
|---|---|---|---|---|
| g129 / g257 / g513 (efit, integrate) | 0.29 / 1.1 / 4.3 MB | 8.59 / 8.59 / 8.45 | 1.6–2.2e-4 | 0.7–1.2e-4 |
| OFT i-file 65×65 / 129×257 / 257×513 | 0.07 / 0.53 / 2.1 MB | 7.44 / 0.35 / −2.50 | 1.2e-3 / 2.3e-3 / 1.3e-3 | 1.5e-3 / 7.6e-3 / 7.4e-3 |
| same, with FF′ and p′ records | same | unchanged | unchanged | unchanged |
| i-file with exact ψ crossings, 129×257 / 257×513 | 0.54 / 2.1 MB | 8.49 / 8.50 | 1.7e-4 / 1.8e-4 | 3.0e-5 / 2.9e-5 |

**Why the `ldp_i` GSE has been high: the problem is in the file, not in GPEC's reader or its GSE check.**
- The R,Z points that OFT `gs_save_ifile` writes are about 2e-7 m off the true ψ = ψ_k crossings (median), with occasional points up to 1e-3 m off. This was measured by Newton iteration on TokaMaker's own FEM ψ along each ray (`diag_ifile_noise.py`).
- The file's own Grad–Shafranov residual, computed independently of GPEC (`ifile_tools.py gs_residual`), grows as the grid is refined:

  | npsi | 65 | 129 | 257 |
  |---|---|---|---|
  | file's own residual | 2e-3 | 5e-3 | 1.8e-2 |

  Errors in individual surface positions are amplified by the second derivatives in ψ.
- When the same points are placed at the exact crossings, the residual stays at 3–4e-4 (the FEM limit) at every resolution.
- GPEC then gives GSE equal to or better than the g-file path, and the integrated GSE is 2–4× lower.
- Its Δ′ agrees with the g-file result at mpsi = 128 and stays put as mpsi is refined.
- **A 129×257 i-file (0.54 MB) beats a 257×257 g-file (1.1 MB).**

## Verification
On this branch (`main` `025debf` + the four commits, built from HEAD `9243fbd`):
- `test_TokaMaker.py -k "test_ITER_eq and not io"`: 4/4 pass. The same tests fail on an install without these commits, as they should.

The results above were measured on `GPECf_interface` (the same four commits on `PSFC_dev`), where the full `test_TokaMaker.py` passed 114/114.

## Reproduce
The diagnosis scripts and inputs ship as `ifile_fixes_for_GPEC_interface_scripts.zip`.

## Status
- [x] Commits cherry-picked onto `main`, no conflicts
- [x] Built; i-file tests pass
