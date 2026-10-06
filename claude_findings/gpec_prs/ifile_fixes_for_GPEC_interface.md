# OFT PR report: `ifile_fixes_for_GPEC_interface` → `main`

- **Branch:** `ifile_fixes_for_GPEC_interface`, four commits on `main` `025debf`.

## Why
The i-file that `save_ifile` writes is read by GPEC's `eq_type='ldp_i'` (`inverse.f`), and that path gave a high Grad–Shafranov error (GSE) and nonsense Δ′. The diagnosis below traced this to the file itself:
- `gs_save_ifile` places R,Z about 2e-7 m off the ψ = ψ_k surfaces (median), with points up to 5e-5 m off. This is surface-tracing error, resampled through a spline.
- Second derivatives in ψ amplify these errors, so the file's own GS residual grows with resolution: 2.4e-3 at npsi = 65, 2.1e-2 at npsi = 257.

## Changes
1. **`gs_save_ifile`** (`7110b88`)
   - After tracing, each point is moved onto its exact surface by Newton iteration on the FE ψ. The search runs along a ray from the axis at angle exactly 2πk/(ntheta−1).
   - F·dF/dψ and dP/dψ (ψ in Wb/rad, P in Pa) are written as two extra records **after** the R and Z records, so older readers that stop after Z are unaffected.
   - Fixes:
     - the `pack_lcfs` optional argument is now read through `do_pack`;
     - the Qmin/Qmax debug print uses the correct column;
     - the abort label now names `gs_save_ifile`.
2. **Python** (`35a7dd2`)
   - `util.read_ifile` returns `ffp` and `pp` when those records are present.
   - `save_ifile` defaults change to `npsi=129, ntheta=257`, the best accuracy for the file size below.
3. **Test** (`7f6aaf4`): `test_ITER_eq` now also checks that ψ at the i-file's R,Z points matches its ψ grid, and that the FF′ and P′ records match `get_profiles`, each to 1e-10.
4. **Fallback** (`9243fbd`): if the Newton snap leaves the mesh or does not converge, `ifile_snap` keeps the traced point instead of using an unconverged one.

## Results
All numbers come from one TokaMaker reference equilibrium (DIII-D-like, q0 = 1.25, q95 = 4.49; F, F′, p, p′ known exactly at any ψ), written with this branch and, for "before", with `main` `025debf`. `lcfs_pad` = 1e-3.

**Distance of the i-file points from the exact ψ crossings** (Newton on TokaMaker's FE ψ along each ray, `diag_ifile_noise.py`), over all 12 files below:

| | median | max |
|---|---|---|
| before (`main`) | 1.8–2.2e-7 m | 4.6e-5 m |
| this branch | 6e-12 m | 1.1e-10 m |

**The file's own GS residual** (median, 0.1 < ψ_N < 0.9; computed independently of GPEC, `ifile_tools.py gs_residual`), this branch (before):

| npsi \ ntheta | 65 | 129 | 257 | 513 |
|---|---|---|---|---|
| 65 | 2.8e-4 (2.4e-3) | 2.3e-4 (2.4e-3) | 3.2e-4 (3.0e-3) | 4.3e-4 (3.6e-3) |
| 129 | 3.1e-4 (5.6e-3) | 2.7e-4 (5.6e-3) | 3.4e-4 (6.2e-3) | 4.1e-4 (6.5e-3) |
| 257 | 3.6e-4 (2.1e-2) | 3.3e-4 (2.1e-2) | 3.9e-4 (2.1e-2) | 4.4e-4 (2.1e-2) |

The residual is now flat with resolution, at the FE limit, and equal to that of points placed on the exact crossings. The FF′ and P′ records agree with the reference profiles to 2e-5, the accuracy of the reference interpolation (the test checks them against `get_profiles` to 1e-10).

**GPEC** (`OFT_interface` branch, STRIDE, `profile_source = integrate`, last surface at the same true ψ_N = 0.985; `run_ifile_comparison.py`). The "before" i-files have no FF′/p′ records, so the reference profiles' are appended. Integrated GSE is the θ-integrated residual over the θ-integrated source, median over 0.05 < ψ_N < 0.95.

| file | size | Δ′(2/1), mpsi = 128 | mpsi = 512 | GSE local / integrated (mpsi = 128) |
|---|---|---|---|---|
| i-file 65×257 | 0.27 MB | 8.59 | 8.60 | 1.5e-4 / 4.2e-5 |
| i-file 129×257 (new default) | 0.54 MB | 8.60 | 8.60 | 1.5e-4 / 2.4e-5 |
| i-file 257×513 | 2.1 MB | 8.59 | 8.62 | 1.7e-4 / 3.0e-5 |
| *before:* i-file 65×257 | 0.27 MB | 8.72 | 8.74 | 7.3e-4 / 1.6e-3 |
| *before:* i-file 129×257 | 0.53 MB | 1.63 | 11.22 | 1.5e-3 / 5.9e-3 |
| *before:* i-file 257×513 | 2.1 MB | 30.41 | −2.55 | 2.1e-3 / 7.4e-3 |
| g-file 129×129 (`efit`) | 0.29 MB | 8.60 | 8.72 | 1.7e-4 / 6.6e-5 |
| g-file 257×257 | 1.1 MB | 8.56 | 8.71 | 2.0e-4 / 9.0e-5 |
| g-file 513×513 | 4.3 MB | 8.52 | 7.96 | 2.1e-4 / 1.2e-4 |

- Before, the i-file Δ′ was meaningless at the finer resolutions and its integrated GSE was 15–100× the g-files'.
- With this branch, every i-file gives Δ′(2/1) = 8.59–8.62 at both mpsi, and an integrated GSE 1.6–5× below the g-files'.
- **A 129×257 i-file (0.54 MB) beats a 513×513 g-file (4.3 MB).**

## Verification
On this branch, built from HEAD `9243fbd`:
- `test_TokaMaker.py -k "test_ITER_eq and not io"`: 4/4 pass; the same tests fail on an install without these commits.
- Full `test_TokaMaker.py`: **106/106 pass** (5.7 min).
- The results above were produced with this build and, for "before", with a build of `main` `025debf`.

## Reproduce
The scripts and inputs ship as `ifile_fixes_for_GPEC_interface_scripts.zip`; its `README.md` has the commands.

## Status
- [x] Commits on `main`, no conflicts
- [x] Built; full `test_TokaMaker.py` passes (106/106)
