# OFT PR report: `GPECf_interface` → `PSFC_dev`

- **Branch:** `GPECf_interface`, branched from `PSFC_dev` `8b2510a`. Local only; it will be merged into `PSFC_dev` when Phase B is done.
- **Worktree:** `$TMDB_SRC/.oft_wt/GPECf_interface`.
- **Build:** `oft_build_local.sh wt:GPECf_interface` with `TMDB_SOFT=$PSCRATCH/tmdb/soft`. This installs to `$PSCRATCH/tmdb/soft/oft/install_GPECf_interface`; `install_release` is untouched.

## Why
The i-file that `save_ifile` writes is read by GPEC's `eq_type='ldp_i'` (`inverse.f`), and that path gave a high Grad–Shafranov error (GSE) and nonsense Δ′. The A6 study in [spline_improvements.md](spline_improvements.md) traced this to the file itself:
- `gs_save_ifile` places R,Z about 2e-7 m off the ψ = ψ_k surfaces (median), with occasional points up to 1e-3 m off. This is surface-tracing error, which is resampled through a spline.
- Second derivatives in ψ amplify these errors, so the file's own GS residual grows with resolution: 2e-3 at npsi = 65, 1.8e-2 at npsi = 257.

## Changes
Four commits:
1. **`gs_save_ifile`** (`df92ada`)
   - After tracing, each point is moved onto its exact surface by Newton iteration on the FE ψ. The search runs along a ray from the axis at angle exactly 2πk/(ntheta−1).
   - F·dF/dψ and dP/dψ (ψ in Wb/rad, P in Pa) are written as two extra records **after** the R and Z records, so older readers that stop after Z are unaffected.
   - Fixes:
     - the `pack_lcfs` optional argument is now read through `do_pack`;
     - the Qmin/Qmax debug print uses the correct column;
     - the abort label now names `gs_save_ifile`.
2. **Python** (`d957919`)
   - `util.read_ifile` returns `ffp` and `pp` when those records are present.
   - `save_ifile` defaults change to `npsi=129, ntheta=257`, the best accuracy for the file size in A6.
3. **Test** (`164d4df`): `test_ITER_eq` now also checks that ψ at the i-file's R,Z points matches its ψ grid, and that the FF′ and P′ records match `get_profiles`, each to 1e-10. The test fails against the old install, as it should.
4. **Review fix** (`7deb6f8`): if the Newton snap leaves the mesh or does not converge, `ifile_snap` keeps the traced point instead of using an unconverged one.

## Results
Same TokaMaker truth equilibrium as in A6. The file's own GS residual (median, 0.1 < ψ_N < 0.9):

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

## Verification
- `test_TokaMaker.py -k "test_ITER_eq and not io"`: 4/4 pass on the new install. On the old install the order-3 cases fail, as they should.
- Full `test_TokaMaker.py` on the new install, built from HEAD `164d4df`: **114/114 pass** (8.4 min).
- After the review fix, rebuilt from HEAD `7deb6f8`: full `test_TokaMaker.py` **114/114 pass** (6.8 min).
- bouquet: the full test suite on `OFT_inverse` with this install passes (all tests, no failures).

## Status
- [x] Code and tests committed
- [x] Full test suite (114/114)
- [ ] Merge into `PSFC_dev`: the merge in the main OFT tree was blocked by a permission check. It needs the user's go-ahead or for them to run it: `cd $TMDB_SRC/OpenFUSIONToolkit && git merge --no-ff GPECf_interface`. This is a fast-forwardable merge onto `8b2510a`. `install_release` has not been rebuilt.
