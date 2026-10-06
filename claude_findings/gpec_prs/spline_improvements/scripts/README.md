# spline_improvements: test, data-generation and plotting scripts

Scripts behind the evidence in the `spline_improvements` PR. Python 3 with numpy, scipy, xarray and
matplotlib; OpenFUSIONToolkit's TokaMaker on `PYTHONPATH` for `make_truth_equilibrium.py`,
`diag_ifile_noise.py` and `make_figures.py`. `BIN` is the `bin/` directory of a GPEC build of this branch.

## Inputs (`inputs/`)
- `D3Dlike_Hmode_baseline.geqdsk`: TokaMaker D3D-like H-mode; supplies the FF′/p′ shapes, Ip, F0 and boundary.
- `DIIID_mesh.h5`: the TokaMaker DIII-D mesh used for the solve.
- `g147131.02300_DIIID_KEFIT`: the classic EFIT g-file formerly in GPEC's DIIID examples.

## Scripts
| script | role |
|---|---|
| `make_truth_equilibrium.py` | Data. Solves the reference TokaMaker equilibrium (C2 profiles on 4001 nodes, so F, F′, F″, p′ are known at any ψ) and writes the truth profiles, g-files at 129/257/513, and i-files over npsi × ntheta, packing and padding. |
| `run_profile_source_comparison.py` | Test. STRIDE Δ′ and GPEC GSE for each `profile_source` on the truth g-files, GPEC's TkMkr example and g147131. |
| `run_ifile_comparison.py` | Test. i-file (`ldp_i`) against g-file (`efit`): GSE, Δ′ and file size; each i-file as written and with appended FF′/p′ records. |
| `diag_ifile_noise.py` | Test/data. Measures how far the i-file R,Z points sit from the exact ψ crossings (Newton on TokaMaker's FE ψ), the file's own GS residual, and writes `*_exact.ifile` copies on the exact crossings. |
| `run_exact_ifiles.py` | Test. GPEC on the `*_exact.ifile` files. |
| `make_figures.py` | Plots the three report figures. |
| `gpec_runs.py` | Library: runs STRIDE from GPEC's `DIIID_ideal_example` inputs with `equil.in` overrides; reads Δ′ and `gsec.bin`. |
| `gpec_profiles.py` | Library: Python replica of how GPEC builds F, p and their slopes in `sq` for each profile source (figure 1). |
| `ifile_tools.py` | Library: i-file read/write, append FF′/p′ records, independent GS residual. |

The Fortran unit test for `spline_fit_hermite`, `spline_fit_pchip` and the `"extrap"` end slopes is in
the branch: `make -C regression/spline_tests run`.

## Reproduce
```bash
S=$PWD; R=results; BIN=<GPEC>/bin
python make_truth_equilibrium.py inputs/D3Dlike_Hmode_baseline.geqdsk inputs/DIIID_mesh.h5 $R/truth
for m in 128 256 512; do
  G147131=inputs/g147131.02300_DIIID_KEFIT python run_profile_source_comparison.py $BIN $R/truth $R/profile_source_m$m mpsi=$m
done
python run_ifile_comparison.py $BIN $R/truth $R/ifile
(cd $R && python $S/diag_ifile_noise.py truth truth/i*.ifile)          # writes $R/*_exact.ifile
for m in 128 256 512; do MPSI=$m python run_exact_ifiles.py $BIN $R/ifile_exact $R/*_exact.ifile; done
python make_figures.py $R figures
```
Environment: `NTHREADS` (TokaMaker threads), `NPROC` (parallel STRIDE runs), `METHODS` (profile sources,
default `values,integrate`; `hermite` needs a GPEC build before `a95a365b`, where it was dropped).

The "i-file as written" rows show the surface-tracing error only with an OpenFUSIONToolkit that predates
its `GPECf_interface` fix; with the fix, `save_ifile` already writes points on the exact crossings.
