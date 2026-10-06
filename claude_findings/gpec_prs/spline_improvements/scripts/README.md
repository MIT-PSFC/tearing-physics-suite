# spline_improvements: test, data-generation and plotting scripts

Scripts behind the evidence in the `spline_improvements` PR. Python 3 with numpy, scipy, xarray and
matplotlib; OpenFUSIONToolkit's TokaMaker on `PYTHONPATH` for `make_truth_equilibrium.py`
and `make_figures.py`. `BIN` is the `bin/` directory of a GPEC build of this branch.

## Inputs (`inputs/`)
- `D3Dlike_Hmode_baseline.geqdsk`: TokaMaker D3D-like H-mode; supplies the FF′/p′ shapes, Ip, F0 and boundary.
- `DIIID_mesh.h5`: the TokaMaker DIII-D mesh used for the solve.
- `g147131.02300_DIIID_KEFIT`: the classic EFIT g-file formerly in GPEC's DIIID examples.

## Scripts
| script | role |
|---|---|
| `make_truth_equilibrium.py` | Data. Solves the reference TokaMaker equilibrium (C2 profiles on 4001 nodes, so F, F′, F″, p′ are known at any ψ) and writes the truth profiles, g-files at 129/257/513 and i-files (with FF′, p′ records) at 65×129/129×257/257×513. |
| `run_profile_source_comparison.py` | Test. STRIDE Δ′ and GPEC GSE for `profile_source = values` and `integrate` on the g-files (`efit`), the i-files (`ldp_i`), GPEC's TkMkr example and g147131. |
| `make_figures.py` | Plots the two report figures. |
| `gpec_runs.py` | Library: runs STRIDE from GPEC's `DIIID_ideal_example` inputs with `equil.in` overrides; reads Δ′ and `gsec.bin`. |
| `gpec_profiles.py` | Library: Python replica of how GPEC builds F, p and their slopes in `sq` for each profile source (figure 1). |

The Fortran unit test for `spline_fit_hermite`, `spline_fit_pchip` and the `"extrap"` end slopes is in
the branch: `make -C regression/spline_tests run`.

## Reproduce
```bash
R=results; BIN=<GPEC>/bin
python make_truth_equilibrium.py inputs/D3Dlike_Hmode_baseline.geqdsk inputs/DIIID_mesh.h5 $R/truth
for m in 128 256 512; do
  G147131=inputs/g147131.02300_DIIID_KEFIT python run_profile_source_comparison.py $BIN $R/truth $R/profile_source_m$m mpsi=$m
done
python make_figures.py $R figures
```
Environment: `NTHREADS` (TokaMaker threads), `NPROC` (parallel STRIDE runs), `METHODS` (profile sources,
default `values,integrate`; `hermite` needs a GPEC build before `a95a365b`, where it was dropped).
The i-files need an OpenFUSIONToolkit whose `save_ifile` writes the FF′ and p′ records.
