# OFT_interface: i-file vs g-file scripts

Scripts behind the i-file vs g-file comparison in the `OFT_interface` PR. Python 3 with numpy, scipy, xarray
and matplotlib; OpenFUSIONToolkit's TokaMaker on `PYTHONPATH` for `make_truth_equilibrium.py` (an OFT whose
`save_ifile` writes the FF′ and p′ records). `BIN` is the `bin/` directory of a GPEC build of this branch.

## Inputs (`inputs/`)
- `D3Dlike_Hmode_baseline.geqdsk`: TokaMaker D3D-like H-mode; supplies the FF′/p′ shapes, Ip, F0 and boundary.
- `DIIID_mesh.h5`: the TokaMaker DIII-D mesh used for the solve.

## Scripts
| script | role |
|---|---|
| `make_truth_equilibrium.py` | Data. Solves the reference TokaMaker equilibrium and writes g-files at 129/257/513 and i-files at 65×129/129×257/257×513. |
| `run_ifile_vs_gfile.py` | Test. STRIDE Δ′, GPEC GSE and file size for every g-file (`efit`) and i-file (`ldp_i`), `profile_source = integrate`. |
| `make_figures.py` | Plots GSE vs file size and Δ′ vs mpsi. |
| `gpec_runs.py` | Library: runs STRIDE from GPEC's `DIIID_ideal_example` inputs with `equil.in` overrides; reads Δ′ and `gsec.bin`. |

## Reproduce
```bash
R=results; BIN=<GPEC>/bin
python make_truth_equilibrium.py inputs/D3Dlike_Hmode_baseline.geqdsk inputs/DIIID_mesh.h5 $R/truth
for m in 128 256 512; do python run_ifile_vs_gfile.py $BIN $R/truth $R/m$m mpsi=$m; done
python make_figures.py $R figures
```
Environment: `NTHREADS` (TokaMaker threads), `NPROC` (parallel STRIDE runs).
