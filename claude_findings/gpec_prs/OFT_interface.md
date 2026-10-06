# GPEC PR report: `OFT_interface` → `develop`

This PR builds on `spline_improvements`.

- **Branch:** `OFT_interface`, 1 commit on top of `spline_improvements` `a95a365b`.

## Purpose
Read OpenFUSIONToolkit/TokaMaker i-files (`save_ifile`, OFT branch `GPECf_interface`) with `eq_type = "ldp_i"`. The pieces that make this path accurate are spread over three PRs:
- **OFT `GPECf_interface`:** points lie exactly on their flux surfaces, and FF′ and p′ records are written. See [GPECf_interface.md](GPECf_interface.md).
- **`spline_improvements`:** reads the optional FF′ and p′ records with `profile_source = integrate`.
- **This PR:** the remaining reader fix.

## Changes
1. **`read_eq_ldp_i` takes |F|** (`19a8099b`), as `read_eq_efit` does.
   - `inverse_run` computes q in proportion to F. An i-file with F < 0 (TokaMaker with B_t < 0) therefore stopped with "Invalid extrapolation near axis".
   - With the fix, flipping the sign of F in a file gives the same Δ′ to every digit: 8.50824 for both.
2. **The i-file format is documented** in the reader header.

## Results
RDCON and STRIDE on the same TokaMaker equilibrium written both ways, nn = 1, mpsi = 257.

| file | RDCON Δ′(2/1) | STRIDE Δ′(2/1) |
|---|---|---|
| g257.geqdsk (efit, integrate) | 8.400 | 8.415 |
| i129x257.ifile (ldp_i) | 8.246 | 8.261 |

The direct STRIDE runs in [spline_improvements.md](spline_improvements.md) and [GPECf_interface.md](GPECf_interface.md) show the i-file result staying fixed as mpsi changes (8.51 to 8.53 from mpsi 128 to 512), while the g-file result moves by about 10% at mpsi 256.

## Status
- [x] Built; the negative-F test above passes
- [x] CI Solovev examples. Compared with `Zeff_profile_support` `99378252`, which also has the vacuum fix, on the full chain (OFT_interface = spline_improvements + this commit):
  - The DCON, STRIDE, RDCON and GPEC outputs of the ideal and resistive Solovev examples are **bit-for-bit identical**.
  - The kinetic example differs by up to 8e-5, and PENTRC by up to 1e-6 in n and T, because PENTRC's kinetic table now uses pchip. The exceptions are the last 2 of 17 PENTRC output points next to the edge (ψ_N > 0.99), where T → 0: there `logLambda`/`nu` were already unphysical (negative) with the cubic spline and change value.
  - The example pass/fail pattern is unchanged; the gpec/pentrc/rmatch "failures" in the DIIID examples happen the same way on the base build.
