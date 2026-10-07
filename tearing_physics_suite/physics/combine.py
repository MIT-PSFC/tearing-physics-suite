import numpy as np
import xarray as xr

from tearing_physics_suite.physics.delta_prime_extraction import extract_delta_primes


def _uniquify_r(ds):
    """
    Make degenerate dims 'r'/'r_prime' concatenable by replacing each with a
    UNIQUE INTEGER-VALUED FLOAT index, while preserving the real values and the
    (from-the-right) occurrence pattern as plain coordinates.

    Using float indices (0.0, 1.0, ...) instead of ints leaves room to insert
    NaN sentinels into the 'r'/'r_prime' coordinates in later steps.
    """
    def _occ_from_right(vals):
        seen = {}
        occ = np.empty(len(vals), dtype=int)
        for i in range(len(vals) - 1, -1, -1):
            v = vals[i]
            occ[i] = seen.get(v, 0)
            seen[v] = occ[i] + 1
        return occ

    def _unique_flag(vals):
        # True where this value occurs exactly once in `vals`; False if it is
        # part of a degenerate group (i.e. will be collapsed later).
        _, inverse, counts = np.unique(
            vals, return_inverse=True, return_counts=True)
        return counts[inverse] == 1

    has_r  = "r" in ds.dims
    has_rp = "r_prime" in ds.dims

    # r_prime is a copy of r -> compute the occurrence pattern once.
    ref_vals = ds["r"].values if has_r else ds["r_prime"].values
    occ = _occ_from_right(ref_vals)
    unique = _unique_flag(ref_vals)

    if has_r and has_rp:
        if not np.array_equal(ds["r"].values, ds["r_prime"].values):
            raise ValueError("r and r_prime differ in values; cannot share the occurrence pattern.")

    def _apply(ds, dim, vals, occ, unique):
        ds = ds.assign({
            f"{dim}_value":  (dim, vals),    # real (degenerate) values
            f"{dim}_occ":    (dim, occ),     # occurrence-from-right
            f"{dim}_unique": (dim, unique),  # True if value occurs exactly once
        })
        # Replace the dim's index with unique integer-VALUED FLOATS 0.0..N-1.0
        ds = ds.assign_coords({dim: np.arange(len(vals), dtype=float)})
        return ds

    if has_r:
        ds = _apply(ds, "r", ref_vals, occ, unique)
    if has_rp:
        ds = _apply(ds, "r_prime", ds["r_prime"].values, occ,
                    _unique_flag(ds["r_prime"].values))
    return ds


def add_unique_label(combined_xr, dims=("r", "r_prime")):
    """Add {dim}_unique to datasets made before it existed ({dim}_value, {dim}_occ must be present)."""

    out = combined_xr
    for dim in dims:
        occ_name    = f"{dim}_occ"
        value_name  = f"{dim}_value"
        unique_name = f"{dim}_unique"
        if occ_name not in out.data_vars:
            raise ValueError(f"Cannot collapse, '{occ_name}' not found in dataset data_vars")
        if value_name not in out.data_vars:
            raise ValueError(f"Cannot collapse, '{value_name}' not found in dataset data_vars")

        # Per-slice occurrence counts along `dim` -> per-slice uniqueness flag.
        # counts comes back in (dim, *val_other) order.
        counts, val_other = _counts_along_dim(out[value_name], dim)
        unique = counts == 1                        # shape (dim, *val_other)

        # {dim}_unique lives on the SAME domain as {dim}_value, and should
        # share its dimension ORDER too.
        out = out.assign({unique_name: ((dim, *val_other), unique)})
        out[unique_name] = out[unique_name].transpose(*out[value_name].dims)
    return out


def _counts_along_dim(da, dim):
    """
    Occurrence count of each value along `dim`, computed INDEPENDENTLY for
    every combination of the other dimensions `da` lives on.
    Returns (counts_ndarray shaped (dim, *other), other_dim_names).
    np.unique groups exactly (no tolerance), matching the degenerate-surface
    model where duplicates are the identical value repeated.
    """
    other = [d for d in da.dims if d != dim]
    da = da.transpose(dim, *other)
    arr = da.values
    a, b = arr[:, None], arr[None, :]               # compare every pair along `dim`
    same = a == b
    if arr.dtype.kind in 'fc':
        same |= np.isnan(a) & np.isnan(b)           # np.unique groups NaNs together
    return same.sum(axis=1), other


# Rational-surface access after _uniquify_r ('r'/'r_prime' are positional indices):
#   positional pick:         combined_xr.isel(r=0)
#   real values:             combined_xr.r_value (may vary along nn / run_idx)
#   select by real value:    sel_rational(combined_xr, 2.0)
#   real-valued r restored:  collapse_to_primary(combined_xr).sel(r=2.0)

def collapse_to_primary(ds, dims=("r", "r_prime")):
    """Restore real-valued 'r'/'r_prime' coordinates, undoing _uniquify_r.

    Keeps the occ==0 (outermost) occurrence of each degenerate value and drops
    NaN padding. If {dim}_value varies along other dims (e.g. nn, run_idx), each
    slice is collapsed separately and re-concatenated with an outer join on the
    real values. {dim}_unique is kept to flag surfaces that were degenerate.
    """
    present = [d for d in dims if d in ds.dims and f"{d}_value" in ds and f"{d}_occ" in ds]
    if not present:
        return ds

    other = [o for d in present for o in ds[f"{d}_value"].dims
             if o not in dims and ds.sizes[o] > 1]
    if other:
        o = other[0]
        parts = [collapse_to_primary(ds.isel({o: [i]}), dims) for i in range(ds.sizes[o])]
        return xr.concat(parts, dim=o, data_vars="minimal", coords="minimal",
                         join="outer", compat="override")

    for d in present:
        # Remaining non-d dims have size 1, so these flatten to 1D along d.
        vals = ds[f"{d}_value"].transpose(..., d).values.astype(float).reshape(-1)
        occ = ds[f"{d}_occ"].transpose(..., d).values.astype(float).reshape(-1)
        keep = np.where((occ == 0) & np.isfinite(vals))[0]
        ds = ds.isel({d: keep})
        ds = ds.drop_vars([f"{d}_value", f"{d}_occ"]).assign_coords({d: vals[keep]})
    return ds


def sel_rational(ds, value, dim="r", atol=1e-8):
    """Select rational surface(s) by real value of 'r' (or 'r_prime'); see collapse_to_primary."""
    collapsed = collapse_to_primary(ds)
    idx = np.where(np.isclose(collapsed[dim].values, value, atol=atol))[0]
    return collapsed.isel({dim: idx})


# Order of codes on the 'code' dim
CODE_ORDER = ('rdcon', 'stride', 'jGPEC_galerkin', 'jGPEC_riccati', 'pest3')
JGPEC_CODES = ('jGPEC_galerkin', 'jGPEC_riccati')


def ordered_codes(datasets):
    """{code: ds} in CODE_ORDER (unknown codes last), dropping None."""
    rank = {c: i for i, c in enumerate(CODE_ORDER)}
    return {c: datasets[c] for c in sorted(datasets, key=lambda c: rank.get(c, len(rank))) if datasets[c] is not None}


def align_surfaces(ds, ref, atol=1e-6):
    """ds on ref's rational surfaces (r, r_prime); NaN where ds has no surface.

    Surfaces are matched by q_rational (to atol), the nearest psi_n_rational among equal q (reversed shear).
    Surfaces of ds that ref lacks are dropped with a warning. Used for jGPEC, whose surface set is its own
    (Riccati: the surfaces it crossed). Apply after the Delta' coupling, which needs ds's own surfaces.
    """
    def flat(d, v):
        return d[v].values.reshape(-1, d.sizes['r'])[0]
    q, psi, ref_q, ref_psi = flat(ds, 'q_rational'), flat(ds, 'psi_n_rational'), flat(ref, 'q_rational'), flat(ref, 'psi_n_rational')
    idx, free = np.full(len(ref_q), -1), np.ones(len(q), bool)
    for k in range(len(ref_q)):
        cand = np.flatnonzero(free & np.isclose(q, ref_q[k], rtol=0, atol=atol))
        if len(cand):
            idx[k] = cand[np.argmin(np.abs(psi[cand] - ref_psi[k]))]
            free[idx[k]] = False
    if free.any():
        print(f"WARNING: surfaces not in the reference dropped: q = {q[free]}, psi_n = {psi[free]}")
    pos, found = np.maximum(idx, 0), idx >= 0
    out = ds.drop_vars(['r', 'r_prime']).isel(r=pos, r_prime=pos)
    for v in out.data_vars:
        for d in ('r', 'r_prime'):
            if d in out[v].dims:
                out[v] = out[v].where(xr.DataArray(found, dims=d))
    return out.assign_coords(r=ref['r'].values, r_prime=ref['r_prime'].values)


def merge_input_dicts(*dicts):
    """Merge input dicts left to right, skipping None (later dicts win). Returns a new dict."""
    out = {}
    for d in dicts:
        if d is not None:
            out.update(d)
    return out


def add_code_dim(ds, code, delete_attrs=True, drop_var_attrs=False):
    """Copy of ds with its attributes copied into variables and a leading 'code' dim of length 1."""
    ds = ds.copy()
    for key, val in ds.attrs.items():
        ds[key] = val
    if delete_attrs:
        ds.attrs = {}
    if drop_var_attrs:
        ds = ds.drop_attrs(deep=True)
    ds = ds.expand_dims(dim='code', axis=0)
    ds['code'] = [code]
    return ds


def code_delta_primes(ds, code):
    """Add coupled Delta' values (extract_delta_primes) if ds has Delta_prime."""
    if 'Delta_prime' not in ds:
        return ds
    if code == 'pest3':
        assert 'Delta_prime_perr' in ds, "Current version of extract_delta_primes assumes this."
    return extract_delta_primes(ds)


def combine_codes(datasets, nn, debug=False, **concat_kwargs):
    """Concatenate per-code datasets along 'code' (in dict order), then add a leading 'nn' dim.

    datasets: {code: Dataset with a 'code' dim, or None}.
    If the concat fails, codes are added one at a time and any code that will not concatenate is
    left out, by name, with a warning. Breaks happen when codes have different rational surfaces at
    the axis (e.g. psilow != 0 while also running PEST3, which has no psilow truncation).
    Returns (combined_xr or None, dropped), dropped = {code: dataset left out}.
    """
    items = [(code, ds) for code, ds in datasets.items() if ds is not None]
    combined, dropped = None, {}
    if items:
        try:
            combined = xr.concat([ds for _, ds in items], dim='code', coords='all', **concat_kwargs)
        except Exception as e:
            print("Error combining xarrays:", e)
            if debug:
                raise
            kept = []
            for code, ds in items:
                try:
                    combined = xr.concat(kept + [ds], dim='code', coords='all', **concat_kwargs)
                    kept.append(ds)
                except Exception as e_code:
                    print(f"Leaving code '{code}' out of the combined dataset: {e_code}")
                    dropped[code] = ds
            if not kept:
                combined = None
    if combined is not None:
        assert 'nn' not in combined, 'nn already defined, debug this function.'
        if nn is None:
            raise KeyError('nn')
        combined = combined.expand_dims(dim='nn', axis=0)
        combined['nn'] = [nn]
    return combined, dropped


def compile_xarrays(rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict,
                    jgpec_xrs=None, jgpec_ran=None, calc_dps=True, **kwargs):
    """
    Combine rdcon, stride, jGPEC and pest3 xarrays into a single dataset.

    Merges outputs from run_resistive_calculation into one xarray with a 'code'
    dimension. Optionally computes coupled Delta' values.

    Parameters
    ----------
    rdcon_xr, stride_xr, pest3_xr : xr.Dataset or None
        Per-code output datasets.
    rdcon_ran, stride_ran, pest3_ran : bool
        Whether each code ran successfully.
    rdcon_stride_input_dict, pest3_input_dict : dict or None
        Input parameter dictionaries.
    jgpec_xrs, jgpec_ran : dict or None
        {'jGPEC_<solver>': xr.Dataset or None} and run flags (see run_resistive_calculation).
    calc_dps : bool
        If True, compute coupled Delta' values via extract_delta_primes.

    Returns
    -------
    combined_xr : xr.Dataset or None
        Merged dataset with 'code' dimension.
    pest3_xr : xr.Dataset or None
        PEST3 dataset (returned separately if it couldn't be merged).
    input_dict : dict
        Combined input parameters.
    """
    input_dict = merge_input_dicts(rdcon_stride_input_dict, pest3_input_dict)
    expanded = {}
    datasets = ordered_codes({'rdcon': rdcon_xr, 'stride': stride_xr, 'pest3': pest3_xr, **(jgpec_xrs or {})})
    for code, ds in datasets.items():
        ds = add_code_dim(ds, code, drop_var_attrs=True)
        ds = code_delta_primes(ds, code) if calc_dps else ds
        if code in JGPEC_CODES and 'rdcon' in expanded:
            ds = align_surfaces(ds, expanded['rdcon'])
        expanded[code] = ds
    # PEST3 is also returned on its own if it could not be combined (see combine_codes).
    combined_xr, dropped = combine_codes(expanded, input_dict.get('nn'), **kwargs)
    return combined_xr, dropped.get('pest3'), input_dict
