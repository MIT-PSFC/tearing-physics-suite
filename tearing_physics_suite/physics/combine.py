import xarray as xr
import numpy as np
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
    n = arr.shape[0]
    flat = arr.reshape(n, -1)                       # (n, M) — M = prod(other)
    counts = np.empty_like(flat, dtype=int)
    for j in range(flat.shape[1]):
        _, inv, c = np.unique(flat[:, j], return_inverse=True,
                              return_counts=True)
        counts[:, j] = c[inv]
    return counts.reshape(arr.shape), other


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


def compile_xarrays(rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict, calc_dps=True, **kwargs):
    """
    Combine rdcon, stride, and pest3 xarrays into a single dataset.

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
    # Combine input dictionaries:
    if not (rdcon_stride_input_dict is None): #RDCON dict present
        if not (pest3_input_dict is None): # PEST3 dict present
            rdcon_stride_input_dict.update(pest3_input_dict)
        input_dict = rdcon_stride_input_dict
    elif not (pest3_input_dict is None):
        input_dict = pest3_input_dict
    else:
        input_dict = {}
    
    
    # Combine xarrays:
    xarrays = []
    
    #########################################################################################################
    # RDCON delta xarray and delta prime calculation
    #########################################################################################################
    if not (rdcon_xr is None): 
        # Turn all attributes into variables:
        for attr_key in rdcon_xr.attrs.keys():
            rdcon_xr[attr_key] = rdcon_xr.attrs[attr_key]
        rdcon_xr.attrs = {}
        rdcon_xr = rdcon_xr.drop_attrs(deep=True)
        # Add new dimension for code to rdcon_xr
        rdcon_xr_expanded = rdcon_xr.expand_dims(dim='code', axis=0)
        rdcon_xr_expanded['code'] = ['rdcon']
        if calc_dps and 'Delta_prime' in rdcon_xr_expanded:
            rdcon_xr_expanded = extract_delta_primes(rdcon_xr_expanded)
        # Add to xarrays list
        xarrays.append(rdcon_xr_expanded)
    
    #########################################################################################################
    # STRIDE delta xarray and delta prime calculation
    #########################################################################################################
    if not (stride_xr is None):
        # Turn all attributes into variables:
        for attr_key in stride_xr.attrs.keys():
            stride_xr[attr_key] = stride_xr.attrs[attr_key]
        stride_xr.attrs = {}
        stride_xr = stride_xr.drop_attrs(deep=True)
        # Add new dimension for code to stride_xr
        stride_xr_expanded = stride_xr.expand_dims(dim='code', axis=0)
        stride_xr_expanded['code'] = ['stride']
        if calc_dps and 'Delta_prime' in stride_xr_expanded:
            # Calculate delta' values for stride_xr
            stride_xr_expanded = extract_delta_primes(stride_xr_expanded)
        xarrays.append(stride_xr_expanded)
    
    #########################################################################################################
    # PEST3 delta xarray and delta prime calculation
    #########################################################################################################
    pest3_xr_expanded = None
    if not (pest3_xr is None):
        # Turn all attributes into variables:
        for attr_key in pest3_xr.attrs.keys():
            pest3_xr[attr_key] = pest3_xr.attrs[attr_key]
        pest3_xr.attrs = {}
        pest3_xr = pest3_xr.drop_attrs(deep=True)
        # Add new dimension for code to pest3_xr
        pest3_xr_expanded = pest3_xr.expand_dims(dim='code', axis=0)
        pest3_xr_expanded['code'] = ['pest3']
        if calc_dps and 'Delta_prime' in pest3_xr_expanded:
            assert 'Delta_prime_perr' in pest3_xr_expanded, "Current version of extract_delta_primes assumes this."
            pest3_xr_expanded = extract_delta_primes(pest3_xr_expanded)
        xarrays.append(pest3_xr_expanded)

    # Combine all xarrays into one xarray:
    # Breaks if different number of rational surfaces across different codes at the axis
    #   - beware psilow =/= 0 while also running pest3 (pest3 has no psilow truncation)
    #   - for this reason, we also output pest3_xr_out separately if something goes wrong
    pest3_xr_out = None
    combined_xr = None

    #########################################################################################################
    # Concatenating xarrays
    #########################################################################################################
    if len(xarrays) > 0:
        pest3_xr_out = pest3_xr_expanded
        try:
            combined_xr = xr.concat(xarrays, dim='code', coords='all', **kwargs)
            pest3_xr_out = None
        except Exception as e:
            if not (pest3_xr is None): #We remove pest3_xr_expanded from xarrays and retry
                xarrays = xarrays[:-1]  # Remove the last element (pest3_xr_expanded)
                combined_xr = xr.concat(xarrays, dim='code', coords='all', **kwargs)
            print("Error combining xarrays:", e)

    #########################################################################################################
    # Adding nn to combined_xr:
    #########################################################################################################
    if combined_xr is not None:
        # Check nn isn't already defined:
        assert not 'nn' in combined_xr, 'nn already defined, debug this function.'
        # We expand dims to add nn:
        combined_xr = combined_xr.expand_dims(dim='nn', axis=0)
        combined_xr['nn'] = [input_dict['nn']]

    return combined_xr, pest3_xr_out, input_dict
