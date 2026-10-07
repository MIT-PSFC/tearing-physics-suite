import numpy as np
import xarray as xr


def global_mre_quantities(combined_xr,psi_pedestal_cutoff=0.9):
    """Rank nonlinear stability of all modes across n and rational surfaces.

    Computes least-stable-mode metrics: largest nondimensional island growth rate
    (max_dwdtau) and smallest seed island needed to initiate NTM onset (min_w_marg). Rankings
    are computed per (Delta_prime_type, code, k1, C0) combination within psi_pedestal_cutoff.

    Parameters
    ----------
    combined_xr : xr.Dataset
        Multi-n combined dataset with w_marg_surf and dwdtau_max_surf.
    psi_pedestal_cutoff : float
        Exclude modes with psi_n_rational above this value from ranking.

    Returns
    -------
    xr.Dataset
        Input dataset with min_w_marg_allsurf, max_dwdtau_allsurf,
        min_w_marg_rank, and max_dwdtau_rank variables added.
    """
    # Min w_marg_surf over all m, n
    # Max dwdtau over all m, n
    min_w_marg = combined_xr.w_marg_surf.min(dim=['r', 'nn'])
    max_dwdtau = combined_xr.dwdtau_max_surf.max(dim=['r', 'nn'])
    # Add these to combined_xr:
    combined_xr = combined_xr.assign(
        min_w_marg_allsurf=min_w_marg,
        max_dwdtau_allsurf=max_dwdtau
    )
    # For each Delta_prime_type, code (and k1, C0), we want to rank the modes by min_w_marg and max_dwdtau over all r, nn:
    assert combined_xr.w_marg_surf.dims == combined_xr.dwdtau_max_surf.dims, "Dimensions of w_marg_surf and dwdtau_max_surf do not match"

    #########################################################################################################
    # Applying psi_n_rational mask
    #########################################################################################################
    psi_n_rational_like_surfaces = combined_xr.psi_n_rational+0.0*combined_xr['Delta_prime_surf']
    # Set psi_n_rational_like_surfaces.loc[code='pest3'] equal to psi_n_rational_like_surfaces.loc[code='rdcon'] (since pest3 doesn't compute psi_n_rational)
    if 'rdcon' in combined_xr.code.values and 'pest3' in combined_xr.code.values:
        psi_n_rational_like_surfaces.loc[dict(code='pest3')] = psi_n_rational_like_surfaces.loc[dict(code='rdcon')]
    inside = psi_n_rational_like_surfaces < psi_pedestal_cutoff

    #########################################################################################################
    # Rank w_marg_surf (smallest to largest) and dwdtau_max_surf (largest to smallest) over (nn, r).
    # Combinations with no valid w_marg_surf or dwdtau_max_surf are left NaN.
    #########################################################################################################
    def _ranks(vals, descending):
        """1-based ranks of vals over all its elements (NaN stays NaN)."""
        ranks = np.argsort(np.argsort(-vals if descending else vals, axis=None)) + 1 # +1 to make ranks start from 1
        ranks = ranks.reshape(vals.shape).astype(float) # Convert to float to allow for NaNs
        ranks[np.isnan(vals)] = np.nan
        return ranks

    def _rank_da(da, descending):
        r = xr.apply_ufunc(_ranks, da.where(inside), kwargs={'descending': descending},
                           input_core_dims=[['nn', 'r']], output_core_dims=[['nn', 'r']], vectorize=True)
        return da.copy(data=r.transpose(*da.dims).values)

    has_data = (combined_xr.w_marg_surf.count(['nn', 'r']) > 0) & (combined_xr.dwdtau_max_surf.count(['nn', 'r']) > 0)
    w_marg_rank = _rank_da(combined_xr.w_marg_surf, descending=False).where(has_data)
    dwdtau_rank = _rank_da(combined_xr.dwdtau_max_surf, descending=True).where(has_data)

    combined_xr = combined_xr.assign(
        min_w_marg_rank=w_marg_rank,
        max_dwdtau_rank=dwdtau_rank
    )

    return combined_xr


def delta_prime_variability(xarray,comparison_var='code',
        run_bool_check=False,
        abs_threshold=0.05,
        rel_threshold=0.05
        #abs_PEST_threshold=0.3,
        #rel_PEST_threshold=0.1
        ):
    """Compare Delta_prime_surf across a dimension (e.g. 'code') for all modes.

    Records absolute and relative differences. Handles special comparisons:
    STRIDE vs RDCON ('GPEC') and GPEC vs PEST3 ('GPECvsPEST').

    Boolean True False values are generated to check whether the differences in Delta_prime_surf lie within the bounds
    set by abs_threshold, rel_threshold, abs_PEST_threshold, and rel_PEST_threshold.

    Parameters
    ----------
    xarray : xr.Dataset or xr.xarray.DataArray
        If xr.Dataset, must contain Delta_prime_surf. Must have comparison_var as a dimension.
    comparison_var : str
        Dimension along which to compare (default 'code').
    run_bool_check : bool
        If True, also checks whether differences exceed thresholds.
    abs_threshold, rel_threshold : float
        Thresholds for the boolean checks.

    Returns
    -------
    xr.Dataset
        Dataset with Delta_prime_diff_across_* and Delta_prime_reldiff_across_* added.
        If run_bool_check, also returns four boolean scalars.
    """

    # Convert xarray into Dataset if it isn't already one:
    if isinstance(xarray, xr.DataArray):
        xarray = xarray.to_dataset(name='Delta_prime_surf')

    #########################################################################################################
    # check xarray has comparison_var in it
    #########################################################################################################

    assert comparison_var in xarray.dims, f"'{comparison_var}' not found in xarray dimensions: {list(xarray.dims.keys())}"

    if len(xarray[comparison_var]) == 0:
        print(f"Only one value of {comparison_var}' found, cross-variable comparisons not carried out.")
        return xarray

    #########################################################################################################
    # Record the maximum range of variation in Delta_prime_surf across comparison_var
    #########################################################################################################

    xarray = add_comparison_across_var(xarray,xarray.Delta_prime_surf,comparison_var)
    if run_bool_check: # This is used specifically for input scans.
        xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere = add_bool_checks(xarray, comparison_var, abs_threshold, rel_threshold)

    #########################################################################################################
    # Special case: comparison_var = 'code', just compare stride and rdcon
    #   Will overlap with code case if only STRIDE and RDCON were ran
    #########################################################################################################

    if comparison_var == 'code' and 'stride' in xarray.Delta_prime_surf.code and 'rdcon' in xarray.Delta_prime_surf.code:
        Delta_prime_surf_stride_and_rdcon = xarray.Delta_prime_surf.sel(code=xarray.Delta_prime_surf.code.isin(['stride', 'rdcon']))
        xarray = add_comparison_across_var(xarray,Delta_prime_surf_stride_and_rdcon,comparison_var,override_name='GPEC')

    #########################################################################################################
    # Special case: comparison_var = 'code', just compare stride, rdcon, and pest3 (no cylindrical)
    #   Will overlap with code case if only STRIDE and RDCON were ran
    #########################################################################################################

    if comparison_var == 'code' and 'pest3' in xarray.Delta_prime_surf.code and ('rdcon' in xarray.Delta_prime_surf.code or 'stride' in xarray.Delta_prime_surf.code):
        Delta_prime_surf_pest3_stride_andor_rdcon = xarray.Delta_prime_surf.sel(code=xarray.Delta_prime_surf.code.isin(['stride', 'rdcon','pest3']))
        xarray = add_comparison_across_var(xarray,Delta_prime_surf_pest3_stride_andor_rdcon,comparison_var,override_name='GPECvsPEST')

    #########################################################################################################
    # Special case: comparison_var = 'code', each jGPEC solver vs rdcon ('<code>vsRDCON'), and the two
    #   jGPEC solvers against each other ('jGPEC_solvers')
    #########################################################################################################

    if comparison_var == 'code':
        codes = list(xarray.Delta_prime_surf.code.values)
        jgpec = [c for c in codes if str(c).startswith('jGPEC_')]
        for c in jgpec if 'rdcon' in codes else []:
            xarray = add_comparison_across_var(xarray, xarray.Delta_prime_surf.sel(code=['rdcon', c]), comparison_var,
                                               override_name=f'{c}vsRDCON')
        if len(jgpec) > 1:
            xarray = add_comparison_across_var(xarray, xarray.Delta_prime_surf.sel(code=jgpec), comparison_var,
                                               override_name='jGPEC_solvers')

    if run_bool_check:
        return xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere
    return xarray


def add_comparison_across_var(xarray, Delta_prime_surf, comparison_var,override_name=''):
    """Compute absolute and relative Delta' differences across comparison_var.

    Adds Delta_prime_diff_across_{name}, Delta_prime_reldiff_across_{name} to xarray.

    Parameters
    ----------
    xarray : xr.Dataset
        Target dataset.
    Delta_prime_surf : xr.DataArray
        Delta' values to compare. Must have comparison_var as a dimension.
    comparison_var : str
        Dimension along which to compute max - min.
    override_name : str
        If non-empty, used in output variable names instead of comparison_var.

    Returns
    -------
    xr.Dataset
        Input dataset with difference variables added.
    """

    Delta_prime_diffs_across_var = np.abs(Delta_prime_surf.max(dim=comparison_var)-Delta_prime_surf.min(dim=comparison_var))
    Delta_prime_reldiffs_across_var = Delta_prime_diffs_across_var / np.abs(Delta_prime_surf).mean(dim=comparison_var)
    Delta_prime_reldiffs_across_var2 = Delta_prime_diffs_across_var / Delta_prime_surf.mean(dim=comparison_var)

    if len(override_name) == 0:
        override_name = comparison_var

    # Add Delta_prime_diffs_across_var to xarray, with str(comparison_var) included in name:
    xarray = xarray.assign({
        f'Delta_prime_diff_across_{override_name}': Delta_prime_diffs_across_var,
        f'Delta_prime_reldiff_across_{override_name}': Delta_prime_reldiffs_across_var,
        f'Delta_prime_reldiff_across_{override_name}_2': Delta_prime_reldiffs_across_var2
    })

    return xarray


def add_bool_checks(xarray, comparison_var, abs_threshold, rel_threshold, Delta_prime_type='single helicity',drop_pest=False):
    """Check whether Delta' differences across comparison_var exceed thresholds.

    Tests both all modes and modes within psi_n < 0.95.

    Returns
    -------
    xr.Dataset
        Updated dataset with *_thresh_exceeded variables.
    abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere : xr.DataArray
        Whether absolute / relative thresholds are exceeded for any (r, n).
    abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere : xr.DataArray
        Same, restricted to modes within psi_n < 0.95.
    """

    absdiffs_name = f'Delta_prime_diff_across_{comparison_var}'
    reldiffs_name = f'Delta_prime_reldiff_across_{comparison_var}'

    xarray = xarray.assign({
        f'{absdiffs_name}_thresh_exceeded': xarray[absdiffs_name] > abs_threshold,
        f'{reldiffs_name}_thresh_exceeded': xarray[reldiffs_name] > rel_threshold
    })

    abs_thresh_exceeded_da = xarray[f'{absdiffs_name}_thresh_exceeded']
    rel_thresh_exceeded_da = xarray[f'{reldiffs_name}_thresh_exceeded']

    #########################################################################################################
    # Applying psi_n_rational mask to check if thresh exceeded within the q95 window
    #########################################################################################################
    psi_n_rational_copy = xarray.psi_n_rational.mean(dim=comparison_var)
    psi_n_rational_like_absdiffs = psi_n_rational_copy.broadcast_like(xarray[f'{absdiffs_name}_thresh_exceeded'])

    abs_thresh_exceeded_within_q95 = xarray[f'{absdiffs_name}_thresh_exceeded'].where(psi_n_rational_like_absdiffs < 0.95, drop=True)
    rel_thresh_exceeded_within_q95 = xarray[f'{reldiffs_name}_thresh_exceeded'].where(psi_n_rational_like_absdiffs < 0.95, drop=True)

    #########################################################################################################
    # Optional: drop pest3 from the boolean checks, since it's expected to differ more and is less relevant for input scans.
    #########################################################################################################

    if drop_pest and 'pest3' in xarray.code.values:
        abs_thresh_exceeded_da = abs_thresh_exceeded_da.where(xarray.code != 'pest3', drop=True)
        rel_thresh_exceeded_da = rel_thresh_exceeded_da.where(xarray.code != 'pest3', drop=True)
        abs_thresh_exceeded_within_q95 = abs_thresh_exceeded_within_q95.where(xarray.code != 'pest3', drop=True)
        rel_thresh_exceeded_within_q95 = rel_thresh_exceeded_within_q95.where(xarray.code != 'pest3', drop=True)

    #########################################################################################################
    # Cycling over all m,n
    #########################################################################################################

    # Check if absdiffs_name has dims "nn" in them:
    if "nn" in xarray[absdiffs_name].dims:
        reduce_dims = ["r", "nn"]
    else:
        reduce_dims = ["r"]

    abs_thresh_exceeded_anywhere_da = abs_thresh_exceeded_da.any(dim=reduce_dims)
    rel_thresh_exceeded_anywhere_da = rel_thresh_exceeded_da.any(dim=reduce_dims)

    abs_thresh_exceeded_psi95_anywhere_da = abs_thresh_exceeded_within_q95.any(dim=reduce_dims)
    rel_thresh_exceeded_psi95_anywhere_da = rel_thresh_exceeded_within_q95.any(dim=reduce_dims)

    #########################################################################################################
    # Choosing a specific type of Delta prime for the comparison:
    #########################################################################################################

    abs_thresh_exceeded_anywhere = abs_thresh_exceeded_anywhere_da.sel(Delta_prime_type=Delta_prime_type)
    rel_thresh_exceeded_anywhere = rel_thresh_exceeded_anywhere_da.sel(Delta_prime_type=Delta_prime_type)
    abs_thresh_exceeded_psi95_anywhere = abs_thresh_exceeded_psi95_anywhere_da.sel(Delta_prime_type=Delta_prime_type)
    rel_thresh_exceeded_psi95_anywhere = rel_thresh_exceeded_psi95_anywhere_da.sel(Delta_prime_type=Delta_prime_type)

    return xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere
