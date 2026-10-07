# Python functions to construct and analyse the modified Rutherford equation on modes

import numpy as np
import xarray as xr
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import find_peaks


# If I want: make extra dimension for different versions of generate_wd_function [should probably do this, right now not sure...]
def extract_critical_mre_factors_on_modes(
        code_xarray, # Xarray providing the Delta primes, which will be updated with the MRE analysis
        rdcon_xarray, # Xarray providing surface MRE information, will not be updated unless code_xarray = rdcon_xarray
        k0=0.8227,
        k1=1.7,
        C0=0.6,
        **kwargs):
    """
    Takes in code_xarray with delta prime values already computed, and rdcon_xarray with key mre surface terms computed.
    Returns code_xarray with critical MRE factors including maximum island width,
    location of maximum island width, and minimum marginally stable island width computed using the delta prime values.
    Parameters:
    k0 = 0.8227 comes from private communication w. Eric Howell, but is near identical to LaHaye 2017 10.1051/epjconf/201715703027 Eq. 1.
    k1 = 1.7 comes from Chang et al. PRL 1995
    C0 = 0.6 comes from Schlutt and Hegna PoP 2012
    Returns:
    code_xarray : xarray.DataSet
    The updated xarray with critical MRE terms calculated.
    """
    # Check if Delta_prime_varname is in code_xarray:
    assert 'Delta_prime_surf' in code_xarray.data_vars, "Can't run MRE analysis on code_xarray if Delta_prime_surf isn't present..."
    DP_da = code_xarray.Delta_prime_surf

    # Next steps: we check that the r coordinates are the same:
    r1 = code_xarray.Delta_prime_surf.r
    r2 = rdcon_xarray.psi_n_rational.r
    assert np.allclose(r1, r2), "r coordinate does not match between code_xarray and rdcon_xarray"

    # This code should create a vector of w values, then construct the MRE for each w, pulling the minimum for diffusion etc...
    w_vec = np.logspace(-8,0,num=1000)
    w_vec_lowres = np.logspace(-5,0,num=200)

    # Defining output structures:
    tempda = xr.full_like(code_xarray.Delta_prime_surf, np.nan)
    tempda2 = xr.full_like(rdcon_xarray.psi_n_rational, np.nan)
    tempda3 = xr.DataArray(
        np.full(code_xarray.Delta_prime_surf.shape + (len(w_vec_lowres),), np.nan),
        dims=code_xarray.Delta_prime_surf.dims + ('w_bar',),
        coords={**code_xarray.Delta_prime_surf.coords, 'w_bar': w_vec_lowres}
    )

    w_margs = tempda.copy(deep=True)
    w_sats = tempda.copy(deep=True)
    w_max_locs = tempda.copy(deep=True)
    dwdtau_maxs = tempda.copy(deep=True)
    wd_at_margs = tempda.copy(deep=True)

    prefacs = tempda2.copy(deep=True)
    wd_at_X0s = tempda2.copy(deep=True)
    dwdt_lowres_da = tempda3.copy(deep=True)

    # Start surface by surface
    for ri in range(rdcon_xarray.sizes["r"]):
        # Necessity of going surface by surface = defining wd_function:
        rdcon_surf = rdcon_xarray.isel(r=ri)
        wd_function = generate_wd_function(rdcon_surf,**kwargs)

        # Things needed for calculating MRE data
        Dr = rdcon_surf['Dr_surf'].values
        Di = rdcon_surf['Di_surf'].values
        Dnc = rdcon_surf['Dnc_surf'].values
        H = rdcon_surf['H_surf'].values

        # Things that we will output using the structure: tempda2
        prefac = rdcon_surf['eta_star_surf'].values/k0
        wd_at_X0 = wd_function(rdcon_surf['X0_surf'].values)

        # Update prefacs and wd_at_X0s  (POSITIONAL assignment)
        prefacs[dict(r=ri)]   = prefac
        wd_at_X0s[dict(r=ri)] = wd_at_X0

        # Check improper inputs:
        if np.isnan(Dr) or np.isnan(Di) or np.isnan(Dnc) or np.isnan(H) or Di > 0:
            continue

        # Generate DP_to_MRE function
        DP_to_MRE = mre_combination_wrap(wd_function, Dr, Di, Dnc, H, k1, C0, prefac, w_vec, w_vec_lowres)

        # Apply DP_to_MRE across all delta prime types, record results
        for dpi, Dp_type in enumerate(DP_da.Delta_prime_type):
            DP_val = DP_da.isel(r=ri, Delta_prime_type=dpi).values
            (dwdt_low_res, w_marg, w_sat,
             w_max_loc, dwdtau_max, wd_at_marg) = DP_to_MRE(DP_val)

            # POSITIONAL assignment on both r and Delta_prime_type
            idx = dict(r=ri, Delta_prime_type=dpi)
            w_margs[idx]        = w_marg
            w_sats[idx]         = w_sat
            w_max_locs[idx]     = w_max_loc
            dwdtau_maxs[idx]    = dwdtau_max
            wd_at_margs[idx]    = wd_at_marg
            dwdt_lowres_da[idx] = dwdt_low_res

    # Now we store these data arrays in code_xarray
    code_xarray = code_xarray.assign(
        w_marg_surf = w_margs,
        w_sat_surf = w_sats,
        w_max_loc_surf = w_max_locs,
        dwdtau_max_surf = dwdtau_maxs,
        prefac_surf = prefacs,
        wd_at_marg_surf = wd_at_margs,
        wd_at_X0_surf = wd_at_X0s,
        dwdt_surf = dwdt_lowres_da
    )

    # Want X0_on_w_marg_surf and X0_on_wd_at_marg_surf to be less than 0 for MRE analysis to be valid!
    code_xarray = code_xarray.assign(
        X0_on_w_marg_surf = rdcon_xarray['X0_surf']/code_xarray['w_marg_surf'],
        X0_on_wd_at_marg_surf = rdcon_xarray['X0_surf']/code_xarray['wd_at_marg_surf'],
        X0_on_wd_at_X0_surf = rdcon_xarray['X0_surf']/code_xarray['wd_at_X0_surf'])

    return code_xarray

def mre_combination_wrap(wd_function, Dr, Di, Dnc, H, k1, C0, prefac, w_vec, w_vec_lowres):
    """Build a closure that maps a Delta' value to nonlinear MRE outputs for a single surface.

    Parameters
    ----------
    wd_function : callable
        w_bar -> wd_bar mapping from generate_wd_function.
    Dr, Di, Dnc, H : float
        MRE equilibrium parameters at this surface.
    k1, C0 : float
        MRE model constants.
    prefac : float
        Prefactor converting psi_norm units to SI.
    w_vec, w_vec_lowres : array-like
        Island width grids (high-res for root-finding, low-res for output).

    Returns
    -------
    DP_to_MRE : callable
        Function(delta_prime_surf) -> (dwdt_vec, w_marg, w_sat, w_max_loc, dwdtau_max, wd_at_marg).
    """
    def DP_to_MRE(delta_prime_surf):
        dwdt_vec_low_res = np.full_like(w_vec_lowres, np.nan)
        w_marg = np.nan
        w_sat = np.nan
        w_max_loc = np.nan
        dwdtau_max = np.nan
        wd_at_marg = np.nan
        if not np.isnan(delta_prime_surf):
            dwdtau_loc = lambda w_in: dwdtau(w_in, wd_function, delta_prime_surf,
                                            Dr, Di,
                                            Dnc, H,
                                            k1, C0)
            dwdtau_vec = dwdtau_loc(w_vec)
            dwdt_vec_low_res = prefac*dwdtau_loc(w_vec_lowres)
            w_marg, w_sat, w_max_loc, dwdtau_max = extract_mre_factors(dwdtau_vec, w_vec)
            if not np.isnan(w_marg):
                wd_at_marg = wd_function(w_marg)
            else:
                wd_at_marg = np.nan
        return dwdt_vec_low_res, w_marg, w_sat, w_max_loc, dwdtau_max, wd_at_marg
    return DP_to_MRE

def extract_mre_factors(dwdtau_vec, w_vec):
    """Extract critical MRE factors from dwdtau(w) using cubic spline root-finding.

    Parameters
    ----------
    dwdtau_vec : array-like
        MRE right-hand-side evaluated over w_vec.
    w_vec : array-like
        Island width grid (normalised poloidal flux).

    Returns
    -------
    w_marg : float
        First spline root if dwdtau starts negative. 1.0 if always decaying
        (no roots, dwdtau_vec[0] < 0), 0.0 if always growing (no roots, dwdtau_vec[0] > 0).
    w_sat : float
        Last spline root if dwdtau ends negative. 0.0 if always decaying, 1.0 if
        always growing.
    w_max_loc : float
        Island width at peak dwdtau. nan if dwdtau monotonically increases to w=1 (not onset-relevant).
    dwdtau_max : float
        Peak dwdtau value at w_max_loc.
    """
    if not np.all(np.isfinite(dwdtau_vec)):
        return np.nan, np.nan, np.nan, np.nan

    dwdtau_spln=Akima1DInterpolator(w_vec,dwdtau_vec,extrapolate=False)
    dwdtau_deriv_spln=Akima1DInterpolator(w_vec,dwdtau_spln(w_vec,1),extrapolate=False)

    # Find where dwdtau crosses zero:
    zero_crossings = dwdtau_spln.roots(extrapolate=False)

    w_marg = np.nan
    w_sat = np.nan

    if len(zero_crossings) != 0:
        if dwdtau_vec[0] < 0:
            # Marginally stable island width is the first zero crossing:
            w_marg = zero_crossings[0]
        if dwdtau_vec[-1] < 0:
            # Saturated island width is the last zero crossing:
            w_sat = zero_crossings[-1]
    else:
        if dwdtau_vec[0] < 0:   # Always decaying
            w_marg = 1.0
            w_sat = 0.0
        elif dwdtau_vec[0] > 0: # Always growing
            w_marg = 0.0
            w_sat = 1.0

    # Maximum island width is where dwdtau is maximum:
    max_index = np.argmax(dwdtau_vec)
    w_max_temp = w_vec[max_index]
    dwdtau_max_temp = dwdtau_vec[max_index]

    # Check if max_index is at the start or end of the vector
    if max_index==0:
        return w_marg, w_sat, w_vec[max_index], dwdtau_vec[max_index]
    if max_index==len(dwdtau_vec)-1:
        # Look for local maxes at smaller island widths...
        w_max_temp, dwdtau_max_temp, max_index = get_local_max(w_vec,dwdtau_vec)
        if np.isnan(w_max_temp): # dwdtau monotonically increases to max at w = 1.0
            return w_marg, w_sat, np.nan, np.nan # Ignore dwdtau_max, doesn't relate to onset phenomena...

    extremum_points = dwdtau_deriv_spln.roots(extrapolate=False) # w-location of extremum points in dwdtau
    if len(extremum_points) == 0: # We found a local max but its so slight that only np.diff(dwdtau_vec)/np.diff(w_vec) will pick it up...
        # We fall back to findings of get_local_max
        return w_marg, w_sat, w_max_temp, dwdtau_max_temp

    special_ind = np.argmin(np.abs(extremum_points-w_max_temp)) # Find spline-computed extremum point closest to w_max_temp
    w_max = extremum_points[special_ind]
    dwdtau_max = dwdtau_spln(w_max)

    return w_marg, w_sat, w_max, dwdtau_max

def get_local_max(xvec,yvec):
    """Find the local maximum of yvec, returning the corresponding (x, y) pair.

    If multiple peaks exist, returns the one at the largest x value.
    Returns (nan, nan) if no peaks are found.

    Parameters
    ----------
    xvec, yvec : array-like
        x and y data arrays of equal length.

    Returns
    -------
    x_peak, y_peak : float
        Coordinates of the selected peak.
    peak_ind : int
        Index of the peak.
    """
    peak_inds = find_peaks(yvec)[0]
    if len(peak_inds) == 0:
        return np.nan, np.nan, np.nan
    elif len(peak_inds) > 1:
        print(" Warning, more than one peak in dw/dt, using peak with largest w value.")
        return xvec[peak_inds[-1]], yvec[peak_inds[-1]], peak_inds[-1]
    return xvec[peak_inds[0]], yvec[peak_inds[0]], peak_inds[0]

# When you artificially set chifrac in M3DC1, make another generate_wd_function that just uses that chifrac.
# How to implement this within the island is another question
def generate_wd_function(rdcon_xarray_surf,force_lmfp=False,iterator=False,use_Fitz_formula=False):
    """
    Generates for a particular surface, a function that takes in w_bar
    (island width in normalised poloidal flux), and returns wd_bar
    (Fitzpatrick island width in normalised poloidal flux).
    By default, takes the minimum of chi_para_lmfp and chi_para_smfp,
    but can be set to use only chi_para_lmfp. Assumes that rdcon_xarray
    has already been through cross_field_transport.py.

    If use_Fitz_formula is True, then we apply the Fitzpatrick 2023 formula 14.209. Default is no,
    since if the chi_para_smfp and chi_para_lmfp are equal, this formula cuts chi_para in half, which I don't agree with.

    If iterator is False, we calculate the ratio of chi_perp/chi_para at the specific island size being evaluated (I think this is more correct).
    Requires rdcon_xarray to have been through cross_field_transport.py.

    Parameters
    ----------
    rdcon_xarray_surf : xr.Dataset
        Single-surface slice with chi_perp_surf, chi_para_smfp_surf,
        chi_para_lmfp_no_w_surf, Wc_prefac_m_surf, X0_surf.
    force_lmfp : bool
        Use only the long-mean-free-path chi_para (ignore smfp).
    iterator : bool
        If True, iterates wd to self-consistency (ignores input w_bar).
        If False (default), evaluates chi_para at the given w_bar directly.
    use_Fitz_formula : bool
        Use Fitzpatrick 2023 Eq. 14.209 harmonic mean for chi_para.

    Returns
    -------
    wd_function : callable
        Function(w_bar) -> wd_bar.
    """

    chi_perp = rdcon_xarray_surf['chi_perp_surf'].values
    chi_para_smfp = rdcon_xarray_surf['chi_para_smfp_surf'].values
    chi_para_lmfp_no_w = rdcon_xarray_surf['chi_para_lmfp_no_w_surf'].values
    Wc_prefac_m = rdcon_xarray_surf['Wc_prefac_m_surf'].values
    X0 = rdcon_xarray_surf['X0_surf'].values

    if not iterator: # I think this is more correct
        def wd_function(w_bar: float):
            """
            Takes in w_bar (island width in normalised poloidal flux) and returns wd_bar
            (Fitzpatrick island width in normalised poloidal flux).
            """
            chi_para_lmfp = chi_para_lmfp_no_w/w_bar
            if force_lmfp:
                chi_para=chi_para_lmfp
            else:
                if not use_Fitz_formula:
                    chi_para = np.minimum(chi_para_lmfp, chi_para_smfp)
                else:
                    chi_para = chi_para_lmfp*chi_para_smfp/(chi_para_lmfp+chi_para_smfp) # If the two are equal, it cuts chi_para in half, which I don't agree with.
            chifrac = chi_perp/chi_para
            if Wc_prefac_m < 0:
                return np.nan
            return (chifrac*Wc_prefac_m)**(1/4)
    else:
        def wd_function(w_bar: float):
            """ !!! IGNORES w_bar !!!
            Takes in w_bar (island width in normalised poloidal flux) and returns wd_bar
            (Fitzpatrick island width in normalised poloidal flux).
            """
            wd_bar4=X0**4
            for i in range(10): #Iterate to convergence
                chi_para_lmfp = chi_para_lmfp_no_w/(wd_bar4**(1/4))
                if force_lmfp:
                    chi_para=chi_para_lmfp
                else:
                    if not use_Fitz_formula:
                        chi_para = np.minimum(chi_para_lmfp, chi_para_smfp)
                    else:
                        chi_para = chi_para_lmfp*chi_para_smfp/(chi_para_lmfp+chi_para_smfp)
                chifrac = chi_perp/chi_para
                wd_bar4 = Wc_prefac_m*chifrac
            return wd_bar4**(1/4)
    return wd_function

def dwdtau(w_bar: float, wd_function: 'function', DeltaPrimeGPEC: float, Dr: float, Di: float, Dnc: float, H: float, k1: float, C0: float):
    """Evaluate the right-hand side of the Modified Rutherford Equation.

    Combines Delta' drive, GGJ curvature stabilisation, and neoclassical bootstrap terms.
    From Schlutt & Hegna PoP 2012 and Hegna 1999, in normalised poloidal flux space
    (Rosenburg PoP 2002). Units: psi_p_norm^(-1).
    """
    wd_bar=wd_function(w_bar)
    return DeltaPrime_bar(w_bar, DeltaPrimeGPEC, Di) + Delta_GGJ(w_bar, wd_bar, Dr, Di, H, k1, C0) + Delta_nc(w_bar, wd_bar, Dnc, k1, C0)

def Delta_nc(w_bar: float, wd_bar: float, Dnc: float, k1: float, C0: float):
    """Neoclassical bootstrap current drive term of the MRE (Schlutt & Hegna PoP 2012).

    Converted to normalised poloidal flux space. Units: psi_norm^(-1).
    """
    return k1*Dnc*w_bar/(w_bar**2+(wd_bar**2)*k1/(C0*0.81))

def Delta_GGJ(w_bar: float, wd_bar: float, Dr: float, Di: float, H: float, k1: float, C0: float):
    """Glasser-Greene-Johnson curvature stabilisation term of the MRE (Schlutt & Hegna PoP 2012).

    Converted to normalised poloidal flux space. Note typo in that paper; to agree with Hegna 1999 in
    the toroidal limit, we use k1 instead of k0. Units: psi_norm^(-1).
    """
    alpha_l=0.5-np.sqrt(-Di)
    alpha_s=0.5+np.sqrt(-Di)
    Dh = Dr/(alpha_s-H)
    denom = w_bar + (2*k1*wd_bar)/(C0*(1+alpha_s))
    return k1*Dh/denom

def DeltaPrime_bar(w_bar: float, DeltaPrimeGPEC: float, Di: float):
    """Delta' drive term of the MRE (Hegna PoP 1999, Schlutt & Hegna PoP 2012 - note the type in the latter).

    Converted to normalised poloidal flux space as per Rosenburg PoP 2002.
    Note DeltaPrimeGPEC has units psi_norm^{-2*sqrt(-Di)}. This term has units psi_norm^(-1).
    """
    alpha_l=0.5-np.sqrt(-Di)
    return DeltaPrimeGPEC*(w_bar/2)**(-2*alpha_l)*np.sqrt(-4*Di)

