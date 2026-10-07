# Python functions to call GPEC and PEST3 fortran codes for delta prime calculations

import os

import numpy as np

home_dir = os.environ['TPSHOME']
from tearing_physics_suite.wrappers.gpec import GPEC_resistive_calculation, _default_eq_type
from tearing_physics_suite.wrappers.pest3 import PEST3_resistive_calculation, pest3_special_truncation_loop


def run_resistive_calculation(eq_filename, nn, run_rdcon=True, run_stride=True, run_pest3=True,
        make_working_dir=True,
        make_results_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        gpec_dir=os.path.join(home_dir, 'submodules/GPEC'),
        pest3_dir=os.path.join(home_dir, 'submodules/PEST3/cmake_build/pest3'),
        verbose=True,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        save_input=True,
        save_terminal_output=False, #Currently broken
        pest_match_truncation=True,
        override_save=True,
        pest_pull_mtheta=True, # Change at your own risk, see mtheta_scan scan results
        debug_GPEC_resistive_calculation=False, #Quick exit after GPEC resistive calculation
        ascii_q_plot=False,       # Print an ascii q-profile plot after the GPEC run
        **kwargs):
    """
    Run resistive toroidal calculation for a single toroidal mode number by calling
    the GPEC and PEST3 fortran executables in a working directory.
    Prints files both to the working directory and to output_location if specified.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file. Cannot be too long (Fortran path length limitation).
    nn : int
        Toroidal mode number.
    run_rdcon, run_stride, run_pest3 : bool
        Whether to run each code.
    working_dir : str
        Path to the working directory.
    verbose : bool
        Print verbose output.
    fresh_start : bool
        Remove existing calculations from working_dir before running.
    output_location : str or None
        If specified, save output files to this location.
    output_prefix : str
        Prefix for output filenames.
    pest_match_truncation : bool
        Match PEST3 truncation to GPEC/STRIDE. Overridden by psihigh_pest kwarg.
    **kwargs
        Forwarded to write_rdcon_stride_inputs and PEST3_resistive_calculation.

    Returns
    -------
    rdcon_xr, stride_xr, pest3_xr : xr.Dataset or None
        Output xarrays from each code.
    rdcon_ran, stride_ran, pest3_ran : bool
        Whether each code ran successfully.
    rdcon_stride_input_dict, pest3_input_dict : dict or None
        Input parameters used for each calculation.
    """

    #Extract keyword arguments for PEST3
    pest3_kwargs_dict = {k: v for k, v in kwargs.items() if k.endswith('_pest')}
    if not override_save:
        pest3_kwargs_dict.update({"override_save":False})

    #Remove PEST3 specific keyword arguments from kwargs
    for key in pest3_kwargs_dict.keys():
        if key in kwargs:
            del kwargs[key]

    # Make equilibrium type consistent (default case is eqdsk):
    _default_eq_type(eq_filename, kwargs)
    if 'eq_type' in kwargs:
        if kwargs['eq_type'] == """'ldp_i'""" or kwargs['eq_type'] == '''"ldp_i"''':
            pest3_kwargs_dict['eq_type_pest']=8

    # Run GPEC resistive calculation
    rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict = GPEC_resistive_calculation(
        eq_filename=eq_filename, nn=nn, run_rdcon=run_rdcon, run_stride=run_stride,
        make_working_dir=make_working_dir, working_dir=working_dir, gpec_dir=gpec_dir,
        verbose=verbose, fresh_start=fresh_start, output_location=output_location,
        output_prefix=output_prefix, save_input=save_input, save_terminal_output=save_terminal_output,
        **kwargs)

    if debug_GPEC_resistive_calculation:
        return rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict

    if ascii_q_plot and rdcon_xr is not None:
        ascii_q_plotter(rdcon_xr)

    #########################################################################################################
    # Set up pest3 calculation:
    #########################################################################################################

    # Extract key truncation values from GPEC calculation:
    qlim_actual = -100
    psilow_actual = 100
    qlim_actuals = -100
    psilow_actuals = 100
    q_rationals = None
    r = None
    r_prime = None

    # Define maximum poloidal fourier harmonic with the same logic as in GPEC:
    m_max=0
    m_maxs=0
    m_min=0
    m_mins=0
    delta_mhigh=rdcon_stride_input_dict['delta_mhigh']
    delta_mlow=rdcon_stride_input_dict['delta_mlow']
    num_rat_surfaces=0

    # Pull truncation and poloidal mode information from GPEC calculations:
    if (run_rdcon and rdcon_ran):
        m_max = int(np.ceil(rdcon_xr.qmax*nn+delta_mhigh))
        m_min = int(np.floor(min(rdcon_xr.qmin*nn,0)-4-delta_mlow))
        qlim_actual = max(rdcon_xr.qlim, qlim_actual)
        psilow_actual = min(rdcon_xr.psilow, psilow_actual)
        q_rationals = rdcon_xr.q_rational.values
        r = rdcon_xr.r
        r_prime = rdcon_xr.r_prime
        # Check if rdcon generated Delta_prime
        if "Delta_prime" in rdcon_xr.data_vars:
            num_rat_surfaces=max(len(rdcon_xr.Delta_prime.isel(i=0,r_prime=0).values),num_rat_surfaces)
    if (run_stride and stride_ran):
        m_maxs = int(np.ceil(stride_xr.qmax*nn+delta_mhigh))
        m_mins = int(np.floor(min(stride_xr.qmin*nn,0)-4-delta_mlow))
        qlim_actuals = max(stride_xr.qlim, qlim_actuals)
        psilow_actuals = min(stride_xr.psilow, psilow_actuals)
        if "Delta_prime" in stride_xr.data_vars:
            num_rat_surfaces=max(len(stride_xr.Delta_prime.isel(i=0,r_prime=0).values),num_rat_surfaces)
        if q_rationals is None:
            q_rationals = stride_xr.q_rational.values
            r = stride_xr.r
            r_prime = stride_xr.r_prime
        else:
            #Check they are close in values:
            if len(q_rationals) == len(stride_xr.q_rational.values):
                if np.max(np.abs(q_rationals - stride_xr.q_rational.values)) > 1e-5:
                    raise ValueError("Rational surfaces from rdcon and stride differ. Cannot match truncation.")
            else:
                raise ValueError("Rational surfaces from rdcon and stride differ. Cannot match truncation.")
    m_max = max(m_max, m_maxs)
    m_min = min(m_min, m_mins)
    m_absmax = max(abs(m_max), abs(m_min))

    # Set truncation values:
    if (run_stride and stride_ran) and (run_rdcon and rdcon_ran):
        if 2*(abs(qlim_actual-qlim_actuals)/(abs(qlim_actual)+abs(qlim_actual))) > 1e-4:
            print("WARNING, rdcon and stride truncation is differing.")
            print("qlims: ", qlim_actual, qlim_actuals)
        if 2*(abs(psilow_actual-psilow_actuals)/(abs(psilow_actuals)+abs(psilow_actuals))) > 1e-4:
            print("WARNING, rdcon and stride truncation is differing.")
            print("psilows: ", psilow_actual,psilow_actuals)
        qlim_actual = min(qlim_actual, qlim_actuals)
        psilow_actual = max(psilow_actual,psilow_actuals)
    elif (run_stride and stride_ran):
        qlim_actual=qlim_actuals
        psilow_actual=psilow_actuals
    if verbose: print("Truncation values: psilow", psilow_actual,"qlim", qlim_actual)

    # Check for explicit pest inputs, else use the values from rdcon_stride_input_dict:
    if ('kband_pest' not in pest3_kwargs_dict) and m_absmax>0:
        pest3_kwargs_dict['kband_pest'] = m_absmax

    if 'psilow_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['psilow_pest'] = rdcon_stride_input_dict['psilow']

    if 'psihigh_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['psihigh_pest'] = rdcon_stride_input_dict['psihigh']
        allow_trunc_loop=True
    else:
        if verbose: print("Using user defined PEST3 truncation value psihigh_pest = ", pest3_kwargs_dict['psihigh_pest'])
        allow_trunc_loop=False #If you enter psihigh_pest, will override automatic truncation loop

    gpec_vacuum_pest = pest3_kwargs_dict.get('vacuum_source_pest', 'gpec') == 'gpec'
    if 'a_wall_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['a_wall_pest'] = rdcon_stride_input_dict['a_wall']
        if gpec_vacuum_pest and rdcon_stride_input_dict['vac_flag']=='t' and pest3_kwargs_dict['a_wall_pest'] <= 0:
            pest3_kwargs_dict['a_wall_pest'] = 1 # a_wall_pest > 0 turns the vacuum on; the wall comes from vac.in
        if not gpec_vacuum_pest and rdcon_stride_input_dict['ishape']!=6 and rdcon_stride_input_dict['vac_flag']=='t':
            print("WARNING: PEST3 cannot replicate GPEC's wall shape - calculation will differ.")

    if 'mthvac_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['mthvac_pest'] = rdcon_stride_input_dict['mthvac']

    if ('mtheta_pest' not in pest3_kwargs_dict) and pest_pull_mtheta:
        pest3_kwargs_dict['mtheta_pest'] = rdcon_stride_input_dict['mtheta']

    if 'mpsi_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['mpsi_pest'] = rdcon_stride_input_dict['mpsi']

    if 'rational_surface_control_pest' not in pest3_kwargs_dict:
        if num_rat_surfaces > 15:
            pest3_kwargs_dict['rational_surface_control_pest'] = '''-m"''' + 'x'*num_rat_surfaces + '''"''' # Compute Delta's for first num_rat_surfaces rational surfaces
        if num_rat_surfaces == 0 and nn*5 < 15: # Large n, don't forget to include all these rational surfaces
            pest3_kwargs_dict['rational_surface_control_pest'] = '''-m"''' + 'x'*nn*5 + '''"'''

    #########################################################################################################
    # Run PEST3 resistive calculation:
    #########################################################################################################

    if run_pest3:
        pest3_trunc_ran = False
        try:
            if pest_match_truncation and ((run_rdcon and rdcon_ran) or (run_stride and stride_ran)) and allow_trunc_loop:
                if verbose: print("Running pest3 truncation algorithm. qlim_actual = ", qlim_actual)
                if psilow_actual > rdcon_stride_input_dict['psilow'] and (psilow_actual != 100):
                    print("WARNING: axis truncation in RDCON/STRIDE differs from PEST3 truncation. Results may not be comparable.")
                if qlim_actual > -100:
                    psihigh_trunc_pest, pest3_trunc_ran = pest3_special_truncation_loop(eq_filename, nn, qlim_actual, pest3_kwargs_dict,
                        make_working_dir=make_working_dir,
                        working_dir=working_dir,
                        pest3_dir=pest3_dir,
                        verbose=verbose,
                        fresh_start=fresh_start,
                        output_location=None,
                        save_input=False,
                        output_prefix_special=output_prefix,
                        save_terminal_output=save_terminal_output,
                    )
                    if pest3_trunc_ran:
                        pest3_kwargs_dict['psihigh_pest'] = psihigh_trunc_pest
        except Exception as e:
            print(f"WARNING: PEST3 truncation loop failed: {e}")

        if not ((run_rdcon and rdcon_ran) or (run_stride and stride_ran)):
            print("**************************************************************** WARNING **********************************************************************")
            print("PEST3 calculation is running without rdcon or stride results to match truncation to. Results may not be comparable to GPEC calculations.")
            print("**************************************************************** WARNING **********************************************************************")
            raise RuntimeError

        try:
            pest3_xr, pest3_ran, pest3_input_dict = PEST3_resistive_calculation(
                eq_filename=eq_filename, nn=nn, make_working_dir=make_working_dir,
                working_dir=working_dir, pest3_dir=pest3_dir, verbose=verbose,
                fresh_start=fresh_start, output_location=output_location,
                output_prefix=output_prefix, save_input=save_input,
                save_terminal_output=save_terminal_output, q_rationals=q_rationals, r=r, r_prime=r_prime, **pest3_kwargs_dict)
        except Exception as e:
            pest3_ran = False
            pest3_xr = None
            pest3_input_dict = None
            print(f"WARNING: PEST3 run failed: {e}")

        if verbose:
            print("Pest3 ran:",pest3_ran, "Pest3 truncation ran:", pest3_trunc_ran)

        # See how accurate the truncation was:
        if pest3_ran and pest3_trunc_ran:
            if verbose:
                print(f"Verbose output: PEST3 q-truncation point {pest3_xr['qa'].max().values}, GPEC q-truncation point {qlim_actual}.")
            if abs(pest3_xr['qa'].max().values - qlim_actual) > 0.01:
                print(f"WARNING: PEST3 q-truncation point {pest3_xr['qa'].max().values} differs from GPEC q-truncation point {qlim_actual}. Results may not be comparable.")
    else:
        pest3_xr = None
        pest3_ran = False
        pest3_input_dict = None

    # Return all results:
    return rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict


def ascii_q_plotter(rdcon_xr):
    """ Takes rdcon_xr, prints an ascii plot of rdcon_xr.psi_n versus rdcon_xr.q.
    rdcon_xr.psi_n ranges from 0 to 1 (60-500 pts), rdcon_xr.q lies roughly between
    0.8 and 7, with extra vertical resolution given to the region q < 3.
    Also overlays the rational-surface points (rdcon_xr.psi_n_rational,
    rdcon_xr.q_rational). Plot is ~150 chars wide and < 20 rows high. """

    # --- pull data out as plain numpy -------------------------------------
    psi   = np.asarray(rdcon_xr.psi_n).ravel()
    q     = np.asarray(rdcon_xr.q).ravel()
    psi_r = np.asarray(rdcon_xr.psi_n_rational).ravel()
    q_r   = np.asarray(rdcon_xr.q_rational).ravel()

    # --- plot geometry -----------------------------------------------------
    label_w = 5                 # chars reserved for the y-axis labels
    plot_w  = 150 - label_w - 1 # leave room for label + "|"
    plot_h  = 18                # rows (< 20)

    split_q  = 3.0              # below this we want detail
    frac_low = 0.70             # fraction of vertical space for [qmin, split_q]

    qmin = min(0.8, float(np.nanmin(q)))
    qmax = max(float(np.nanmax(q)), float(np.nanmax(q_r)) if q_r.size else 0.0)
    qmax = max(qmax, split_q + 1e-9)   # guard

    # --- nonlinear y mapping: q -> fractional height in [0, 1] -------------
    def q_to_frac(val):
        val = np.clip(val, qmin, qmax)
        if val <= split_q:
            return (val - qmin) / (split_q - qmin) * frac_low
        return frac_low + (val - split_q) / (qmax - split_q) * (1.0 - frac_low)

    def q_to_row(val):
        f = q_to_frac(val)
        return int(round((1.0 - f) * (plot_h - 1)))   # row 0 = top

    def psi_to_col(val):
        val = np.clip(val, 0.0, 1.0)
        return int(round(val * (plot_w - 1)))

    # --- build the grid ----------------------------------------------------
    grid = [[' '] * plot_w for _ in range(plot_h)]

    # main q profile
    for p, qq in zip(psi, q):
        if np.isfinite(p) and np.isfinite(qq):
            grid[q_to_row(qq)][psi_to_col(p)] = '.'

    # rational surfaces (overlay, take priority)
    for p, qq in zip(psi_r, q_r):
        if np.isfinite(p) and np.isfinite(qq):
            grid[q_to_row(qq)][psi_to_col(p)] = 'x'

    # --- y-axis labels at integer q values --------------------------------
    row_label = {}
    for qi in range(int(np.ceil(qmin)), int(np.floor(qmax)) + 1):
        row_label[q_to_row(qi)] = f"{qi:>{label_w}.0f}"

    # --- render ------------------------------------------------------------
    print(f"q vs psi_n   (x = rational surface;  q<{split_q:.0f} region expanded)")
    for r in range(plot_h):
        lab = row_label.get(r, ' ' * label_w)
        print(lab + '|' + ''.join(grid[r]))

    # x-axis
    print(' ' * label_w + '+' + '-' * plot_w)
    axis = [' '] * plot_w
    for xt in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = f"{xt:.2f}"
        c = psi_to_col(xt)
        start = min(max(c - len(s) // 2, 0), plot_w - len(s))
        for i, ch in enumerate(s):
            axis[start + i] = ch
    print(' ' * (label_w + 1) + ''.join(axis) + '   psi_n')




