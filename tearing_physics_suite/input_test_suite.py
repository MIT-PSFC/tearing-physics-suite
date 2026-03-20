# Python functions to test the numerical and physics stability of delta' calculations for a given equilibrium
# Tests scan all input parameters using the scan_1D_input function. Refer to input_test_runner.py to run multiple tests at once.

# README:
#   All computations are done with the default wrapper inputvalues unless otherwise specified.
#   All computation is internal mode (vac_flag = 'f') unless otherwise specified.
#   All computations are done with nn=1 unless otherwise specified.
# quick_test = True <=> deviate slightly from default values to make sure no large changes are detected.
# quick_test = False <=> Proper scoping over reasonable the reasonable range for a given input parameter.

import os
import pandas as pd
import xarray as xr
import numpy as np

from tearing_physics_suite.utils import create_dense_log_paramvals
from tearing_physics_suite.environment import home_dir
from tearing_physics_suite.input_scans import scan_1D_input

#########################################################################################################
# NUMERICAL STABILITY TESTS:
#########################################################################################################
#########################################################################################################
# Varying number of fourier modes used in analysis:
#########################################################################################################
def fourier_mode_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the fourier modes used in resistive calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [7,8]
        else:
            scan_vals = [0,1,2,3,4,5,6,7,8]
    result, message = scan_1D_input('delta_mhigh', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','fourier_mode_scan'),
            set_delta_mlow_to_delta_mhigh=True,
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Fourier mode scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying order of asymptotic expansion:
#########################################################################################################
def sing_order_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the sing_order used in STRIDE and RDCON calculation.

    """
    # Note: PEST3 doesn't have an option to vary sing_order, so we don't run it here.
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [5,6]
        else:
            scan_vals = [0,1,2,3,4,5,6]
    result, message = scan_1D_input('sing_order', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','sing_order_scan'),
            vac_flag='f',
            run_pest3=False # PEST3 doesn't have an option to vary sing_order
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Asymptotic expansion order scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying asmpyotic matching point:
#########################################################################################################
def matching_point_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the matching point used in STRIDE and RDCON calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [1e-3,2e-3]
        else:
            scan_vals = tps.create_dense_log_paramvals(start=1e-1,end=1e-6,points_per_decade=2)
    result, message = scan_1D_input('dx0', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','matching_point_scan'),
            set_dx1dx2_dx0_mult=2,
            set_singfac_min_to_dx=True, 
            vac_flag='f',
            run_pest3=False # PEST3 numerical method passes through each rational surface...
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Matching point scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying psilow truncation point:
#########################################################################################################
def psilow_truncation_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psilow truncation point used in STRIDE and RDCON calculation.

    delta_prime_variability-applicable in most cases.
    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [1e-4,2e-4]
        else:
            scan_vals = tps.create_dense_log_paramvals(start=1e-1,end=1e-5,points_per_decade=2)
    result, message = scan_1D_input('psilow', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','psilow_truncation_scan'),
            vac_flag='f',
            run_pest3=False # PEST3 always extrapolates to the magnetic axis
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psilow truncation scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying equilibrium integration tolerance:
#########################################################################################################
def equilibrium_integrator_tolerance_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the equilibrium integration tolerance used in STRIDE and RDCON calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [1e-9,1e-10]
        else:
            scan_vals = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11]
    result, message = scan_1D_input('etol', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','equilibrium_integrator_tolerance_scan'),
            run_pest3=False,
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Equilibrium integrator tolerance scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying mtheta:
#########################################################################################################
def mtheta_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mtheta used in resistive calculation.

    """
    # check if scan_vals in kwargs, if so use those instead of default values
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [129,130,257,513]
        else:
            scan_vals = [70,90,129,257,513,1025]
    result, message = scan_1D_input('mtheta', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','mtheta_scan'),
            vac_flag='f',
            pest_pull_mtheta=True
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("mtheta scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying mtheta: no wall
#########################################################################################################
def mtheta_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mtheta used in the resistive calculation, with no wall.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [129,130,257,513]
        else:
            scan_vals = [70,90,129,257,513,1025]
    result, message = scan_1D_input('mtheta', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','mtheta_scan'),
            vac_flag='t',
            output_prefix='no_wall_',
            pest_pull_mtheta=True
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("mtheta scan results (no wall):")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying mpsi:
#########################################################################################################
def mpsi_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mpsi used in the resistive calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [257,258,513]
        else:
            scan_vals = [70,90,129,257,513,1025]
    result, message = scan_1D_input('mpsi', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','mpsi_scan'),
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("mpsi scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying mpsi: no wall
#########################################################################################################
def mpsi_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mpsi used in the resistive calculation, with no wall.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [257,258,513]
        else:
            scan_vals = [70,90,129,257,513,1025]
    result, message = scan_1D_input('mpsi', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','mpsi_scan'),
            vac_flag='t',
            output_prefix='no_wall_'
            ,**kwargs)

    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("mpsi scan results (no wall):")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying psihigh truncation within q surface:
#########################################################################################################
def edge_truncation_within_surface_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),psihigh=0.998,quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [0.15,0.2]
        else:
            scan_vals = [0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            print("Warning: it's likely the q-surface being truncated will jump by one over the course of this scan. Read edge_truncation_within_surface_scan & fiddle with psihigh to stop this happening.")
    result, message = scan_1D_input('dmlim', scan_vals, eq_filename,
            1,
            psihigh=psihigh,
            output_location=os.path.join(results_dir,'1D_scans','edge_truncation_within_surface_scan'),
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psihigh truncation within q surface scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying psihigh truncation within q surface: no wall
#########################################################################################################
def edge_truncation_within_surface_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation, with no wall.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [0.15,0.2]
        else:
            scan_vals = [0.00001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            print("Warning: it's likely the q-surface being truncated will jump by one over the course of this scan. Read edge_truncation_within_surface_scan_no_wall & fiddle with psihigh to stop this happening.")
    result, message = scan_1D_input('dmlim', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','edge_truncation_within_surface_scan'),
            vac_flag='t',
            output_prefix='no_wall_'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psihigh truncation within q surface scan results (no wall):")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying number of finite elements: RDCON
#########################################################################################################
def rdcon_finite_element_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the number of finite elements used in RDCON calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [257,128]
        else:
            scan_vals = [60,70,80,90,100,120,257,512]
    result, message = scan_1D_input('nx', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','nx_rdcon_scan'),
            run_pest3=False, 
            run_stride=False, 
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("RDCON finite element scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying number of finite elements: PEST3
#########################################################################################################
def pest3_finite_element_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,debug=False,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the number of finite elements used in PEST3 calculation.


    The correct Delta prime value is calculated from extrapolating a straight line in nx_pest^(-2) to nx_pest = infinity (nx_pest^(-2) = 0).
    HOWEVER, above a certain value of nx_pest (~ > 140 < 200), the Delta prime values no longer obey a straight
    line relationship w.r.t nx_pest^(-2). (Higher res. equilibria push this point to higher nx_pest values - Dylan Brennan). You want 
    to use values of nx_pest that are below this point to ensure convergence is correctly calculated.
    """
    if debug:
        gal_flag='f'
        run_stride=False
    else:
        gal_flag='t'
        run_stride=True
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = ['''-k"100 50 80 140"''','''-k"90 46 70 130"''','''-k"70 30 50 100"''']
            var_name='nx_string_pest'
        else:
            scan_vals = [30,40,50,60,70,80,90,100,120,140,160,180,200,225,257,300,400,512]
            var_name='nx_pest'
    else:
        # If scan_vals is provided, check first element to determine whether we're scanning over nx_pest or nx_string_pest
        if isinstance(scan_vals[0], str):
            var_name='nx_string_pest'
        else:
            var_name='nx_pest'
    result, message = scan_1D_input(var_name, scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','nx_pest3_scan'),
            run_pest3=True, 
            run_stride=True, 
            run_rdcon=False, # Run it for comparison
            gal_flag=gal_flag, 
            ode_flag='f', # Speeds up rdcon
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("PEST3 finite element scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying integration tolerance near rational surfaces:
#########################################################################################################
def integrator_tolerance_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the integrator tolerance used in STRIDE and RDCON calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [1e-9,1e-10]
        else:
            scan_vals = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11]
    result, message = scan_1D_input('tol_r', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','integrator_tolerance_scan'),
            set_int_tolerances_equal=True,
            run_pest3=False, 
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Integrator tolerance scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying magnetic coordinate type:
#########################################################################################################
def mag_coord_type_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the magnetic coordinate type used in STRIDE and RDCON calculation.

    """
    # Note: PEST3 doesn't have an option to vary jac_type, so we don't run it here.
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = ["""'hamada'""","""'pest'"""]
        else:
            scan_vals = ["""'hamada'""","""'boozer'""","""'pest'""","""'equal_arc'"""]
    result, message = scan_1D_input('jac_type', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','mag_coord_type_scan'),
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Magnetic coordinate type scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying grid_type:
#########################################################################################################
def grid_type_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the grid type used in STRIDE and RDCON calculation.

    """
    # Note: PEST3 doesn't have an option to vary grid_type, so we don't run it here.
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = ["""'ldp'""","""'pow1'"""]
        else:
            scan_vals = ["""'ldp'""","""'pow1'""","""'pow2'""","""'rho'""","""'original'"""] #original breaks...
    result, message = scan_1D_input('grid_type', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','grid_pack_scan'),
            vac_flag='f',
            run_pest3=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Grid type scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying vacuum theta spline density:
#########################################################################################################
def vacuum_mtheta_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the amount of theta spline points for the vacuum-plasma interface.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [960,1200]
        else:
            scan_vals = [80,100,200,300,400,500,600,700,800,900,1000,1200,1500,2000]
    result, message = scan_1D_input('mthvac', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','mthvac_scan'),
            vac_flag='t',
            a_wall=0.1,
            run_pest3=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Vacuum theta spline density scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 

#EXTRA:
#########################################################################################################
# Varying cutoff: n = 1
#########################################################################################################
def RDCON_cutoff_scan_n1(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the cutoff value used in the resistive calculation in RDCON.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [9,10]
        else:
            scan_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    result, message = scan_1D_input('cutoff', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','RDCON_cutoff_scan'),
            vac_flag='f',
            run_pest3=False,
            run_stride=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("RDCON cutoff scan results: n = 1")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying cutoff: n = 4
#########################################################################################################
def RDCON_cutoff_scan_n4(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the cutoff value used in the resistive calculation in RDCON.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [9,10]
        else:
            scan_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    result, message = scan_1D_input('cutoff', scan_vals, eq_filename,
            4,
            output_location=os.path.join(results_dir,'1D_scans','RDCON_cutoff_scan'),
            vac_flag='f',
            run_pest3=False,
            run_stride=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("RDCON cutoff scan results: n = 4")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying solver: 
#########################################################################################################
def RDCON_solver_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),verbose=True,output_prefix='',quick_test=False, **kwargs):
    """
    Runs a 1D scan over the solver used in the resistive calculation in RDCON.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        scan_vals = ["""'LU'""","""'cholesky'"""]
    result, message = scan_1D_input('solver', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','solver_scan'),
            vac_flag='f',
            run_pest3=False,
            run_stride=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("RDCON solver scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# Varying axis_mid_pt_skew: 
#########################################################################################################
def STRIDE_axis_mid_pt_skew_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the axis_mid_pt_skew used in the resistive calculation in STRIDE.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [11.0,12.0,13.0]
        else:
            scan_vals = [2.0,4.0,8.0,10.0,12.0,14.0,18.0,20.0]
    result, message = scan_1D_input('axis_mid_pt_skew', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','STRIDE_axis_mid_pt_skew_scan'),
            vac_flag='f',
            run_pest3=False,
            run_rdcon=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("STRIDE axis_mid_pt_skew scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# large_sol_extent_pest
#########################################################################################################
def large_sol_extent_pest_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the large_sol_extent_pest used in the resistive calculation in PEST3.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [0.8,0.9]
        else:
            scan_vals = [0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95]
    result, message = scan_1D_input('large_sol_extent_pest', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','large_sol_extent_pest_scan'),
            vac_flag='f',
            run_pest3=True,
            run_stride=False,
            run_rdcon=True, # Run it for truncation logic, not delta'
            gal_flag='f', # No delta' from rdcon
            ode_flag='f' # Speeds up rdcon
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("PEST3 large_sol_extent_pest scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 

#EXTRA:
#########################################################################################################
# Varying nq: 
#########################################################################################################
def RDCON_nq_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the number of Gaussian points in each Galerkin element in the resistive calculation in RDCON.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [5,6]
        else:
            scan_vals = [1,2,3,4,5,6,7]
    result, message = scan_1D_input('nq', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','RDCON_nq_scan'),
            vac_flag='f',
            run_pest3=False,
            run_stride=False
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("RDCON nq scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# sing1_flag
#########################################################################################################
def sing1_flag_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over sing1_flag used in the resistive calculation in RDCON.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        scan_vals = ['t','f']
    result, message = scan_1D_input('sing1_flag', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','sing1_flag_scan'),
            vac_flag='f',
            run_pest3=False,
            run_stride=True,
            run_rdcon=True
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("sing1_flag scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# regrid_flag - probably broken
#########################################################################################################
def regrid_flag_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over regrid_flag used in the resistive calculation in RDCON.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        scan_vals = ['t','f']
    result, message = scan_1D_input('regrid_flag', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','regrid_flag_scan'),
            vac_flag='f',
            run_pest3=False,
            run_stride=True,
            run_rdcon=True
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("regrid_flag scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# RDCON/STRIDE inputs I'm ignoring:
#   pfac - varying nx in rdcon will determine whether we need more finite elements near the rational surfaces
#   sing_start - not really interested in cutting out the effect of various rational surfaces
#                if q_0 < 1, and the plasma is ideal unstable, then delta' at other surfaces doesn't mean much
#   crossover - effect will allow speedup by varying tol_r, tol_nr
#   ucrit - doesn't effect resistive calculations in stride or rdcon
#   nIntervalsTot - is autmatically increased by stride to minimally cover the number of singular intervals
#                   seems to affect threading/speed
#########################################################################################################

#########################################################################################################
# PHYSICS PARAMETER DEPENDENCIES:
#########################################################################################################
#########################################################################################################
# varying q-surface truncation:
#########################################################################################################
def edge_truncation_q_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation.

    delta_prime_variability may break
    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [5.2,6.2,7.2]
        else:
            scan_vals = [2.2,3.2,4.2,5.2,6.2,7.2,8.2]
    result, message = scan_1D_input('qhigh', scan_vals, eq_filename,
            1,
            psihigh=0.9999,
            ode_flag='f', #avoid numerical instability at high psihigh for DCON shooting method in RDCON
            output_location=os.path.join(results_dir,'1D_scans','edge_truncation_q_scan'),
            vac_flag='f',
            sas_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psihigh truncation scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 

# This function can pick up a discrepancy in truncation between rdcon and stride for the ideal test case equilibrium
# Needs fix to truncation logic in GPEC when both sasflag and qhigh are set - """qhigh overrode sasflag in RDCON but sasflag overrode qhigh in STRIDE"""
def edge_truncation_q_scan_DEPRECATED(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation.
    """
    if quick_test:
        scan_vals = [4,5,6]
    else:
        scan_vals = [2,3,4,5,6,7,8]
    result, message = scan_1D_input('qhigh', scan_vals, eq_filename,
            1,
            psihigh=0.9999,
            ode_flag='f', #avoid numerical instability at high psihigh for DCON shooting method in RDCON
            output_location=os.path.join(results_dir,'1D_scans','edge_truncation_q_scan'),
            vac_flag='f'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psihigh truncation scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# varying q-surface truncation: no wall
#########################################################################################################
def edge_truncation_q_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation, with no wall.

    delta_prime_variability may break.
    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if quick_test:
            scan_vals = [5.2,6.2,7.2]
        else:
            scan_vals = [2.2,3.2,4.2,5.2,6.2,7.2,8.2]
    result, message = scan_1D_input('qhigh', scan_vals, eq_filename,
            1,
            psihigh=0.9999,
            ode_flag='f', #avoid numerical instability at high psihigh for DCON shooting method in RDCON
            output_location=os.path.join(results_dir,'1D_scans','edge_truncation_q_scan'),
            vac_flag='t',
            output_prefix='no_wall_',
            sas_flag='f' 
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psihigh truncation scan results (no wall):")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 

# Needs fix to truncation logic in GPEC when both sasflag and qhigh are set
def edge_truncation_q_scan_no_wall_DEPRECATED(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation, with no wall.
    """
    if quick_test:
        scan_vals = [5,6,7]
    else:
        scan_vals = [2,3,4,5,6,7,8]
    result, message = scan_1D_input('qhigh', scan_vals, eq_filename,
            1,
            psihigh=0.9999,
            ode_flag='f', #avoid numerical instability at high psihigh for DCON shooting method in RDCON
            output_location=os.path.join(results_dir,'1D_scans','edge_truncation_q_scan'),
            vac_flag='t',
            output_prefix='no_wall_'
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Psihigh truncation scan results (no wall):")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 
#########################################################################################################
# varying wall radius:
#########################################################################################################
def wall_radius_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),superquick=False, quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the wall radius used in the resistive calculation.

    """
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if superquick:
            scan_vals = [0]
        elif quick_test:
            scan_vals = [0,0.1,0.2,21]
        else:
            scan_vals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,15,21]
    result, message = scan_1D_input('a_wall', scan_vals, eq_filename,
            1,
            output_location=os.path.join(results_dir,'1D_scans','wall_scan')
            ,**kwargs)
    if verbose:
        print("#########################################################################################################")
        print("#########################################################################################################")
        print("Wall radius scan results:")
        print(message)
        print("#########################################################################################################")
        print("#########################################################################################################")
    return result, message 


# Things that might vary for certain equilibria
key_numerical_tests_n4 = [
    'mtheta_scan',
    'mpsi_scan',
    'edge_truncation_within_surface_scan_no_wall',
    'pest3_finite_element_scan',
    'mag_coord_type_scan',
    'grid_type_scan',
    'vacuum_mtheta_scan',
    'RDCON_cutoff_scan_n4'
]

key_numerical_tests = [
    'mtheta_scan',
    'mpsi_scan',
    'edge_truncation_within_surface_scan_no_wall',
    'pest3_finite_element_scan',
    'mag_coord_type_scan',
    'grid_type_scan',
    'vacuum_mtheta_scan',
]

# Will have a big effect (physics motivated)
physics_tests = [
    'edge_truncation_q_scan',
    'edge_truncation_q_scan_no_wall',
    'wall_radius_scan'
]

# Things I expect to barely change things
extra_numerical_tests = [
    'fourier_mode_scan',
    'sing_order_scan',
    'matching_point_scan',
    'psilow_truncation_scan',
    'equilibrium_integrator_tolerance_scan',
    'mtheta_scan_no_wall',
    'mpsi_scan_no_wall',
    'edge_truncation_within_surface_scan',
    'rdcon_finite_element_scan',
    'integrator_tolerance_scan',
    'RDCON_cutoff_scan_n1',
    'RDCON_solver_scan',
    'STRIDE_axis_mid_pt_skew_scan',
    'large_sol_extent_pest_scan',
    'RDCON_nq_scan',
    'sing1_flag_scan',
    'regrid_flag_scan'
]

#List the name of every scan function defined in this file:
scan_functions = [
    'fourier_mode_scan',
    'RDCON_solver_scan',
    'sing_order_scan',
    'matching_point_scan',
    'psilow_truncation_scan',
    'equilibrium_integrator_tolerance_scan',
    'mtheta_scan',
    'mtheta_scan_no_wall',
    'mpsi_scan',
    'mpsi_scan_no_wall',
    'edge_truncation_within_surface_scan',
    'edge_truncation_within_surface_scan_no_wall',
    'rdcon_finite_element_scan',
    'pest3_finite_element_scan',
    'integrator_tolerance_scan',
    'mag_coord_type_scan',
    'grid_type_scan',
    'vacuum_mtheta_scan',
    'RDCON_cutoff_scan_n1',
    'RDCON_cutoff_scan_n4',
    'STRIDE_axis_mid_pt_skew_scan',
    'large_sol_extent_pest_scan',
    'RDCON_nq_scan',
    'sing1_flag_scan',
    'regrid_flag_scan',
    'edge_truncation_q_scan',
    'edge_truncation_q_scan_no_wall',
    'wall_radius_scan'
]
