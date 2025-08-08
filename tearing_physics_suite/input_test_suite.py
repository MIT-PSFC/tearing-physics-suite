# Python functions test the numerical and physics stability of delta' calculations for a given equilibrium
# Tests scan all input parameters using scan_1D_input. Use run_multiple_scans with a list of input functions 
# to run multiple tests in this file (default is all tests).

# README:
#   All computations are done with the default wrapper inputvalues unless otherwise specified.
#   All computation is internal mode (vac_flag = 'f') unless otherwise specified.
#   All computations are done with nn=1 unless otherwise specified.
# quick_test = True <=> deviate slightly from default values to make sure no large changes are detected.
# quick_test = False <=> Proper scoping over reasonable the reasonable range for a given input parameter.

# STILL TO DO:
# - change inputs to make the default case handle tokamaker equilibria better: Cholesky, lower psihigh
# - formalise this into a test suite (test for <1% difference in single helicity delta prime for quicktest = True)
# - fix sasflag vs qhigh truncation logic in GPEC - remove functions labelled as 'DEPRECATED'
# - 2D scans?

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
    if quick_test:
        scan_vals = [7,8]
    else:
        scan_vals = [0,1,2,3,4,5,6,7,8]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('delta_mhigh', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying order of asymptotic expansion:
#########################################################################################################
def sing_order_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the sing_order used in STRIDE and RDCON calculation.
    """
    # Note: PEST3 doesn't have an option to vary sing_order, so we don't run it here.
    if quick_test:
        scan_vals = [5,6]
    else:
        scan_vals = [0,1,2,3,4,5,6]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('sing_order', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying asmpyotic matching point:
#########################################################################################################
def matching_point_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the matching point used in STRIDE and RDCON calculation.
    """
    if quick_test:
        scan_vals = [1e-3,2e-3]
    else:
        scan_vals = tps.create_dense_log_paramvals(start=1e-1,end=1e-6,points_per_decade=2)
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('dx0', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying psilow truncation point:
#########################################################################################################
def psilow_truncation_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psilow truncation point used in STRIDE and RDCON calculation.
    """
    if quick_test:
        scan_vals = [1e-4,2e-4]
    else:
        scan_vals = tps.create_dense_log_paramvals(start=1e-1,end=1e-5,points_per_decade=2)
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('psilow', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying equilibrium integration tolerance:
#########################################################################################################
def equilibrium_integrator_tolerance_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the equilibrium integration tolerance used in STRIDE and RDCON calculation.
    """
    if quick_test:
        scan_vals = [1e-9,1e-10]
    else:
        scan_vals = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('etol', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying mtheta:
#########################################################################################################
def mtheta_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mtheta used in resistive calculation.
    """
    if quick_test:
        scan_vals = [129,130,257,513]
    else:
        scan_vals = [70,90,129,257,513,1025]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('mtheta', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying mtheta: no wall
#########################################################################################################
def mtheta_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mtheta used in the resistive calculation, with no wall.
    """
    if quick_test:
        scan_vals = [129,130,257,513]
    else:
        scan_vals = [70,90,129,257,513,1025]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('mtheta', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying mpsi:
#########################################################################################################
def mpsi_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mpsi used in the resistive calculation.
    """
    if quick_test:
        scan_vals = [257,258,513]
    else:
        scan_vals = [70,90,129,257,513,1025]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('mpsi', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying mpsi: no wall
#########################################################################################################
def mpsi_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the mpsi used in the resistive calculation, with no wall.
    """
    if quick_test:
        scan_vals = [257,258,513]
    else:
        scan_vals = [70,90,129,257,513,1025]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('mpsi', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying psihigh truncation within q surface:
#########################################################################################################
def edge_truncation_within_surface_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),psihigh=0.998,quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation.
    """
    if quick_test:
        scan_vals = [0.15,0.2]
    else:
        scan_vals = [0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        print("Warning: it's likely the q-surface being truncated will jump by one over the course of this scan. Read edge_truncation_within_surface_scan & fiddle with psihigh to stop this happening.")
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('dmlim', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying psihigh truncation within q surface: no wall
#########################################################################################################
def edge_truncation_within_surface_scan_no_wall(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation, with no wall.
    """
    if quick_test:
        scan_vals = [0.15,0.2]
    else:
        scan_vals = [0.00001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        print("Warning: it's likely the q-surface being truncated will jump by one over the course of this scan. Read edge_truncation_within_surface_scan_no_wall & fiddle with psihigh to stop this happening.")
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('dmlim', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying number of finite elements: RDCON
#########################################################################################################
def rdcon_finite_element_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the number of finite elements used in RDCON calculation.
    """
    if quick_test:
        scan_vals = [257,128]
    else:
        scan_vals = [60,70,80,90,100,120,257,512]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('nx', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
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
    if quick_test:
        scan_vals = ['''-k"100 50 80 140"''','''-k"90 46 70 130"''','''-k"70 30 50 100"''']
        var_name='nx_string_pest'
    else:
        scan_vals = [30,40,50,60,70,80,90,100,120,140,160,180,200,225,257,300,400,512]
        var_name='nx_pest'
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input(var_name, scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying integration tolerance near rational surfaces:
#########################################################################################################
def integrator_tolerance_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the integrator tolerance used in STRIDE and RDCON calculation.
    """
    if quick_test:
        scan_vals = [1e-9,1e-10]
    else:
        scan_vals = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('tol_r', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying magnetic coordinate type:
#########################################################################################################
def mag_coord_type_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the magnetic coordinate type used in STRIDE and RDCON calculation.
    """
    # Note: PEST3 doesn't have an option to vary jac_type, so we don't run it here.
    if quick_test:
        scan_vals = ["""'hamada'""","""'pest'"""]
    else:
        scan_vals = ["""'hamada'""","""'boozer'""","""'pest'""","""'equal_arc'"""]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('jac_type', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying grid_type:
#########################################################################################################
def grid_type_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the grid type used in STRIDE and RDCON calculation.
    """
    # Note: PEST3 doesn't have an option to vary grid_type, so we don't run it here.
    if quick_test:
        scan_vals = ["""'ldp'""","""'pow1'""","""'pow2'""","""'rho'"""]
    else:
        scan_vals = ["""'ldp'""","""'pow1'""","""'pow2'""","""'rho'""","""'original'"""] #original breaks...
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('grid_type', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying vacuum theta spline density:
#########################################################################################################
def vacuum_mtheta_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the amount of theta spline points for the vacuum-plasma interface.
    """
    if quick_test:
        scan_vals = [960,1200]
    else:
        scan_vals = [80,100,200,300,400,500,600,700,800,900,1000,1200,1500,2000]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('mthvac', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 

#EXTRA:
#########################################################################################################
# Varying cutoff: n = 1
#########################################################################################################
def RDCON_cutoff_scan_n1(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the cutoff value used in the resistive calculation in RDCON.
    """
    if quick_test:
        scan_vals = [9,10]
    else:
        scan_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('cutoff', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying cutoff: n = 4
#########################################################################################################
def RDCON_cutoff_scan_n4(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the cutoff value used in the resistive calculation in RDCON.
    """
    if quick_test:
        scan_vals = [9,10]
    else:
        scan_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('cutoff', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying solver: 
#########################################################################################################
def RDCON_solver_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),verbose=True,output_prefix='',quick_test=False, **kwargs):
    """
    Runs a 1D scan over the solver used in the resistive calculation in RDCON.
    """
    scan_vals = ["""'LU'""","""'cholesky'"""]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('solver', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# Varying axis_mid_pt_skew: 
#########################################################################################################
def STRIDE_axis_mid_pt_skew_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the axis_mid_pt_skew used in the resistive calculation in STRIDE.
    """
    if quick_test:
        scan_vals = [11.0,12.0,13.0]
    else:
        scan_vals = [2.0,4.0,8.0,10.0,12.0,14.0,18.0,20.0]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('axis_mid_pt_skew', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# large_sol_extent_pest
#########################################################################################################
def large_sol_extent_pest_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the large_sol_extent_pest used in the resistive calculation in PEST3.
    """
    if quick_test:
        scan_vals = [0.8,0.9]
    else:
        scan_vals = [0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('large_sol_extent_pest', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 

#EXTRA:
#########################################################################################################
# Varying nq: 
#########################################################################################################
def RDCON_nq_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the number of Gaussian points in each Galerkin element in the resistive calculation in RDCON.
    """
    if quick_test:
        scan_vals = [5,6]
    else:
        scan_vals = [1,2,3,4,5,6,7]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('nq', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# sing1_flag
#########################################################################################################
def sing1_flag_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over sing1_flag used in the resistive calculation in RDCON.
    """
    scan_vals = ['t','f']
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('sing1_flag', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# regrid_flag - probably broken
#########################################################################################################
def regrid_flag_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over regrid_flag used in the resistive calculation in RDCON.
    """
    scan_vals = ['t','f']
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('regrid_flag', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# RDCON/STRIDE inputs I'm ignoring:
#   pfac - varying nx in rdcon will determine whether we need more finite elements near the rational surfaces
#   sing_start - not really interested in cutting out the effect of various rational surfaces
#                if q_0 < 1, and the plasma is ideal unstable, then delta' at other surfaces doesn't mean much
#   crossover - effect will allow speedup by varying tol_r, tol_nr
#   ucrit - doesn't effect resistive calculations in stride or rdcon
#   nIntervalsTot - is autmatically increased by stride to minimally cover the number of singular intervals
#                   seems affect threading/speed
#########################################################################################################

#########################################################################################################
# PHYSICS PARAMETER DEPENDENCIES:
#########################################################################################################
#########################################################################################################
# varying q-surface truncation:
#########################################################################################################
def edge_truncation_q_scan(eq_filename,scan_vals=None,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation.
    """
    if scan_vals is None:
        if quick_test:
            scan_vals = [5.2,6.2,7.2]
        else:
            scan_vals = [2.2,3.2,4.2,5.2,6.2,7.2,8.2]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('qhigh', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 

# This function can pick up a discrepancy in truncation between rdcon and stride for the ideal test case equilibrium
# Needs fix to truncation logic in GPEC when both sasflag and qhigh are set - """qhigh overrode sasflag in RDCON but sasflag overrode qhigh in STRIDE"""
def edge_truncation_q_scan_DEPRECATED(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation.
    """
    if quick_test:
        scan_vals = [5,6,7]
    else:
        scan_vals = [2,3,4,5,6,7,8]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('qhigh', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# varying q-surface truncation: no wall
#########################################################################################################
def edge_truncation_q_scan_no_wall(eq_filename,scan_vals=None, results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation, with no wall.
    """
    if scan_vals is None:
        if quick_test:
            scan_vals = [5.2,6.2,7.2]
        else:
            scan_vals = [2.2,3.2,4.2,5.2,6.2,7.2,8.2]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('qhigh', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 

# Needs fix to truncation logic in GPEC when both sasflag and qhigh are set
def edge_truncation_q_scan_no_wall_DEPRECATED(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the psihigh truncation point used in the resistive calculation, with no wall.
    """
    if quick_test:
        scan_vals = [5,6,7]
    else:
        scan_vals = [2,3,4,5,6,7,8]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('qhigh', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 
#########################################################################################################
# varying wall radius:
#########################################################################################################
def wall_radius_scan(eq_filename,results_dir = os.path.join(home_dir, 'tests/test_results'),superquick=False, quick_test=True,verbose=True,output_prefix='', **kwargs):
    """
    Runs a 1D scan over the wall radius used in the resistive calculation.
    """
    if superquick:
        scan_vals = [0]
    elif quick_test:
        scan_vals = [0,0.1,0.2,21]
    else:
        scan_vals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,15,21]
    xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps =scan_1D_input('a_wall', scan_vals, eq_filename,
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
    return xarrays, pest3_xarrays, input_dicts, var_values, var_name, message, dps 


# Things that might vary for certain equilibria
key_numerical_tests = [
    'mtheta_scan',
    'mpsi_scan',
    'edge_truncation_within_surface_scan_no_wall',
    'pest3_finite_element_scan',
    'mag_coord_type_scan',
    'grid_type_scan',
    'vacuum_mtheta_scan',
    'RDCON_cutoff_scan_n4'
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
    'RDCON_solver_scan',
    'STRIDE_axis_mid_pt_skew_scan',
    'large_sol_extent_pest_scan',
    'RDCON_nq_scan',
    'sing1_flag_scan',
    'regrid_flag_scan',
    'edge_truncation_q_scan',
    'edge_truncation_q_scan_no_wall',
    'wall_radius_scan'
]

def run_multiple_scans(eq_filename, scan_namelist=scan_functions,  results_dir=os.path.join(home_dir, 'tests/test_results'), quick_test=True, verbose=True, debug=False, hotstart=''):
    """
    Runs multiple scan functions consecutively based on a list of scan names.
    
    Parameters:
    -----------
    scan_namelist : list
        List of scan function names to run
    eq_filename : str
        Path to equilibrium file
    results_dir : str
        Directory to store results
    quick_test : bool
        Whether to run quick tests or full scans
    verbose : bool
        Whether to print verbose output
    
    Returns:
    --------
    results : dict
        Dictionary with scan names as keys and results as values
    scan_namelist : list
        The input scan name list
    """
    # Dictionary mapping scan names to their corresponding functions
    scan_function_map = {
        'fourier_mode_scan': fourier_mode_scan,
        'sing_order_scan': sing_order_scan,
        'matching_point_scan': matching_point_scan,
        'psilow_truncation_scan': psilow_truncation_scan,
        'equilibrium_integrator_tolerance_scan': equilibrium_integrator_tolerance_scan,
        'mtheta_scan': mtheta_scan,
        'mtheta_scan_no_wall': mtheta_scan_no_wall,
        'mpsi_scan': mpsi_scan,
        'mpsi_scan_no_wall': mpsi_scan_no_wall,
        'edge_truncation_within_surface_scan': edge_truncation_within_surface_scan,
        'edge_truncation_within_surface_scan_no_wall': edge_truncation_within_surface_scan_no_wall,
        'rdcon_finite_element_scan': rdcon_finite_element_scan,
        'pest3_finite_element_scan': pest3_finite_element_scan,
        'integrator_tolerance_scan': integrator_tolerance_scan,
        'mag_coord_type_scan': mag_coord_type_scan,
        'grid_type_scan': grid_type_scan,
        'vacuum_mtheta_scan': vacuum_mtheta_scan,
        'RDCON_cutoff_scan_n1': RDCON_cutoff_scan_n1,
        'RDCON_cutoff_scan_n4': RDCON_cutoff_scan_n4,
        'RDCON_solver_scan': RDCON_solver_scan,
        'STRIDE_axis_mid_pt_skew_scan': STRIDE_axis_mid_pt_skew_scan,
        'large_sol_extent_pest_scan': large_sol_extent_pest_scan,
        'RDCON_nq_scan': RDCON_nq_scan,
        'sing1_flag_scan': sing1_flag_scan,
        'regrid_flag_scan': regrid_flag_scan,
        'edge_truncation_q_scan': edge_truncation_q_scan,
        'edge_truncation_q_scan_no_wall': edge_truncation_q_scan_no_wall,
        'wall_radius_scan': wall_radius_scan
    }
    if len(hotstart) > 0:
        # Check if hotstart is in scan_namelist
        if hotstart not in scan_namelist:
            print(f"Warning: '{hotstart}' is not in the scan_namelist. Running through all scans.")
        else:
            # Cut all entries before hotstart
            hotstart_index = scan_namelist.index(hotstart)
            scan_namelist = scan_namelist[hotstart_index:]
            print(f"Running scans starting from '{hotstart}' and onwards.")
    results = {}
    for scan_name in scan_namelist:
        if scan_name not in scan_function_map:
            print(f"Warning: Scan function '{scan_name}' not found. Skipping.")
            continue
            
        scan_func = scan_function_map[scan_name]
        messages = []
        failed_cases = []
        try:
            results[scan_name] = scan_func(eq_filename, results_dir=results_dir, quick_test=quick_test, verbose=True)
            messages.append(results[scan_name][-1])
            if verbose:
                print("#########################################################################################################")
                print("#########################################################################################################")
                print("#########################################################################################################")
                print(f"Completed scan: {scan_name} for equilibrium {eq_filename.split('/')[-1]}")
                print(f"Results summary: {len(results[scan_name][0])} data points")
                if not hotstart:
                    print(f"Progress through test set: {scan_namelist.index(scan_name) + 1}/{len(scan_namelist)}")
                else:
                    print(f"Progress through test set: {scan_namelist.index(scan_name) + 1}/{len(scan_namelist)} (hotstart skipped first {hotstart_index} scans)")
                print(f"List of failed cases: {failed_cases}")
                print("#########################################################################################################")
                print("#########################################################################################################")
                print("#########################################################################################################")
        except Exception as e:
            print("#########################################################################################################")
            print("#########################################################################################################")
            print("!!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ")
            print(f"Error running scan '{scan_name}': {str(e)}")
            results[scan_name] = None
            failed_cases.append(scan_name)
            print("!!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ERROR !!! !!! !!! ")
            print("#########################################################################################################")
            print("#########################################################################################################")
            if debug:
                raise e
    if verbose:
        for i in messages:
            print("#########################################################################################################")
            print("#########################################################################################################")
            print(i)
            print("#########################################################################################################")
            print("#########################################################################################################")
        print(" Failed cases:")
        for i in failed_cases:
            print(f" - {i}")
    return results, scan_namelist
