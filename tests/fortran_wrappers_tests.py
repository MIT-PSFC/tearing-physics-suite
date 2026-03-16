# Scripts to test the functions in fortran_wrappers.py, GPEC_write_inputs.py, and PEST3_wrappers.py

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
import pickle as pkl

import tearing_physics_suite
from tearing_physics_suite.environment import home_dir
import tearing_physics_suite.fortran_wrappers as tfw

os.chdir(home_dir)

working_dir = os.path.join(home_dir, 'tests/test_working_dir')
results_dir = os.path.join(home_dir, 'tests/test_results')

# Choose equilibrium file
default_equilibrium =  os.path.join(home_dir, 'submodules/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT')
eq_filename = default_equilibrium

#Options:
pest3_truncation_test=True
multi_wrapper_test1=True
read_write_test=True
multi_wrapper_test2=True

print(" Getting equilibrium file from ", default_equilibrium)
eq_filename_short= eq_filename.split('/')[-1]

#########################################################################################################
# PEST3 truncation test:
#########################################################################################################
    
if pest3_truncation_test:
    qlim_actual=5.2
    nn=1
    pest3_kwargs_dict={'psihigh_pest':1}
    output_prefix_special='trunctest1'
    psi_trunc_frac, pest3_trunc_ran = tfw.pest3_special_truncation_loop(eq_filename, nn, qlim_actual, pest3_kwargs_dict, debug=True, output_prefix_special=output_prefix_special)
    print(f"PEST3 truncation test: psi_trunc_frac = {psi_trunc_frac}, pest3_trunc_ran = {pest3_trunc_ran}")
    pest3_truncation_test_result=pest3_trunc_ran
    print("PEST3 truncation test result:", pest3_truncation_test_result)

#########################################################################################################
# Multi-wrapper test: ideal wall calculation
#########################################################################################################
output_prefix='test_1'
nn=1
if multi_wrapper_test1:
    rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict = tfw.run_resistive_calculation(
        eq_filename,
        nn,
        working_dir=working_dir,
        run_rdcon=True,
        run_stride=True,
        etol=1e-10,
        mpsi_pest=400,
        kband_pest=15,
        run_pest3=True,
        vac_flag='f',
        gal_flag='t',
        ode_flag='t',
        output_location=results_dir,
        output_prefix=output_prefix,
        pest_match_truncation=True,
        save_input=True,
        nx_string_pest='''-k"100 70 140 50 120"''',
        verbose=True
    )
#########################################################################################################
# Checking values
#########################################################################################################
    RDCON_DP21_RE = rdcon_xr.Delta_prime.isel(r=0,r_prime=0,i=0)
    PEST3_DP21_RE = pest3_xr.Delta_prime.isel(r=0,r_prime=0,i=0)
    STRIDE_DP21_RE = stride_xr.Delta_prime.isel(r=0,r_prime=0,i=0)
    #test that they're near 2:
    RDCON_DP21_val_good = np.isclose(RDCON_DP21_RE, 2.0, atol=0.05)
    PEST3_DP21_val_good = np.isclose(PEST3_DP21_RE, 2.1, atol=0.15)
    STRIDE_DP21_val_good = np.isclose(STRIDE_DP21_RE, 2.0, atol=0.05)
    allvals_good = RDCON_DP21_val_good and PEST3_DP21_val_good and STRIDE_DP21_val_good
    print("RDCON Delta prime 21 real part:", RDCON_DP21_RE)
    print("PEST3 Delta prime 21 real part:", PEST3_DP21_RE)
    print("STRIDE Delta prime 21 real part:", STRIDE_DP21_RE)
    print("RDCON Delta prime 21 value good:", RDCON_DP21_val_good)
    print("PEST3 Delta prime 21 value good:", PEST3_DP21_val_good)
    print("STRIDE Delta prime 21 value good:", STRIDE_DP21_val_good)
#########################################################################################################
# Checking for read-write success
#########################################################################################################
    if read_write_test:
        #Read the results
        print("Reading results from multi-wrapper test...")
        rdcon_xr_read = xr.open_dataset(os.path.join(results_dir, output_prefix + eq_filename_short +  '_rdcon_n' + str(nn) + '.nc'))
        stride_xr_read = xr.open_dataset(os.path.join(results_dir, output_prefix + eq_filename_short +  '_stride_n' + str(nn) + '.nc'))
        pest3_xr_read = xr.open_dataset(os.path.join(results_dir, output_prefix + eq_filename_short +  '_pest3_n' + str(nn) + '.nc'))
        fpkl = open(os.path.join(results_dir, output_prefix + eq_filename_short + '_rdcon_stride_input_n'+str(nn)+'.pkl'),"rb")
        rdcon_stride_input_dict_read = pd.read_pickle(fpkl)
        fpkl.close()
        fpkl = open(os.path.join(results_dir, output_prefix + eq_filename_short + '_pest3_input_n'+str(nn)+'.pkl'),"rb")
        pest3_input_dict_read = pd.read_pickle(fpkl)
        fpkl.close()
        #Check they're the same
        rdcon_xr_equal = rdcon_xr.equals(rdcon_xr_read)
        stride_xr_equal = stride_xr.equals(stride_xr_read)
        pest3_xr_equal = pest3_xr.equals(pest3_xr_read)
        #Check the input dictionaries
        rdcon_stride_input_dict_equal = rdcon_stride_input_dict == rdcon_stride_input_dict_read
        pest3_input_dict_equal = pest3_input_dict == pest3_input_dict_read
        #Reading and writing success:
        read_write_success = rdcon_xr_equal and stride_xr_equal and pest3_xr_equal and rdcon_stride_input_dict_equal and pest3_input_dict_equal
        print("Read-write success:", read_write_success)
        if not read_write_success:
            print("There was an error in reading or writing the results.")
            print("rdcon_xr read write test:", rdcon_xr_equal)
            print("stride_xr read write test:", stride_xr_equal)
            print("pest3_xr read write test:", pest3_xr_equal)
            print("rdcon_stride_input_dict read write test:", rdcon_stride_input_dict_equal)
            print("pest3_input_dict read write test:", pest3_input_dict_equal)

#########################################################################################################
# Multi-wrapper test: no wall calculation
#########################################################################################################

output_prefix='test_2'
nn=1
if multi_wrapper_test2:
    rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict = tfw.run_resistive_calculation(
        eq_filename,
        nn,
        working_dir=working_dir,
        run_rdcon=True,
        run_stride=True,
        etol=1e-10,
        mpsi_pest=400,
        kband_pest=15,
        run_pest3=True,
        vac_flag='t',
        gal_flag='t',
        ode_flag='t',
        output_location=results_dir,
        output_prefix=output_prefix,
        save_input=True,
        nx_string_pest='''-k"100 70 140 50 120"''',
        verbose=True
    )
#########################################################################################################
# Checking values
#########################################################################################################
    RDCON_DP21_RE = rdcon_xr.Delta_prime.isel(r=0,r_prime=0,i=0)
    PEST3_DP21_RE = pest3_xr.Delta_prime.isel(r=0,r_prime=0,i=0)
    STRIDE_DP21_RE = stride_xr.Delta_prime.isel(r=0,r_prime=0,i=0)
    #test that they're near 2:

    
if multi_wrapper_test2:
    RDCON_DP21_wall_val_good = np.isclose(RDCON_DP21_RE, 7.3, atol=0.03)
    PEST3_DP21_wall_val_good = np.isclose(PEST3_DP21_RE, 7.3, atol=0.03)
    STRIDE_DP21_wall_val_good = np.isclose(STRIDE_DP21_RE, 7.3, atol=0.3)
    print("RDCON Delta prime 21 wall real part:", RDCON_DP21_RE)
    print("PEST3 Delta prime 21 wall real part:", PEST3_DP21_RE)
    print("STRIDE Delta prime 21 wall real part:", STRIDE_DP21_RE)
#########################################################################################################
# Summarize all results again:
#########################################################################################################

if True:
    if pest3_truncation_test:
        print("PEST3 truncation test result:", pest3_truncation_test_result)
    if read_write_test:
        print("Read-write success:", read_write_success)
        print("Save terminal output to file is working =", False) #Debug this capability later
    if multi_wrapper_test1:
        print("RDCON Delta prime 21 internal mode value correct:", RDCON_DP21_val_good)
        print("PEST3 Delta prime 21 internal mode value correct:", PEST3_DP21_val_good)
        print("STRIDE Delta prime 21 internal mode value correct:", STRIDE_DP21_val_good)
    if multi_wrapper_test2: 
        print("RDCON Delta prime 21 no-wall value correct:", RDCON_DP21_wall_val_good)
        print("PEST3 Delta prime 21 no-wall value correct:", PEST3_DP21_wall_val_good)
        print("STRIDE Delta prime 21 no-wall value correct:", STRIDE_DP21_wall_val_good)

