# Python functions to call GPEC and PEST3 fortran codes for delta prime calculations

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
import unittest

from tearing_physics_suite.environment import home_dir

test_rdcon=True
test_stride=True
test_pest3=True

os.chdir(home_dir)

#########################################################################################################
# Defining test paths:
#########################################################################################################
rdcon_test_path_1 = os.path.join(home_dir, 'submodules/GPEC/docs/examples/DIIID_resistive_example')
rdcon_test_path_2 = os.path.join(home_dir, 'submodules/GPEC/docs/examples/solovev_resistive_example')
rdcon_test_path_3 = os.path.join(home_dir, 'submodules/GPEC/docs/examples/a5_tearing_example')
stride_test_path_4 = os.path.join(home_dir, 'submodules/GPEC/docs/examples/DIIID_ideal_example')
stride_test_path_1 = rdcon_test_path_1
stride_test_path_2 = rdcon_test_path_2
stride_test_path_3 = rdcon_test_path_3
pest_test_path_1 = os.path.join(home_dir, 'submodules/PEST3/examples')

print("Here are the rdcon test paths:")
print("cd ",rdcon_test_path_1)
print("cd ",rdcon_test_path_2)
print("cd ",rdcon_test_path_3)  
print("Here are the stride test paths:")
print("cd ",stride_test_path_1)
print("cd ",stride_test_path_2)
print("cd ",stride_test_path_3)
print("cd ",stride_test_path_4)
print("Here is the pest test path:")
print("cd ",pest_test_path_1)


#########################################################################################################
# Copying executables to test directories:
#########################################################################################################
#copy the rdcon files to each test directory
print("Copying rdcon to test directories...")
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/rdcon/rdcon'), rdcon_test_path_1)
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/rdcon/rdcon'), rdcon_test_path_2)
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/rdcon/rdcon'), rdcon_test_path_3)
#copy the stride files to each test directory
print("Copying stride to test directories...")
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/stride/stride'), stride_test_path_1)
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/stride/stride'), stride_test_path_2)
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/stride/stride'), stride_test_path_3)
shutil.copy(os.path.join(home_dir, 'submodules/GPEC/stride/stride'), stride_test_path_4)
print("Copying pest3x to test directories...")
shutil.copy(os.path.join(home_dir, 'submodules/PEST3/build/pest3/pest3x'), pest_test_path_1) 
#########################################################################################################
# Running rdcon tests:
#########################################################################################################
os.chdir(rdcon_test_path_1)
print("Running rdcon test 1 in directory:", os.getcwd())
rdcon_test1=subprocess.run('./rdcon', check=False)
os.chdir(rdcon_test_path_2)
print("Running rdcon test 2 in directory:", os.getcwd())
rdcon_test2=subprocess.run('./rdcon', check=False)
os.chdir(rdcon_test_path_3)
print("Running rdcon test 3 in directory:", os.getcwd())
rdcon_test3=subprocess.run('./rdcon', check=False)
os.chdir(os.path.join(home_dir, 'submodules/GPEC/docs'))

rdcon_tests_ran = rdcon_test1.returncode == 0 and rdcon_test2.returncode == 0 and rdcon_test3.returncode == 0

#########################################################################################################
# Reading rdcon results:
#########################################################################################################
os.chdir(rdcon_test_path_1)
rdcon_results_xr_1 = xr.open_dataset('rdcon_output_n1.nc')
os.chdir(rdcon_test_path_2)
rdcon_results_xr_2 = xr.open_dataset('rdcon_output_n1.nc')
os.chdir(rdcon_test_path_3)
rdcon_results_xr_3 = xr.open_dataset('rdcon_output_n1.nc')

# Testing reasonable values:
rdcon_valtest1 = (abs(rdcon_results_xr_1['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-2.0) < 0.1)
rdcon_valtest2 = (abs(rdcon_results_xr_2['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-4.567634e+09) < 1e+08)
rdcon_valtest3 = (abs(rdcon_results_xr_3['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-13.2) < 0.3)

rdcon_valtests_passed = rdcon_valtest1 and rdcon_valtest2 and rdcon_valtest3

#########################################################################################################
# Running stride tests:
#########################################################################################################
os.chdir(stride_test_path_1)
stride_test_1=subprocess.run('./stride', check=True)
os.chdir(stride_test_path_2)
stride_test_2=subprocess.run('./stride', check=True)
os.chdir(stride_test_path_3)
stride_test_3=subprocess.run('./stride', check=True)
os.chdir(stride_test_path_4)
stride_test_4=subprocess.run('./stride', check=True)

stride_tests_ran = stride_test_1.returncode == 0 and stride_test_2.returncode == 0 and stride_test_3.returncode == 0 and stride_test_4.returncode == 0

#########################################################################################################
# Reading stride results:
#########################################################################################################
os.chdir(stride_test_path_1)
if stride_test_1.returncode == 0:
    try:
        stride_results_xr_1 = xr.open_dataset('stride_output_n1.nc')
        stride_valtest1= True
    except:
        stride_valtest1= False


os.chdir(stride_test_path_2)
if stride_test_2.returncode == 0:
    try:
        stride_results_xr_2 = xr.open_dataset('stride_output_n1.nc')
        stride_valtest2= True
    except:
        stride_valtest2= False


os.chdir(stride_test_path_3)
if stride_test_3.returncode == 0:
    try:
        stride_results_xr_3 = xr.open_dataset('stride_output_n1.nc')
        stride_valtest3= True
    except:
        stride_valtest3= False


os.chdir(stride_test_path_4)
if stride_test_4.returncode == 0:
    try:
        stride_results_xr_4 = xr.open_dataset('stride_output_n1.nc')
        stride_valtest4= True
    except:
        stride_valtest4= False


# Testing reasonable values:
if stride_valtest1:
    stride_valtest1 = (abs(stride_results_xr_1['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-8.0) < 0.1)


if stride_valtest2:
    stride_valtest2 = (abs(stride_results_xr_2['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-6.4e+12) < 1e+11)


if stride_valtest3:
    stride_valtest3 = (abs(stride_results_xr_3['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-13.2) < 0.3)


if stride_valtest4:
    stride_valtest4 = (abs(stride_results_xr_4['Delta_prime'].sel(i=0, r=2.0,r_prime=2.0).values-8.0) < 0.1)


stride_all_valtests_passed = stride_valtest1 and stride_valtest2 and stride_valtest3 and stride_valtest4

#########################################################################################################
# Running pest tests:
#########################################################################################################

os.chdir(pest_test_path_1)
print("Running pest3 tests in directory:", os.getcwd())

test1_pest_input_string = ''' ./pest3x -i2 -feqdsk.cdf -k"70 100 140 200" -l10'''
print("Input terminal:", test1_pest_input_string)
ran1=os.system(test1_pest_input_string)
if ran1==0:
    try:
        pest_results_xr_1 = xr.open_dataset('pest3.nc')
    except:
        pest_results_xr_1 = None


test2_pest_input_string = ''' ./pest3x -i3 -fgeqdsk -k"70 100 140 200" -l10'''
print("Input terminal:", test2_pest_input_string)
ran2=os.system(test2_pest_input_string)
if ran2==0:
    try:
        pest_results_xr_2 = xr.open_dataset('pest3.nc')
    except:
        pest_results_xr_2 = None


test3_pest_input_string = ''' ./pest3x -i1 -finp1.cdf -k"70 100 140 200" -l10'''
print("Input terminal:", test3_pest_input_string)
ran3=os.system(test3_pest_input_string)
if ran3==0:
    try:
        pest_results_xr_3 = xr.open_dataset('pest3.nc')
    except:
        pest_results_xr_3 = None


pest_tests_ran = ran1 == 0 and ran2 == 0 and ran3 == 0

#########################################################################################################
# Reading pest results:
#########################################################################################################
pest_valtest1 = False
pest_valtest2 = False
pest_valtest3 = False


if pest_results_xr_1 is not None:
    pest_valtest1 = (abs(pest_results_xr_1['dprim_re'].values-2.43) < 0.1)


if pest_results_xr_2 is not None:
    pest_valtest2 = (abs(pest_results_xr_2['dprim_re'].values+0.17) < 0.01)


if pest_results_xr_3 is not None:
    pest_valtest3 = (abs(pest_results_xr_3['dprim_re'].values+0.00356) < 0.001)


pest_all_valtests_passed = pest_valtest1 and pest_valtest2 and pest_valtest3

#########################################################################################################
# Summary of test results:
#########################################################################################################

if True:
    print("\nSummary of test results:")
    print("rdcon tests ran:", rdcon_tests_ran)
    print("rdcon all value tests passed:", rdcon_valtests_passed)
    print("stride tests ran:", stride_tests_ran)
    print("stride value tests passed:", stride_valtest1, stride_valtest2, stride_valtest3, stride_valtest4)
    print("pest tests ran:", pest_tests_ran)
    print("pest value tests passed:", pest_valtest1, pest_valtest2, pest_valtest3)