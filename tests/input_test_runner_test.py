# Scripts to test the functions in input_test_runner.py

import os 

from tearing_physics_suite.environment import home_dir
from tearing_physics_suite.input_test_suite import *
from tearing_physics_suite.input_test_runner import run_multiple_scans, run_multiple_scans_parallel

#########################################################################################################
# load equilibrium:
#########################################################################################################

# Choose equilibrium file
eq_filename = os.path.join(home_dir, 'submodules/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT')

print(" Getting equilibrium file from ", eq_filename)
eq_filename_short= eq_filename.split('/')[-1]

run_single_test = True
run_short_test = True
run_all_tests = False
run_parallel_tests = False

if __name__ == '__main__':
    #########################################################################################################
    # Run single case:
    #########################################################################################################
    if run_single_test:
        restarray,message = edge_truncation_q_scan(eq_filename, run_stride=True,run_pest3=True, results_dir=os.path.join(home_dir, 'tests/test_results/rdcon_finite_element_scan'), verbose=True)
        #restarray,message = rdcon_finite_element_scan(eq_filename, results_dir=os.path.join(home_dir, 'tests/test_results/rdcon_finite_element_scan'), verbose=True, ode_flag='f')
    #########################################################################################################
    # Run scans:
    #########################################################################################################
    if run_short_test:
        results, messages, scan_namelist, failed_cases = run_multiple_scans(eq_filename, scan_namelist=physics_tests, results_dir=os.path.join(home_dir, 'tests/test_results/physics_tests'), quick_test=True, verbose=True,debug=True)
    if run_all_tests:
        results, messages, scan_namelist, failed_cases = run_multiple_scans(eq_filename, scan_namelist=key_numerical_tests, results_dir=os.path.join(home_dir, 'tests/test_results/key_numerical_tests'), quick_test=True, verbose=True,debug=True)
        results, messages, scan_namelist, failed_cases = run_multiple_scans(eq_filename, scan_namelist=scan_functions, results_dir=os.path.join(home_dir, 'tests/test_results/input_scans'), quick_test=True, verbose=True,debug=True)
    #########################################################################################################
    # Run parallel scans:
    #########################################################################################################
    if run_parallel_tests:
        results, messages, scan_namelist, failed_cases = run_multiple_scans_parallel(eq_filename, os.path.join(home_dir, 'tests/test_results/parallel_input_scans'), scan_namelist=physics_tests, quick_test=True, verbose=True, debug=True) 