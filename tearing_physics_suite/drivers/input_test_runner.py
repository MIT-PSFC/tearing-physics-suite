# Default combinations of input scans from input_test_suite.py, to test sensitivity of Delta' calculations to various input parameters.

import multiprocessing
import os
import pickle as pkl

from tearing_physics_suite.drivers.input_test_suite import SCANS, run_scan, scan_functions
from tearing_physics_suite.utils import _get_num_cpus, tps_home


def run_multiple_scans(eq_filename, scan_namelist=scan_functions,  results_dir=None, quick_test=True, verbose=True, debug=False, hotstart=''):
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
    if results_dir is None:
        results_dir = os.path.join(tps_home(), 'tests/test_results')
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
    messages = []
    failed_cases = []

    for scan_name in scan_namelist:
        if scan_name not in SCANS:
            print(f"Warning: Scan function '{scan_name}' not found. Skipping.")
            continue

        try:
            result, message = run_scan(scan_name, eq_filename, results_dir=results_dir, quick_test=quick_test, verbose=True)
            messages.append(message)
            results[scan_name] = result
            if verbose:
                print("#########################################################################################################")
                print("#########################################################################################################")
                print("#########################################################################################################")
                print(f"Completed scan: {scan_name} for equilibrium {eq_filename.split('/')[-1]}")
                print(f"Results summary: {len(results[scan_name]['xarrays'])} data points")
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
    return results, messages, scan_namelist, failed_cases

def run_multiple_scans_parallel(eq_filename, master_working_dir, scan_namelist=scan_functions,
        results_dir=None,
        quick_test=True, verbose=True, debug=False, **kwargs):
    """Distribute all scans in scan_namelist across available CPU cores and run in parallel.

    Each CPU core runs one scan at a time from its own private working directory
    inside master_working_dir. Results are pickled per scan and collected once
    all workers finish.

    Parameters
    ----------
    eq_filename : str
        Path to equilibrium file.
    master_working_dir : str
        Top-level directory for worker subdirectories and pickled results.
    scan_namelist : list of str
        Scan function names to run (default: all scan_functions).
    results_dir : str
        Directory for scan output files (passed through to each scan function).
    quick_test : bool
        If True, run quick (short) versions of each scan.
    verbose : bool
        If True, print per-scan progress and final summary.
    debug : bool
        If True, re-raise exceptions from failed scans.
    **kwargs
        Forwarded to each scan function (e.g. custom scan_vals).

    Returns
    -------
    results : dict
        Scan name -> result dict from extract_scanned_xrs (None for failures).
    messages : list of str
        Summary messages from each scan.
    scan_namelist : list of str
        The (possibly filtered) list of scans that were attempted.
    failed_cases : list of str
        Names of scans that raised exceptions.
    """
    if results_dir is None:
        results_dir = os.path.join(tps_home(), 'tests/test_results')
    valid_scans = [(i, name) for i, name in enumerate(scan_namelist) if name in SCANS]
    skipped = [name for name in scan_namelist if name not in SCANS]
    if skipped:
        print(f"[run_multiple_scans_parallel] Skipping unknown scan names: {skipped}")

    os.makedirs(master_working_dir, exist_ok=True)

    n_cpus = _get_num_cpus()
    n_scans = len(valid_scans)
    n_workers = min(n_cpus, n_scans)
    print(f"[run_multiple_scans_parallel] Distributing {n_scans} scans across {n_workers} workers ({n_cpus} CPUs available).")

    # Distribute scans round-robin across workers
    worker_batches = [[] for _ in range(n_workers)]
    for i, (idx, name) in enumerate(valid_scans):
        worker_batches[i % n_workers].append((idx, name))

    worker_args = [
        (w, worker_batches[w], os.path.join(master_working_dir, f'worker_{w}'),
         master_working_dir, eq_filename, results_dir, quick_test, verbose, debug, kwargs)
        for w in range(n_workers)
    ]

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        all_worker_results = pool.map(_scan_worker_batch, worker_args)

    # Collect results in original scan_namelist order
    results = {}
    messages = []
    failed_cases = []

    for worker_results in all_worker_results:
        for scan_name, success, err_msg in worker_results:
            if success:
                result_path = os.path.join(master_working_dir, f'scan_result_{scan_name}.pkl')
                with open(result_path, 'rb') as f:
                    data = pkl.load(f)
                results[scan_name] = data['result']
                messages.append(data['message'])
            else:
                results[scan_name] = None
                failed_cases.append(scan_name)
                print(f"[run_multiple_scans_parallel] WARNING: Scan '{scan_name}' failed: {err_msg}")

    if verbose:
        for msg in messages:
            print("#" * 105)
            print(msg)
            print("#" * 105)
        print(" Failed cases:")
        for fc in failed_cases:
            print(f" - {fc}")

    return results, messages, [name for _, name in valid_scans], failed_cases


def _scan_worker_batch(args):
    """Worker function: sequentially runs a batch of scan functions in a private working directory.

    Results for each scan are pickled to master_working_dir/scan_result_{name}.pkl.
    """
    worker_id, batch, working_dir, master_working_dir, eq_filename, results_dir, quick_test, verbose, debug, kwargs = args

    # Restrict this process to a single thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    os.makedirs(working_dir, exist_ok=True)

    run_results = []
    for _idx, scan_name in batch:
        try:
            result, message = run_scan(
                scan_name,
                eq_filename,
                results_dir=results_dir,
                quick_test=quick_test,
                verbose=verbose,
                working_dir=working_dir,
                **kwargs
            )
            result_path = os.path.join(master_working_dir, f'scan_result_{scan_name}.pkl')
            with open(result_path, 'wb') as f:
                pkl.dump({'result': result, 'message': message, 'scan_name': scan_name}, f)
            print(f"[run_multiple_scans_parallel] Worker {worker_id}: '{scan_name}' completed.")
            run_results.append((scan_name, True, None))
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"[run_multiple_scans_parallel] Worker {worker_id}: '{scan_name}' failed: {e}\n{tb_str}")
            if debug:
                raise RuntimeError(f"Worker {worker_id}: scan '{scan_name}' failed") from e
            run_results.append((scan_name, False, f"{e}\n{tb_str}"))

    return run_results
