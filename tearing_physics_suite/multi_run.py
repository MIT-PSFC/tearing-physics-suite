# Scripts to parallelise tearing_physics_suite.py

### Structure:
## For a given type of input file, create a function that can read the input and generate the necessary splines and input dict for the main function in mre_analysis.py,
## Pass a list of equilibrium file names, and corresponding profile file names, as well as that function, to the multi_run function. It then actively
## distributes the runs across available CPU cores, and collects the outputs into a list of xarrays and a list of input dicts.

import os
import sys
import glob
import multiprocessing
import pickle as pkl
import xarray as xr
import numpy as np

from tearing_physics_suite.tearing_physics_suite import nonlinear_resistive_calculation


def multi_run(eq_filenames, profile_filenames, read_profile_function, master_working_dir, verbose=False, cluster_manager='slurm', return_lists=False, fail_fast=False, **kwargs):
    """Run nonlinear resistive calculations in parallel across equilibria and profiles.

    Reads profiles using read_profile_function, then delegates to multi_run_.
    Each CPU core processes its assigned equilibria sequentially in a single
    reusable working directory.

    Parameters
    ----------
    eq_filenames : list of str
        Paths to equilibrium files.
    profile_filenames : str or list of str
        Profile file path(s). A single string is applied to all equilibria.
    read_profile_function : callable
        Function(filename, verbose) -> list of profile dicts.
    master_working_dir : str
        Top-level directory for worker subdirectories and results.
    **kwargs
        Forwarded to nonlinear_resistive_calculation.

    Returns
    -------
    combined_xr_list : list of xr.Dataset or None
        One dataset per equilibrium (None entries for failures).
    input_dict_list : list of dict or None
        Input parameters per equilibrium.
    errors : dict
        Mapping of index -> error message for failed runs.
    """

    # Check if profile_filenames is a list of filenames, or just a single filename to be used for all equilibria.
    if isinstance(profile_filenames, str):
        profile_list = read_profile_function(profile_filenames, verbose=verbose)
        assert len(profile_list) == len(eq_filenames), "read_profile_function must return a list of profile dicts with the same length as eq_filenames."
    elif len(profile_filenames) == 1:
        profile_list = read_profile_function(profile_filenames[0], verbose=verbose)
        assert len(profile_list) == len(eq_filenames), "read_profile_function must return a list of profile dicts with the same length as eq_filenames."
    else: 
        assert len(profile_filenames) == len(eq_filenames), "If profile_filenames is a list, it must have the same length as eq_filenames."
        profile_list = [read_profile_function(profile_filenames[i], verbose=verbose) for i in range(len(profile_filenames))]

    return multi_run_(eq_filenames, profile_list, master_working_dir, verbose=verbose, cluster_manager=cluster_manager, return_lists=return_lists, fail_fast=fail_fast, **kwargs)

def multi_run_(eq_filenames, profile_list, master_working_dir, verbose=False, cluster_manager='slurm', return_lists=False, fail_fast=False, **kwargs):
    """Inner parallelisation driver: distributes equilibria across CPU workers.

    Each worker gets a private working directory and processes its batch sequentially.
    Results are pickled to master_working_dir/result_{idx}.pkl.

    Parameters
    ----------
    eq_filenames : list of str
        Paths to equilibrium files.
    profile_list : list of dict
        Pre-loaded profile dicts (one per equilibrium).
    master_working_dir : str
        Top-level directory for worker subdirectories and results.
    return_lists : bool
        If True, load and return result datasets. If False, results stay on disk.
    fail_fast : bool
        If True, abort remaining work after first failure.
    **kwargs
        Forwarded to nonlinear_resistive_calculation.

    Returns
    -------
    combined_xr_list : list of xr.Dataset or None
    input_dict_list : list of dict or None
    errors : dict
    """

    assert len(eq_filenames) == len(profile_list), "Length of eq_filenames and profile_list must be the same."

    # Parallel loop starts here:
    os.makedirs(master_working_dir, exist_ok=True)

    n_cpus = _get_num_cpus()
    n_runs = len(eq_filenames)
    n_workers = min(n_cpus, n_runs)
    print(f"[multi_run] Distributing {n_runs} runs across {n_workers} workers ({n_cpus} CPUs available via SLURM).")

    # Distribute runs across workers in round-robin fashion
    worker_batches = [[] for _ in range(n_workers)]
    for i in range(n_runs):
        worker_batches[i % n_workers].append((i, eq_filenames[i], profile_list[i]))

    # Build argument tuples — one working directory per worker
    worker_args = []
    for w in range(n_workers):
        worker_working_dir = os.path.join(master_working_dir, f'worker_{w}')
        worker_args.append((w, worker_batches[w], worker_working_dir, master_working_dir, kwargs, fail_fast))

    # Use 'spawn' to avoid fork-safety issues with Fortran subprocesses
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        all_results = pool.map(_worker_batch, worker_args)

    # Flatten results and collect from disk in original order
    combined_xr_list = [None] * n_runs
    input_dict_list = [None] * n_runs
    errors = {}

    for worker_results in all_results:
        for idx, success, err_msg in worker_results:
            if success and return_lists:
                result_path = os.path.join(master_working_dir, f'result_{idx}.pkl')
                with open(result_path, 'rb') as f:
                    data = pkl.load(f)
                combined_xr_list[idx] = data['combined_xr']
                input_dict_list[idx] = data['input_dict_out']
            else:
                errors[idx] = err_msg
                print(f"[multi_run] WARNING: Run {idx} ({eq_filenames[idx]}) failed: {err_msg}")

    n_success = sum(1 for x in combined_xr_list if x is not None)
    print(f"[multi_run] Completed: {n_success}/{n_runs} runs succeeded.")

    if errors:
        print(f"[multi_run] Failed runs: {sorted(errors.keys())}")
        # We save errors to a file for later inspection
        error_path = os.path.join(master_working_dir, 'errors.pkl')
        with open(error_path, 'wb') as f:
            pkl.dump(errors, f)
        print(f"[multi_run] Saved error details to {error_path}")

    if return_lists:
        return combined_xr_list, input_dict_list, errors
    else:
        return None, None, errors


def multi_compile(eq_filenames, master_working_dir, shot_time_list=None, debug=False):
    """Compile per-equilibrium results from multi_run into a single xarray dataset.

    Reads pickled result files from master_working_dir and concatenates them
    along a new 'eq' dimension.

    Parameters
    ----------
    eq_filenames : list of str
        Equilibrium filenames (used as coordinate labels).
    master_working_dir : str
        Directory containing result_{idx}.pkl files from multi_run.
    shot_time_list : list or None
        Optional shot/time labels for the eq dimension.
    debug : bool
        Return intermediate data for debugging.

    Returns
    -------
    compiled_xr : xr.Dataset
        Combined dataset across all equilibria.
    compiled_inputs_xr : xr.Dataset
        Combined input parameters.
    """
    n_runs = len(eq_filenames)

    # Read errors from multi_run to determine which runs succeeded
    if os.path.exists(os.path.join(master_working_dir, 'errors.pkl')):
        with open(os.path.join(master_working_dir, 'errors.pkl'), 'rb') as f:
            errors = pkl.load(f)
        num_errors = sum(1 for v in errors.values() if v is not None)
        print(f"[multi_compile] Found error log with {num_errors} failed runs.")
        if debug:
            for idx, err_msg in sorted(errors.items()):
                print(f"[multi_compile] Run {idx} error: {err_msg}")
    else:
        errors = {}
        print(f"[multi_compile] No error log found. Assuming all {n_runs} runs succeeded.")

    # Load all results from disk
    combined_xr_list = []
    input_dict_list = []
    eq_labels = []

    for i in range(n_runs):
        result_path = os.path.join(master_working_dir, f'result_{i}.pkl')
        xarray_path = os.path.join(master_working_dir, f'combined_xr_{i}.nc')
        if not os.path.exists(result_path):
            print(f"[multi_compile] WARNING: Result file not found for run {i}, skipping.")
            continue
        with open(result_path, 'rb') as f:
            data = pkl.load(f)
        if data['combined_xr'] is None:
            print(f"[multi_compile] WARNING: Run {i} has no combined_xr (failed run), skipping.")
            continue
        combined_xr_list.append(data['combined_xr'])
        input_dict_list.append(data['input_dict_out'])
        eq_labels.append(os.path.basename(eq_filenames[i]))

    if len(combined_xr_list) == 0:
        print("[multi_compile] No successful runs to compile.")
        return None, None

    # Concatenate xarrays along a new 'equilibrium' dimension
    compiled_xr = xr.concat(combined_xr_list, dim='run_idx')
    #compiled_xr = compiled_xr.assign_coords(run_idx=eq_labels)

    # Compile scalar input parameters into an xarray Dataset
    # Extract keys that have scalar (non-dict, non-list) values
    scalar_keys = []
    for key in input_dict_list[0]:
        val = input_dict_list[0][key]
        if isinstance(val, (int, float, bool, str, np.integer, np.floating)):
            scalar_keys.append(key)

    input_data_vars = {}
    for key in scalar_keys:
        vals = []
        for d in input_dict_list:
            vals.append(d.get(key, np.nan))
        try:
            input_data_vars[key] = ('equilibrium', vals)
        except Exception:
            pass

    compiled_inputs_xr = xr.Dataset(input_data_vars, coords={'equilibrium': eq_labels})

    # Save compiled results
    #compiled_path = os.path.join(master_working_dir, 'compiled_results.pkl')
    #with open(compiled_path, 'wb') as f:
    #    pkl.dump({
    #        'compiled_xr': compiled_xr,
    #        'compiled_inputs_xr': compiled_inputs_xr,
    #    }, f)
    #     print(f"[multi_compile] Compiled {len(combined_xr_list)} runs. Saved to {compiled_path}")


    compiled_xr.to_netcdf(os.path.join(master_working_dir, 'compiled_combined_xr.nc'), engine="scipy")
    compiled_inputs_xr.to_netcdf(os.path.join(master_working_dir, 'compiled_inputs_xr.nc'), engine="scipy")

    return compiled_xr, compiled_inputs_xr

def _get_num_cpus():
    """Get the number of available CPUs. Default is to use SLURM environment variables."""
    for var in ['SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE']:
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    # Fallback to OS-reported CPU count
    return len(os.sched_getaffinity(0))


def _clean_working_dir(working_dir):
    """Remove all files from working_dir except executables (rdcon, stride, pest3x).

    This keeps the working directory reusable across successive equilibria
    without needing to re-copy the Fortran executables each time.
    """
    keep = {'rdcon', 'stride', 'pest3x'}
    for entry in os.listdir(working_dir):
        if entry in keep:
            continue
        path = os.path.join(working_dir, entry)
        if os.path.isfile(path):
            os.remove(path)


def _worker_batch(args):
    """Worker function that sequentially processes a batch of equilibria.

    Each worker restricts itself to a single CPU thread and reuses a single
    persistent working directory.  Results for each equilibrium are pickled
    into master_working_dir/result_{idx}.pkl.
    """
    worker_id, batch, working_dir, master_working_dir, kwargs, fail_fast = args
    # batch is a list of (idx, eq_filename, profile_dict) tuples

    # Restrict this process to a single thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    os.makedirs(working_dir, exist_ok=True)

    log_path = os.path.join(master_working_dir, f"worker_{worker_id}.log")
    _log_fh = open(log_path, "w", buffering=1)  # line-buffered
    os.dup2(_log_fh.fileno(), 1)
    os.dup2(_log_fh.fileno(), 2)
    sys.stdout = _log_fh
    sys.stderr = _log_fh

    run_results = []
    for idx, eq_filename, profile_dict in batch:
        # Extract per-profile rotation splines if present, allowing kwargs to override
        per_profile_kwargs = dict(kwargs)
        if 'Er_spline' not in per_profile_kwargs and 'Er_spline' in profile_dict:
            per_profile_kwargs['Er_spline'] = profile_dict['Er_spline']
        if 'omega_splines' not in per_profile_kwargs and 'omega_splines' in profile_dict:
            per_profile_kwargs['omega_splines'] = profile_dict['omega_splines']
        if 'Zeff_spline' not in per_profile_kwargs and 'Zeff_spline' in profile_dict:
            per_profile_kwargs['Zeff'] = profile_dict['Zeff_spline']
        if 'chi_perp_spline' not in per_profile_kwargs and 'chi_perp_spline' in profile_dict:
            per_profile_kwargs['chi_perp_spline'] = profile_dict['chi_perp_spline']
        if 'energy_confinement_time' not in per_profile_kwargs and 'energy_confinement_time' in profile_dict:
            per_profile_kwargs['energy_confinement_time'] = profile_dict['energy_confinement_time']
        if 'average_ion_mass' not in per_profile_kwargs and 'average_ion_mass' in profile_dict:
            per_profile_kwargs['average_ion_mass'] = profile_dict['average_ion_mass']
        try:
            combined_xr, input_dict_out, pest3_xr_vec, xarray_vec = nonlinear_resistive_calculation(
                eq_filename,
                profile_dict['ni_spline'],
                profile_dict['ne_spline'],
                profile_dict['te_keV_spline'],
                profile_dict['ti_keV_spline'],
                working_dir=working_dir,
                **per_profile_kwargs
            )
            # Save result to master_working_dir so it persists after cleanup
            result_path = os.path.join(master_working_dir, f'result_{idx}.pkl')
            with open(result_path, 'wb') as f:
                pkl.dump({
                    'idx': idx,
                    'combined_xr': combined_xr,
                    'input_dict_out': input_dict_out,
                    'pest3_xr_vec': pest3_xr_vec,
                    'xarray_vec': xarray_vec,
                    'eq_filename': eq_filename,
                }, f)
            # Also save combined_xr directly as netCDF
            if combined_xr is not None:
                xr_path = os.path.join(master_working_dir, f'combined_xr_{idx}.nc')
                #check if 'time' is in profile_dict:
                combined_xr = combined_xr.assign_coords(run_idx=idx)
                if 'time' in profile_dict:
                    combined_xr = combined_xr.assign(time=profile_dict['time'])
                if 'time_idx' in profile_dict:  
                    combined_xr = combined_xr.assign(time_idx=profile_dict['time_idx'])
                if 'shot_id' in profile_dict:
                    combined_xr = combined_xr.assign(shot_id=profile_dict['shot_id'])
                combined_xr.to_netcdf(xr_path, engine="scipy")
            print(f"[multi_run] Worker {worker_id}: Run {idx} ({os.path.basename(eq_filename)}) completed successfully.")
            run_results.append((idx, True, None))
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            print(f"[multi_run] Worker {worker_id}: Run {idx} ({os.path.basename(eq_filename)}) failed: {e}\n{tb_str}")
            if fail_fast:
                raise RuntimeError(f"Worker {worker_id}: Run {idx} ({os.path.basename(eq_filename)}) failed") from e
            run_results.append((idx, False, f"{e}\n{tb_str}"))

        # Clean the working directory for the next equilibrium
        _clean_working_dir(working_dir)

    return run_results

