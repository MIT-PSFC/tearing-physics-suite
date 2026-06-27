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
import traceback
from scipy.interpolate import Akima1DInterpolator, PPoly

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
    n_success = 0

    for worker_results in all_results:
        for idx, success, err_msg in worker_results:
            if success:
                n_success += 1
                if return_lists:
                    result_path = os.path.join(master_working_dir, f'result_{idx}.pkl')
                    with open(result_path, 'rb') as f:
                        data = pkl.load(f)
                    combined_xr_list[idx] = data['combined_xr']
                    input_dict_list[idx] = data['input_dict_out']
            else:
                errors[idx] = err_msg
                print(f"[multi_run] WARNING: Run {idx} ({eq_filenames[idx]}) failed: {err_msg}")

    print(f"[multi_run] Completed: {n_success}/{n_runs} runs succeeded.")

    # Always write (or overwrite) the error log so stale results from previous runs don't persist
    error_path = os.path.join(master_working_dir, 'errors.pkl')
    with open(error_path, 'wb') as f:
        pkl.dump(errors, f)
    if errors:
        print(f"[multi_run] Failed runs: {sorted(errors.keys())}")
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
    print("Beginning concatenation...")
    print(combined_xr_list[0].run_idx)
    import sys
    sys.exit()
    try:
        compiled_xr = xr.concat(combined_xr_list, dim='run_idx')
        broken_xr_list = None
    except:
        print("Regular concat failed, attempting intelligent_concat...")
        compiled_xr, broken_xr_list = intelligent_concat(combined_xr_list)
    print("Success")

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

    print("Attempting to save to ", os.path.join(master_working_dir, 'compiled_combined_xr.nc'), os.path.join(master_working_dir, 'compiled_inputs_xr.nc'), os.path.join(master_working_dir, 'broken_xr_list.pkl'))
    compiled_xr.to_netcdf(os.path.join(master_working_dir, 'compiled_combined_xr.nc'), engine="scipy")
    compiled_inputs_xr.to_netcdf(os.path.join(master_working_dir, 'compiled_inputs_xr.nc'), engine="scipy")
    if broken_xr_list:
        broken_xr_path = os.path.join(master_working_dir, 'broken_xr_list.pkl')
        with open(broken_xr_path, 'wb') as f:
            pkl.dump(broken_xr_list, f)
        print(f"[multi_compile] Saved {len(broken_xr_list)} broken xarrays to {broken_xr_path}")
    print("Save successful")

    return compiled_xr, compiled_inputs_xr

def _signature(ds, dim='run_idx'):
    """Cheap structural fingerprint that must match for a clean concat along `dim`.
    Captures the things that typically break xr.concat: differing data variables,
    differing non-concat dims/sizes, and differing coordinate sets."""
    data_vars  = frozenset(ds.data_vars)
    dims       = frozenset(d for d in ds.dims if d != dim)
    dim_sizes  = frozenset((d, ds.sizes[d]) for d in ds.dims if d != dim)
    coords     = frozenset(ds.coords)
    return (data_vars, dims, dim_sizes, coords)

def _divide_and_conquer(chunk, dim='run_idx'):
    """Find value-level offenders (e.g. conflicting non-index coordinate values)
    that survive metadata filtering. Returns (combined_or_None, broken_list).
    Costs ~O(k log n) concat attempts for k bad arrays."""
    broken = []

    def attempt(sub):
        if not sub:
            return None
        try:
            return xr.concat(sub, dim=dim)
        except Exception:
            if len(sub) == 1:
                broken.append(sub[0])
                return None
            mid = len(sub) // 2
            left  = attempt(sub[:mid])
            right = attempt(sub[mid:])
            parts = [p for p in (left, right) if p is not None]
            if not parts:
                return None
            try:
                return xr.concat(parts, dim=dim)
            except Exception:
                # Two individually-clean halves are mutually incompatible.
                # Keep the larger half; demote the smaller's members to broken.
                left_n  = left.sizes.get(dim, 1)  if left  is not None else 0
                right_n = right.sizes.get(dim, 1) if right is not None else 0
                if left_n >= right_n:
                    if right is not None:
                        broken.extend(sub[mid:])
                    return left
                else:
                    if left is not None:
                        broken.extend(sub[:mid])
                    return right

    combined = attempt(list(chunk))
    return combined, broken

def intelligent_concat(combined_xr_list, dim='run_idx'):
    """Weed out xarrays that prevent xr.concat(combined_xr_list, dim=dim).

    Strategy:
      1. Bucket arrays by a cheap structural signature -> O(n) scan.
      2. Concatenate the largest (majority) bucket directly.
      3. If that still fails due to value-level conflicts, fall back to a
         divide-and-conquer search within the bucket to isolate offenders.
      4. Normalize so the return contract is consistent.

    Returns
    -------
    combined : xarray object or None
        A SINGLE concatenated xarray that always has `dim` present,
        or None if nothing could be concatenated.
    broken : list
        The arrays that were excluded.
    """
    if not combined_xr_list:
        return None, []

    # --- Step 1: bucket by signature -------------------------------------
    buckets = {}
    for x in combined_xr_list:
        buckets.setdefault(_signature(x, dim), []).append(x)

    # --- Step 2: pick the majority bucket; everything else is broken ------
    best_sig   = max(buckets, key=lambda s: len(buckets[s]))
    candidates = buckets[best_sig]
    broken     = [x for s, grp in buckets.items() if s != best_sig for x in grp]

    # --- Step 3: concat, with value-level fallback ------------------------
    try:
        combined = xr.concat(candidates, dim=dim)
    except Exception:
        combined, extra_broken = _divide_and_conquer(candidates, dim)
        broken.extend(extra_broken)

    # --- Step 4: normalize the contract ----------------------------------
    # Guarantee a single xarray with `dim` present (or None).
    if combined is not None and dim not in combined.dims:
        combined = combined.expand_dims(dim)

    return combined, broken

import os
import gc
import shutil
import pickle as pkl

import numpy as np
import xarray as xr
import zarr

# Reuse the streaming helpers from the attached module
from tearing_physics_suite.dataset_utils import add_to_zarr_store, zarr_chunk

def _zarr_signature(ds, dim="run_idx"):
    """Relaxed structural fingerprint for streaming concat.

    Unlike a size-sensitive signature, this allows arrays with the SAME
    variables/dims/coords but DIFFERENT sizes along non-concat dims, because
    `add_to_zarr_store` pads/extends ragged dims with NaN automatically.
    """
    data_vars = frozenset(ds.data_vars)
    dims = frozenset(d for d in ds.dims if d != dim)
    coords = frozenset(ds.coords)
    return (data_vars, dims, coords)

def multi_compile_zarr(
    eq_filenames,
    master_working_dir,
    shot_time_list=None,
    debug=True,
    episode_dim="run_idx",
    varying_dim=None,           # ragged dim for store_time_dim_size hint (auto-detected from first run)
    mb_per_chunk=10,
    report_errs=True,
    meta_data_dicts=[],
):
    """Memory-bounded version of multi_compile.

    Instead of loading every per-run xarray into RAM and calling xr.concat
    (which needs the whole stack + the merged copy resident at once), this
    streams each run into a Zarr store on disk, holding only ~one array in
    memory at a time.

    Returns
    -------
    compiled_xr : xr.Dataset
        A LAZY (dask-backed) dataset opened from the Zarr store. It is never
        fully materialized in memory; slice/compute the parts you need.
    compiled_inputs_xr : xr.Dataset
        Small scalar-input dataset (safe to keep in memory).
    """
    n_runs = len(eq_filenames)
    if meta_data_dicts:
        assert len(meta_data_dicts)==n_runs
    zarr_path = os.path.join(master_working_dir, "compiled_combined_xr.zarr")

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    # --- error log handling ----------------------------------------------
    errors_path = os.path.join(master_working_dir, "errors.pkl")
    if os.path.exists(errors_path) and report_errs:
        with open(errors_path, "rb") as f:
            errors = pkl.load(f)
        num_errors = sum(1 for v in errors.values() if v is not None)
        print(f"[multi_compile] Found error log with {num_errors} failed runs.")
        if debug:
            for idx, err_msg in sorted(errors.items()):
                if err_msg is not None:
                    print(f"[multi_compile] Run {idx} error: {err_msg}")
    elif report_errs:
        print(f"[multi_compile] No error log found. Assuming all {n_runs} runs succeeded.")

    # --- stream each run into the Zarr store -----------------------------
    eq_labels = []
    input_dict_list = []
    broken_runs = []
    ref_signature = None
    store_dim_sizes = {}   # tracks current max size of each non-episode dim
    store_time_dim_size = None  # kept for add_to_zarr_store API compat
    n_added = 0

    print("Beginning streaming concatenation into Zarr...")
    for i in range(3):
        result_path = os.path.join(master_working_dir, f"result_{i}.pkl")
        if not os.path.exists(result_path):
            print(f"[multi_compile] WARNING: Result file not found for run {i}, skipping.")
            continue

        with open(result_path, "rb") as f:
            data = pkl.load(f)

        ds = data["combined_xr"]
        if ds is None:
            print(f"[multi_compile] WARNING: Run {i} has no combined_xr, skipping.")
            del data
            continue

        # Structural compatibility gate (replaces intelligent_concat bucketing).
        sig = _zarr_signature(ds, dim=episode_dim)
        if ref_signature is None:
            ref_signature = sig
        elif sig != ref_signature:
            print(f"[multi_compile] Run {i} has incompatible structure -> broken.")
            broken_runs.append(i)
            del ds, data
            continue

        eq_label = os.path.basename(eq_filenames[i])
        
        # Set Zeff's coordinate to be psi_N_Zeff
        zeff_values = np.asarray(ds["Zeff"].values)
        ds = ds.drop_vars("Zeff")
        if "Zeff" in ds.dims:
            if ds.sizes["Zeff"] != ds.sizes["psi_N_Zeff"]:
                raise ValueError(
                    f"Cannot merge: Zeff has length {ds.sizes['Zeff']} "
                    f"but psi_N_Zeff has length {ds.sizes['psi_N_Zeff']}."
                )
            rebuilt = {}
            for name, da in ds.data_vars.items():
                if "Zeff" in da.dims:
                    new_dims = tuple("psi_N_Zeff" if d == "Zeff" else d for d in da.dims)
                    rebuilt[name] = (new_dims, da.values)
            for name, val in rebuilt.items():
                ds[name] = val
        ds["Zeff"] = xr.DataArray(zeff_values, dims="psi_N_Zeff")
        ds = ds.rename({"psi_N_Zeff": "psi_n_Zeff"})

        # Cast r / r_prime index coords to float to allow nans
        for _dim in ("r", "r_prime"):
            if _dim in ds.coords and np.issubdtype(ds[_dim].dtype, np.integer):
                ds = ds.assign_coords({_dim: ds[_dim].astype("float64")})

        if meta_data_dicts:
            meta_ds = meta_dict_to_dataset(meta_data_dicts[i], ds=ds,
                                        episode_dim=episode_dim, verbose=debug)
            meta_ds = meta_ds.drop_vars('geqdsk_path')
            # Set the episode coordinate for everything inside meta_ds:
            #meta_ds = meta_ds.assign_coords({episode_dim: np.array([i])})
            """
            ds_vars     = set(ds.data_vars)
            meta_vars   = set(meta_ds.data_vars)
            ds_dims     = set(ds.dims)
            meta_dims   = set(meta_ds.dims)

            # data var in one side whose name is a dimension in the other
            c = ds_vars   & meta_dims     # data_var in ds, dimension in meta_ds
            d = meta_vars & ds_dims       # data_var in meta_ds, dimension in ds

            # also catch: data var in one side whose name is a dimension on its OWN side
            # (dim-without-coordinate colliding with a data var of the same name)
            e = ds_vars   & ds_dims
            f = meta_vars & meta_dims

            print("data_var in ds, dim in meta_ds:", c)
            print("data_var in meta_ds, dim in ds:", d)
            print("data_var collides with own dim (ds):", e)
            print("data_var collides with own dim (meta_ds):", f)
            print("REAL CULPRITS:", c | d | e | f)
            print(ds.dims)
            print(meta_ds.dims)
            print(ds.psi_n_Zeff)
            print(meta_ds.psi_n_IDA)
            sys.exit()
            """
            ds = xr.merge([ds, meta_ds], compat="no_conflicts", combine_attrs="override")
            #if "psi_n_IDA" in ds.dims:
            #    ds = ds.assign_coords(psi_n_IDA=np.arange(ds.sizes["psi_n_IDA"]))

        # --- ROBUST run_idx handling -------------------------------------
        # add_to_zarr_store calls reset_coords(), which would demote/lose a
        # run_idx coordinate (esp. with string labels). So we feed run_idx in
        # as a BARE integer dimension (no coord) and carry the eq label as an
        # ordinary data variable that survives reset_coords() and the append.
        if episode_dim in ds.coords:
            ds = ds.reset_coords(episode_dim, drop=True)
        if episode_dim not in ds.dims:
            ds = ds.expand_dims(episode_dim)
        # ds["eq_label"] = (episode_dim, np.array([eq_label], dtype=object))
        # -----------------------------------------------------------------

        if debug:
            print(f"[multi_compile] Run {i} dataset structure:")
            dims_str = "\n    ".join(f"{k}: {v}" for k, v in ds.sizes.items())
            print(f"  Dimensions:\n    {dims_str}")
            print()
            coords_str = "\n    ".join(ds.coords)
            print(f"  Coordinates:\n    {coords_str}")
            print()
            vars_str = "\n    ".join(ds.data_vars)
            print(f"  Variables:\n    {vars_str}")
            #import sys
            #sys.exit()

        #try:
        success = add_to_zarr_store(
            ds,
            zarr_path,
            episode_dim=episode_dim,
            store_time_dim_size=store_time_dim_size,
        )
        #except Exception as e:
        #    print(f"[multi_compile] Run {i} failed to append ({e}) -> broken.")
        #    broken_runs.append(i)
        #    del ds, data
        #    continue

        if success:
            # Update tracked max sizes for all non-episode dims
            for _d in ds.dims:
                if _d != episode_dim:
                    store_dim_sizes[_d] = max(store_dim_sizes.get(_d, 0), ds.sizes[_d])
            # Keep varying_dim / store_time_dim_size in sync for add_to_zarr_store API
            if varying_dim is None:
                varying_dim = next((d for d in ds.dims if d != episode_dim), episode_dim)
            store_time_dim_size = store_dim_sizes.get(varying_dim)
            eq_labels.append(eq_label)
            input_dict_list.append(data["input_dict_out"])
            n_added += 1

        del ds, data
        if i > 0 and i % 40 == 0:
            gc.collect()

    if n_added == 0:
        print("[multi_compile] No successful runs to compile.")
        return None, None

    # --- chunk + consolidate (so the store reads back efficiently) -------
    ds_store = xr.open_zarr(zarr_path, consolidated=True)
    bytes_per_episode = ds_store.isel({episode_dim: 0}).nbytes
    mean_mb = bytes_per_episode / (1024 * 1024)
    episodes_per_chunk = min(max(1, int(mb_per_chunk / max(mean_mb, 1e-9))), n_added)

    chunk_spec = {episode_dim: episodes_per_chunk} | {
        k: ds_store.sizes[k] for k in ds_store.dims if k != episode_dim
    }
    ds_store = zarr_chunk(ds_store, chunk_spec=chunk_spec)

    tmp_path = zarr_path + ".tmp"
    ds_store.to_zarr(tmp_path, mode="w", consolidated=True)
    shutil.rmtree(zarr_path)
    os.rename(tmp_path, zarr_path)
    zarr.consolidate_metadata(zarr_path)

    # --- reopen lazily and attach run_idx labels from the in-memory list ---
    compiled_xr = xr.open_zarr(zarr_path, consolidated=True)  # lazy, not in RAM

    # eq_labels was appended only on successful store, in the same order as
    # the run_idx dimension -> safe to assign positionally as the coordinate.
    assert len(eq_labels) == compiled_xr.sizes[episode_dim], (
        f"label/run_idx length mismatch: {len(eq_labels)} vs "
        f"{compiled_xr.sizes[episode_dim]}"
    )
    compiled_xr = compiled_xr.assign_coords({episode_dim: eq_labels})

    # --- scalar inputs (small, fine in memory) ---------------------------
    scalar_keys = [
        key
        for key in input_dict_list[0]
        if isinstance(
            input_dict_list[0][key],
            (int, float, bool, str, np.integer, np.floating),
        )
    ]
    input_data_vars = {}
    for key in scalar_keys:
        vals = [d.get(key, np.nan) for d in input_dict_list]
        input_data_vars[key] = (episode_dim, vals)

    compiled_inputs_xr = xr.Dataset(
        input_data_vars, coords={episode_dim: eq_labels}
    )
    inputs_path = os.path.join(master_working_dir, "compiled_inputs_xr.nc")
    compiled_inputs_xr.to_netcdf(inputs_path, engine="scipy")

    if broken_runs:
        with open(os.path.join(master_working_dir, "broken_runs.pkl"), "wb") as f:
            pkl.dump(broken_runs, f)
        print(f"[multi_compile] {len(broken_runs)} runs excluded: {broken_runs}")

    print(f"[multi_compile] Compiled {n_added} runs into {zarr_path}")
    return compiled_xr, compiled_inputs_xr

def intelligent_concat_check(combined_xr_list, dim='run_idx'):
    """Weed out xarrays that prevent xr.concat(combined_xr_list, dim=dim).

    Strategy:
      1. Bucket arrays by a cheap structural signature -> O(n) scan.
      2. Concatenate the largest (majority) bucket directly.
      3. If that still fails due to value-level conflicts, fall back to a
         divide-and-conquer search within the bucket to isolate offenders.
      4. Normalize so the return contract is consistent.

    Returns
    -------
    combined : xarray object or None
        A SINGLE concatenated xarray that always has `dim` present,
        or None if nothing could be concatenated.
    broken : list
        The arrays that were excluded.
    """
    if not combined_xr_list:
        return None, []

    # --- Step 1: bucket by signature -------------------------------------
    buckets = {}
    for x in combined_xr_list:
        buckets.setdefault(_signature(x, dim), []).append(x)

    # --- Step 2: pick the majority bucket; everything else is broken ------
    best_sig   = max(buckets, key=lambda s: len(buckets[s]))
    candidates = buckets[best_sig]
    broken     = [x for s, grp in buckets.items() if s != best_sig for x in grp]

    return candidates, broken

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
    # Disable HDF5 file locking (required for NFS filesystems)
    # os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

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
        all_profile_kwargs = dict(kwargs)
        tau_e_label = all_profile_kwargs.pop('tau_e_label',None)
        if 'Er_spline' not in all_profile_kwargs and 'Er_spline' in profile_dict:
            all_profile_kwargs['Er_spline'] = profile_dict['Er_spline']
        if 'omega_splines' not in all_profile_kwargs and 'omega_splines' in profile_dict:
            all_profile_kwargs['omega_splines'] = profile_dict['omega_splines']
        if 'Zeff' not in all_profile_kwargs and 'Zeff' in profile_dict:
            all_profile_kwargs['Zeff'] = profile_dict['Zeff']
        if 'chi_perp_spline' not in all_profile_kwargs and 'chi_perp_spline' in profile_dict:
            all_profile_kwargs['chi_perp_spline'] = profile_dict['chi_perp_spline']
        if 'energy_confinement_time' not in all_profile_kwargs and 'energy_confinement_time' in profile_dict:
            all_profile_kwargs['energy_confinement_time'] = profile_dict['energy_confinement_time']
        if 'average_ion_mass' not in all_profile_kwargs and 'average_ion_mass' in profile_dict:
            all_profile_kwargs['average_ion_mass'] = profile_dict['average_ion_mass']
        if 'energy_confinement_time' not in all_profile_kwargs and tau_e_label in profile_dict:
            all_profile_kwargs['energy_confinement_time'] = profile_dict[tau_e_label]
        try:
            combined_xr, input_dict_out, pest3_xr_vec, xarray_vec = nonlinear_resistive_calculation(
                eq_filename,
                profile_dict['ni_spline'],
                profile_dict['ne_spline'],
                profile_dict['te_keV_spline'],
                profile_dict['ti_keV_spline'],
                working_dir=working_dir,
                **all_profile_kwargs
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
            run_results.append((idx, False, f"{e}\n{tb_str}"))
            if fail_fast:
                raise RuntimeError(f"Worker {worker_id}: Run {idx} ({os.path.basename(eq_filename)}) failed") from e

        # Clean the working directory for the next equilibrium
        _clean_working_dir(working_dir)

    return run_results

def meta_dict_to_dataset(meta, ds=None, episode_dim="run_idx",
                         rtol=1e-9, atol=1e-12, verbose=False) -> xr.Dataset:
    reg = _GridRegistry(ds=ds, episode_dim=episode_dim)

    scalars_done, vec_scalars, xy_entries = {}, [], []
    for item in _flatten_meta(meta):
        if item[0] == "scalar":
            kind, payload = _scalar_spec(item[2])
            if kind == "done":
                scalars_done[item[1]] = payload
            elif kind == "vector":
                vec_scalars.append((item[1], payload))     # (name, array)
        else:
            xy_entries.append(item[1:])                     # (name, x, y)

    dv = dict(scalars_done)

    # --- Phase 1: ds-coord match, or new cluster (unchanged) -----------------
    assigned, clusters = {}, []
    for idx, (name, x, y) in enumerate(xy_entries):
        dim = reg.match(x, rtol, atol)
        if dim is not None:
            assigned[idx] = dim
            if verbose:
                tag = "ds-coord" if dim in reg.from_ds else "shared"
                print(f"[meta] '{name}' reuses {tag} grid '{dim}'")
            continue
        for cl in clusters:
            if _grids_close(x, cl["x"], rtol, atol):
                cl["members"].append((idx, name)); break
        else:
            clusters.append({"x": np.asarray(x, float), "members": [(idx, name)]})

    # --- Phase 2: name new clusters (IDA rule) -------------------------------
    new_grid_x = {}
    for cl in clusters:
        names = [nm for _, nm in cl["members"]]
        base = "psi_n_IDA" if any(nm.startswith("IDA") for nm in names) else f"{names[0]}_knot"
        dim = reg.register(cl["x"], base)
        new_grid_x[dim] = cl["x"]
        for idx, _ in cl["members"]:
            assigned[idx] = dim
        if verbose:
            print(f"[meta] new grid '{dim}' <- {names} (len {len(cl['x'])})")

    # --- Phase 3: emit xy y-vars + each new grid's x -------------------------
    for idx, (name, x, y) in enumerate(xy_entries):
        dv[f"{name}"] = (assigned[idx], np.asarray(y, float))
    for dim, x in new_grid_x.items():
        dv[f"{dim}_x"] = (dim, np.asarray(x, float))

    # --- Phase 4: place deferred vectors (IDA_* -> psi_n_IDA if length fits) --
    ida_x = reg.grids.get("psi_n_IDA")          # now populated if a cluster made it
    ida_len = len(ida_x) if ida_x is not None else None
    for name, arr in vec_scalars:
        if name.startswith("IDA") and ida_len is not None and arr.shape[0] == ida_len:
            dv[name] = ("psi_n_IDA", arr)        # <-- ride the shared IDA grid
            if verbose:
                print(f"[meta] vector '{name}' (len {arr.shape[0]}) -> 'psi_n_IDA'")
        else:
            dv[name] = (f"{name}_dim", arr)      # own dim (fallback)

    ds_out = xr.Dataset(dv)

    # Elevate the IDA grid's x-variable to a 1-D coordinate on its own dim.
    # (Do this BEFORE expand_dims so it stays shared, not per-run.)
    ida_dim = "psi_n_IDA"                      # the registered IDA dim name
    xname = f"{ida_dim}_x"                     # -> "psi_n_IDA_x"
    if xname in ds_out.data_vars and ida_dim in ds_out.dims:
            ds_out = ds_out.rename({xname: ida_dim}).set_coords(ida_dim)
            # set_coords() does NOT build an index -> add one explicitly
            if ida_dim not in ds_out.xindexes:
                ds_out = ds_out.set_xindex(ida_dim)
            if verbose:
                print(f"[meta] elevated '{xname}' -> indexed coord '{ida_dim}' on dim '{ida_dim}'")
    
    return ds_out.expand_dims(episode_dim)

def _flatten_meta(meta, prefix=""):
    """Yield ('scalar', name, val) or ('xy', name, x, y)."""
    for key, val in meta.items():
        if key in DROP_KEYS:                       # <-- requirement 1
            continue
        name = _sanitize(prefix + key)
        if isinstance(val, (Akima1DInterpolator, PPoly)):
            x, y = _spline_xy(val)
            yield ("xy", name, x, y)
        elif isinstance(val, dict):
            if {"x", "y"} <= set(val) and not isinstance(val.get("x"), dict):
                yield ("xy", name,
                       np.asarray(val["x"], float), np.asarray(val["y"], float))
            else:
                yield from _flatten_meta(val, prefix=name + "_")
        else:
            yield ("scalar", name, val)

def _scalar_spec(val):
    """Classify a non-xy meta value.
    Returns ('done', (dims, data)) for true scalars/strings/0-d,
            ('vector', np.ndarray)  for length>1 numeric arrays (defer),
            ('skip', None)          for unsupported.
    """
    if isinstance(val, (int, float, bool, np.integer, np.floating)):
        return ("done", ((), np.asarray(val)))
    if isinstance(val, str):
        return ("done", ((), np.array(val, dtype=object)))
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            data = np.array(val.item(), dtype=object) if val.dtype == object else val
            return ("done", ((), data))
        if val.size == 1 and (val.dtype == object or np.issubdtype(val.dtype, np.str_)):
            return ("done", ((), np.array(val.reshape(()).item(), dtype=object)))
        return ("vector", np.asarray(val))      # <-- defer: dim decided later
    return ("skip", None)

class _GridRegistry:
    """Maps an x-grid to a dimension name, reusing ds coords where possible."""
    def __init__(self, ds=None, episode_dim="run_idx"):
        self.grids = {}        # dim_name -> x array
        self.from_ds = set()   # dims that already live in ds (don't re-store x)
        if ds is not None:
            for c in ds.coords:
                arr = ds[c]
                if (arr.ndim == 1 and np.issubdtype(arr.dtype, np.number)
                        and arr.dims[0] != episode_dim):
                    dim = arr.dims[0]
                    self.grids[dim] = np.asarray(arr.values, float)
                    self.from_ds.add(dim)

    def match(self, x, rtol=1e-9, atol=1e-12):
        for dim, gx in self.grids.items():
            if _grids_close(x, gx, rtol, atol):
                return dim
        return None

    def register(self, x, preferred):
        dim, base, i = preferred, preferred, 1
        while dim in self.grids:
            dim, i = f"{base}_{i}", i + 1
        self.grids[dim] = np.asarray(x, float)
        return dim

# 1) Keys to discard entirely (already in the main dataset)
DROP_KEYS = {
    "ne_spline", "ni_spline", "te_keV_spline", "ti_keV_spline",
    "Er_spline", "omega_splines", "Zeff"
}

def _sanitize(name: str) -> str:
    for a, b in [("(", "_"), (")", ""), (" ", "_"), ("[", ""), ("]", ""),
                 ("^", ""), ("-", "m"), ("/", "_"), (",", "_")]:
        name = name.replace(a, b)
    return name.strip("_")

def _spline_xy(spl):
    x = np.asarray(spl.x, dtype="float64")
    return x, np.asarray(spl(x), dtype="float64")     # exact for Akima

def _grids_close(a, b, rtol=1e-9, atol=1e-12):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)