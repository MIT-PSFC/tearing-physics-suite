# "dataset_utils_TPS.py"
import gc
import os
import shutil
from collections.abc import Callable, Iterable
from typing import Any

import jax
import loguru
import xarray as xr
import zarr
from tqdm import tqdm

GC_INTERVAL = 40  # Force garbage collection every 40 shots


# multi_compile_zarr

import os
import gc
import shutil
import pickle as pkl

import numpy as np
import xarray as xr
import zarr

import os
import sys
import glob
import multiprocessing
import pickle as pkl
import xarray as xr
import numpy as np
import traceback
from scipy.interpolate import Akima1DInterpolator, PPoly


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

def _n_cpus():
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        v = os.environ.get(var)
        if v:
            return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _transform_run(i, master_working_dir, eq_filename, episode_dim,
                   meta_dict=None, debug=False):
    """Load result_{i}.pkl and apply the same per-run transform used by
    multi_compile_zarr. Returns (ds, input_dict, eq_label) or (None, None, None).
    NOTE: eq_filename is a SINGLE path (not the whole list) to keep task
    payloads tiny when this is dispatched across 10k parallel tasks."""
    result_path = os.path.join(master_working_dir, f"result_{i}.pkl")
    if not os.path.exists(result_path):
        return None, None, None
    with open(result_path, "rb") as f:
        data = pkl.load(f)
    ds = data.get("combined_xr")
    if ds is None:
        return None, None, None

    eq_label = os.path.basename(eq_filename)

    # --- Zeff -> psi_n_Zeff ---
    zeff_values = np.asarray(ds["Zeff"].values)
    ds = ds.drop_vars("Zeff")
    if "Zeff" in ds.dims:
        if ds.sizes["Zeff"] != ds.sizes["psi_N_Zeff"]:
            raise ValueError(
                f"Cannot merge: Zeff has length {ds.sizes['Zeff']} "
                f"but psi_N_Zeff has length {ds.sizes['psi_N_Zeff']}.")
        rebuilt = {}
        for name, da in ds.data_vars.items():
            if "Zeff" in da.dims:
                new_dims = tuple("psi_N_Zeff" if d == "Zeff" else d for d in da.dims)
                rebuilt[name] = (new_dims, da.values)
        for name, val in rebuilt.items():
            ds[name] = val
    ds["Zeff"] = xr.DataArray(zeff_values, dims="psi_N_Zeff")
    ds = ds.rename({"psi_N_Zeff": "psi_n_Zeff"})

    # --- float-cast integer index coords so NaN padding works ---
    for _dim in ("r", "r_prime"):
        if _dim in ds.coords and np.issubdtype(ds[_dim].dtype, np.integer):
            ds = ds.assign_coords({_dim: ds[_dim].astype("float64")})

    if meta_dict is not None:
        meta_ds = meta_dict_to_dataset(meta_dict, ds=ds,
                                       episode_dim=episode_dim, verbose=debug)
        meta_ds = meta_ds.drop_vars("geqdsk_path")
        ds = xr.merge([ds, meta_ds], compat="no_conflicts", combine_attrs="override")

    if episode_dim in ds.coords:
        ds = ds.reset_coords(episode_dim, drop=True)
    if episode_dim not in ds.dims:
        ds = ds.expand_dims(episode_dim)

    return ds, data.get("input_dict_out"), eq_label


def _normalize_for_store(ds, episode_dim):
    """Reproduce add_to_zarr_store's normalization: non-index coords become
    per-episode data vars; every data var carries the episode dim."""
    ds = ds.reset_coords()  # non-dimension coords -> data vars (vary per episode)
    if episode_dim not in ds.dims:
        ds = ds.expand_dims(episode_dim)
    for v in list(ds.data_vars):
        if episode_dim not in ds[v].dims:
            ds[v] = ds[v].expand_dims(episode_dim)
    return ds


def _promote_and_pad(ds, max_sizes, ragged_dims, union_coords=None):
    """Cast numeric ragged-dim vars to float64, reindex union dims onto the
    shared lattice, then NaN-pad the remaining ragged dims (left-aligned).

    - UNION dims (Option B): reindexed onto a common coordinate lattice so a
      given physical value (e.g. mode number m=5) lands in the SAME slot in
      every episode. Missing values become NaN in the correct positions.
    - Other ragged dims: left-aligned NaN pad up to max_sizes (assumes the
      coord is a simple left-aligned/contiguous extension, verified in Pass 1).
    """
    union_coords = union_coords or {}

    # 1) float-cast numeric data vars on any ragged dim (so NaN is legal
    #    and dtypes match the skeleton)
    for v in list(ds.data_vars):
        vdims = ds[v].dims
        if any(d in ragged_dims for d in vdims) \
           and np.issubdtype(ds[v].dtype, np.number) \
           and not np.issubdtype(ds[v].dtype, np.floating):
            ds[v] = ds[v].astype("float64")

    # 2) UNION dims: reindex onto the shared lattice (physical registration)
    for d, target in union_coords.items():
        if d in ds.dims:
            ds = ds.reindex({d: target})   # missing values -> NaN, correct slots

    # 3) remaining ragged dims: left-aligned NaN pad (old behavior)
    pad = {}
    for d in ragged_dims:
        if d in union_coords:
            continue                       # already handled by reindex
        if d in ds.dims and ds.sizes[d] < max_sizes[d]:
            pad[d] = (0, max_sizes[d] - ds.sizes[d])
    if pad:
        ds = ds.pad(pad, constant_values=np.nan)
    return ds

def _compute_union_grids(scan_results, union_dims, ref_sig, episode_dim):
    """Option B: build a common (union) coordinate lattice for each dim in
    `union_dims`, across all structurally-compatible runs.

    Each run may cover a different *range* of the same underlying lattice
    (e.g. poloidal mode number m = -12..40 vs -11..39). We take the sorted
    union of all coord values so that, after reindex, a given physical value
    (m=5) always lands in the SAME slot in every episode.

    ONLY-IF-NECESSARY: a dim is unionized only when its coordinate actually
    VARIES across runs. If every run shares an identical coord for a dim, we
    skip it -> it remains a plain shared coord (no reindex, no store growth).
    Consequently `union_coords` may be a subset of `union_dims`, or empty.

    If anything needed unioning, a PICKLEABLE report (plain dict of built-in
    types / lists) is printed summarizing what was merged.

    Returns
    -------
    union_coords : dict[str, np.ndarray]
        dim -> sorted unique coord values (target lattice), for dims that vary.
    """
    # Collect each run's coord (and its run index) per union dim.
    accum = {d: [] for d in union_dims}       # d -> list of (run_idx, np.ndarray)
    for (i, ok, err, sizes, sig, eq_label, input_dict) in scan_results:
        if not ok or sig != ref_sig or sizes is None:
            continue
        coords = sizes.get("__coords__", {})
        for d in union_dims:
            arr = coords.get(d)
            if arr is not None:
                accum[d].append((i, np.asarray(arr)))

    union_coords = {}
    report = {}   # pickleable summary, only populated for dims that were unioned

    for d, entries in accum.items():
        if not entries:
            continue
        arrs = [a for _, a in entries]

        # --- necessity check: are all runs' coords identical? --------------
        ref = arrs[0]
        varies = False
        for a in arrs[1:]:
            if a.shape != ref.shape or not np.array_equal(a, ref):
                varies = True
                break
        if not varies:
            # identical everywhere -> no union needed; stays a shared coord
            continue

        # --- build the shared lattice --------------------------------------
        u = np.unique(np.concatenate(arrs))   # sorted + de-duplicated
        if all(np.issubdtype(a.dtype, np.integer) for a in arrs):
            u = u.astype(ref.dtype)
        elif np.allclose(u, np.round(u)):
            pass  # all-float but integral-valued lattice -> keep as-is
        union_coords[d] = u

        # --- pickleable report entry ---------------------------------------
        sizes_seen = sorted({int(a.size) for a in arrs})
        # Which runs are "short" (missing at least one union value) -> get NaNs.
        uset = set(u.tolist())
        short_runs = [int(ri) for ri, a in entries
                      if not uset.issubset(set(a.tolist()))]
        report[d] = {
            "union_len": int(u.size),
            "union_dtype": str(u.dtype),
            "per_run_sizes": sizes_seen,        # e.g. [52, 53]
            "min_size": int(min(sizes_seen)),
            "max_size": int(max(sizes_seen)),
            "union_min": (int(u.min()) if np.issubdtype(u.dtype, np.integer)
                          else float(u.min())),
            "union_max": (int(u.max()) if np.issubdtype(u.dtype, np.integer)
                          else float(u.max())),
            "n_runs_padded": len(short_runs),   # runs that gain NaN slots
            "example_padded_runs": short_runs[:10],
        }

    # --- print the pickleable findings, only if anything was unioned -------
    if report:
        # report is a plain dict of str -> dict of {int,float,str,list} -> pickle-safe
        print("[multi_compile] Option B union report (pickleable):")
        for d, info in report.items():
            print(f"    {d!r}: {info}")
        # sanity: prove it round-trips through pickle (cheap, sizes are tiny)
        try:
            pkl.loads(pkl.dumps(report))
        except Exception as e:  # pragma: no cover - should never happen
            print(f"[multi_compile] WARNING: union report not pickleable: {e}")

    return union_coords, report

def _scan_run_worker(args):
    """PASS 1 task: transform + normalize a run, return its dim sizes,
    structural signature, label and input_dict. Discards the array (low mem).

    Also records, for each ragged-candidate INDEX coord, its first value and
    stride (under sizes['__align__']) so the driver can verify left-aligned
    runs, and the full unique coord values for union dims (under
    sizes['__coords__']) so the driver can build a shared lattice (Option B)."""
    (i, master_working_dir, eq_filename, episode_dim, meta_dict, debug,
     union_dims) = args
    try:
        ds, input_dict, eq_label = _transform_run(
            i, master_working_dir, eq_filename, episode_dim, meta_dict, debug)
        if ds is None:
            return (i, False, "missing/none", None, None, None, None)
        ds = _normalize_for_store(ds, episode_dim)

        sizes = {d: int(ds.sizes[d]) for d in ds.dims if d != episode_dim}

        # Alignment fingerprint for any dim that is ALSO an index coordinate.
        # Keyed under synthetic "__align__" so the driver can pull it out and
        # strip it before building max_sizes.
        align = {}
        for d in list(ds.dims):
            if d == episode_dim:
                continue
            if d in ds.coords:
                c = np.asarray(ds[d].values)
                if c.size and np.issubdtype(c.dtype, np.number):
                    first = float(c[0])
                    step = float(c[1] - c[0]) if c.size > 1 else 0.0
                    is_int = bool(np.issubdtype(c.dtype, np.integer))
                    align[d] = (first, step, is_int)
        sizes["__align__"] = align

        # Option B: ship unique coord values for union dims so the driver can
        # build a shared lattice. Only these dims are serialized -> small.
        coords = {}
        for d in union_dims:
            if d in ds.coords:
                coords[d] = np.asarray(ds[d].values)
        sizes["__coords__"] = coords

        sig = _zarr_signature(ds, dim=episode_dim)
        return (i, True, None, sizes, sig, eq_label, input_dict)
    except Exception:
        import traceback
        return (i, False, traceback.format_exc(), None, None, None, None)


def _write_run_worker(args):
    """PASS 2 task: transform, normalize, pad/reindex, drop coords, region-write
    one run into its own episode slice. Safe in parallel because episode chunk
    == 1."""
    (pos, i, master_working_dir, eq_filename, episode_dim, meta_dict,
     zarr_path, max_sizes, ragged_dims, union_coords) = args
    try:
        ds, _, _ = _transform_run(
            i, master_working_dir, eq_filename, episode_dim, meta_dict, debug=False)
        if ds is None:
            return (i, False, "missing/none at write")
        ds = _normalize_for_store(ds, episode_dim)
        ds = _promote_and_pad(ds, max_sizes, ragged_dims, union_coords)
        # region writes only touch data vars; shared coords already live in skeleton
        ds = ds.drop_vars(list(ds.coords))
        ds = ds.load()  # one episode -> tiny, force numpy
        ds.to_zarr(zarr_path, region={episode_dim: slice(pos, pos + 1)}, mode="r+")
        return (i, True, None)
    except Exception:
        import traceback
        return (i, False, traceback.format_exc())

def multi_compile_zarr(
    eq_filenames,
    master_working_dir,
    shot_time_list=None,
    debug=True,
    episode_dim="run_idx",
    varying_dim=None,           # ragged dim for store_time_dim_size hint (auto-detected from first run)
    union_dims=("m", "mode", "lr_index", "lr_prime"),  # Option B: unionize these
    mb_per_chunk=10,
    report_errs=True,
    meta_data_dicts=[],
    debug_len=0,
    remove_dir=True,
):
    """Memory-bounded version of multi_compile.

    Instead of loading every per-run xarray into RAM and calling xr.concat
    (which needs the whole stack + the merged copy resident at once), this
    streams each run into a Zarr store on disk, holding only ~one array in
    memory at a time.

    Ragged (varying-size) dimensions are handled in two ways:
      - `union_dims` are reindexed onto a shared coordinate lattice (Option B),
        so a given physical value lands in the SAME slot in every episode.
      - all other ragged dims are left-aligned NaN-padded up to their max size.

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
    union_dims = set(union_dims or ())
    zarr_path = os.path.join(master_working_dir, "compiled_combined_xr.zarr")

    if os.path.exists(zarr_path) and remove_dir:
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
    # ================= PASS 1: parallel scan for exact ragged sizes =========
    n_effective = min(debug_len, n_runs) if debug_len else n_runs
    ctx = multiprocessing.get_context("spawn")
    n_workers = min(_n_cpus(), max(1, n_effective))
    print(" N WORKERS::::" ,n_workers)

    scan_tasks = [
        (i, master_working_dir, eq_filenames[i], episode_dim,
         meta_data_dicts[i] if meta_data_dicts else None, debug, union_dims)
        for i in range(n_effective)
    ]

    print(f"[multi_compile] PASS 1: scanning {n_effective} runs on {n_workers} workers...")
    scan_results = []
    with ctx.Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_scan_run_worker, scan_tasks),
                      total=len(scan_tasks), desc="Scanning runs"):
            scan_results.append(r)
    scan_results.sort(key=lambda x: x[0])

    ok = [r for r in scan_results if r[1]]
    if not ok:
        print("[multi_compile] No successful runs to compile.")
        return None, None

    ref_sig = ok[0][4]

    # Option B: build union lattices BEFORE we pop __coords__ in the loop below
    union_coords, union_report = _compute_union_grids(scan_results, union_dims, ref_sig, episode_dim)
    if union_report:
        with open(os.path.join(master_working_dir, "union_report.pkl"), "wb") as f:
            pkl.dump(union_report, f)

    kept, broken_runs = [], []
    eq_labels, input_dict_list = [], []
    dim_min, dim_max = {}, {}
    align_ref = {}   # dim -> (first, step, is_int) from the first kept run
    align_bad = {}   # dim -> list of run indices that disagree
    for (i, s, err, sizes, sig, eq_label, input_dict) in scan_results:
        if not s:
            if err and err != "missing/none":
                print(f"[multi_compile] Run {i} scan failed -> broken.\n{err}")
                broken_runs.append(i)
            continue
        if sig != ref_sig:
            print(f"[multi_compile] Run {i} incompatible structure -> broken.")
            broken_runs.append(i)
            continue

        align = sizes.pop("__align__", {})    # strip synthetic key
        sizes.pop("__coords__", None)         # discard now; union already computed

        for d, sz in sizes.items():
            dim_min[d] = min(dim_min.get(d, sz), sz)
            dim_max[d] = max(dim_max.get(d, 0), sz)

        # --- alignment consistency check (first value + stride) -------------
        for d, (first, step, is_int) in align.items():
            if d not in align_ref:
                align_ref[d] = (first, step, is_int)
            else:
                rfirst, rstep, _ = align_ref[d]
                if not (np.isclose(first, rfirst, rtol=0, atol=1e-9)
                        and np.isclose(step, rstep, rtol=0, atol=1e-9)):
                    align_bad.setdefault(d, []).append(i)

        kept.append(i)
        eq_labels.append(eq_label)
        input_dict_list.append(input_dict)

    n_added = len(kept)
    if n_added == 0:
        print("[multi_compile] No compatible runs to compile.")
        return None, None

    max_sizes = dim_max
    ragged_dims = {d for d in dim_max if dim_min[d] < dim_max[d]}

    # Option B: union dims are sized by the lattice, and are correctly
    # registered by reindex -> treat as ragged, max size = union length.
    for d, u in union_coords.items():
        max_sizes[d] = len(u)
        ragged_dims.add(d)

    # Fail loudly if any RAGGED index coord is not left-aligned/contiguous.
    # Union dims are EXEMPT: reindex handles registration regardless of range.
    ragged_align_bad = {
        d: runs for d, runs in align_bad.items()
        if d in ragged_dims and d not in union_coords
    }
    if ragged_align_bad:
        msg = "\n".join(
            f"  dim '{d}': ref start/step={align_ref[d][:2]}, "
            f"{len(runs)} run(s) differ (e.g. {runs[:5]})"
            for d, runs in ragged_align_bad.items()
        )
        raise ValueError(
            "[multi_compile] Ragged index coords are NOT left-aligned across "
            "runs; NaN-padding onto a shared coordinate would MISREGISTER data:\n"
            f"{msg}\n"
            "See notes: add these dims to `union_dims` (reindex onto a shared "
            "lattice), store them as per-episode data vars, or realign."
        )

    print(f"[multi_compile] kept={n_added}, ragged dims={sorted(ragged_dims)}, "
          f"max_sizes={max_sizes}")

    # ================= build the pre-sized skeleton once ====================
    ref_i = kept[0]
    template, _, _ = _transform_run(
        ref_i, master_working_dir, eq_filenames[ref_i], episode_dim,
        meta_data_dicts[ref_i] if meta_data_dicts else None, debug)
    template = _normalize_for_store(template, episode_dim)
    template = _promote_and_pad(template, max_sizes, ragged_dims, union_coords)

    one = template.isel({episode_dim: 0}, drop=True)          # drop episode axis
    skeleton = one.expand_dims({episode_dim: n_added})        # broadcast (view)
    # episode chunk == 1 so parallel region writes never share a chunk:
    skeleton = zarr_chunk(skeleton, {episode_dim: 1})
    # ensure shared coords are eager numpy (so compute=False writes them now):
    for c in list(skeleton.coords):
        skeleton = skeleton.assign_coords({c: skeleton[c].compute()})

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    # compute=False -> writes metadata + coords, defers (dask) data vars:
    skeleton.to_zarr(zarr_path, mode="w", compute=False, consolidated=False)
    del template, one, skeleton
    gc.collect()

    # ================= PASS 2: parallel region-writes =======================
    write_tasks = [
        (pos, i, master_working_dir, eq_filenames[i], episode_dim,
         meta_data_dicts[i] if meta_data_dicts else None,
         zarr_path, max_sizes, ragged_dims, union_coords)
        for pos, i in enumerate(kept)
    ]

    print(f"[multi_compile] PASS 2: writing {n_added} regions on {n_workers} workers...")
    with ctx.Pool(n_workers) as pool:
        for (i, okflag, err) in tqdm(pool.imap_unordered(_write_run_worker, write_tasks),
                                     total=len(write_tasks), desc="Writing regions"):
            if not okflag:
                # NOTE: this episode's slice keeps fill values (garbage) --
                # inspect/rerun if this fires. Its label still occupies a slot.
                print(f"[multi_compile] region write FAILED run {i}:\n{err}")
                broken_runs.append(i)

    # ================= consolidate (+ optional single rechunk) ==============
    ds_store = xr.open_zarr(zarr_path, consolidated=False)
    if mb_per_chunk is not None:
        bytes_per_episode = ds_store.isel({episode_dim: 0}).nbytes
        mean_mb = bytes_per_episode / (1024 * 1024)
        episodes_per_chunk = min(max(1, int(mb_per_chunk / max(mean_mb, 1e-9))), n_added)
        if episodes_per_chunk > 1:
            chunk_spec = {episode_dim: episodes_per_chunk} | {
                k: ds_store.sizes[k] for k in ds_store.dims if k != episode_dim}
            ds_store = zarr_chunk(ds_store, chunk_spec)
            tmp_path = zarr_path + ".tmp"
            ds_store.to_zarr(tmp_path, mode="w", consolidated=True)
            shutil.rmtree(zarr_path)
            os.rename(tmp_path, zarr_path)
    zarr.consolidate_metadata(zarr_path)

    # ================= reopen lazily + attach labels ========================
    compiled_xr = xr.open_zarr(zarr_path, consolidated=True)  # whole dataset, lazy
    assert len(eq_labels) == compiled_xr.sizes[episode_dim], (
        f"label/run_idx length mismatch: {len(eq_labels)} vs "
        f"{compiled_xr.sizes[episode_dim]}")

    # --- store eq_labels as their OWN variable in the Zarr store ------------
    # We do NOT touch the existing (integer) run_idx index array. Instead we add
    # a separate string variable 'eq_label' along the run_idx dimension so the
    # labels persist on disk and can be recovered on reload without a pickle.
    eq_labels_arr = np.asarray(eq_labels, dtype="U")
    xr.Dataset({"eq_label": (episode_dim, eq_labels_arr)}).to_zarr(
        zarr_path, mode="a", consolidated=True
    )
    print(f"[multi_compile] Wrote 'eq_label' variable "
          f"({eq_labels_arr.size} labels) to {zarr_path}")

    # reopen so the returned object carries the new variable, then attach
    # eq_label as a non-dimension coordinate (run_idx stays integer everywhere)
    compiled_xr = xr.open_zarr(zarr_path, consolidated=True)
    compiled_xr = compiled_xr.assign_coords(
        eq_label=(episode_dim, eq_labels_arr)
    )

    # --- scalar inputs (small, fine in memory) ------------------------------
    scalar_keys = [
        key for key in input_dict_list[0]
        if isinstance(input_dict_list[0][key],
                      (int, float, bool, str, np.integer, np.floating))
    ]
    input_data_vars = {}
    for key in scalar_keys:
        vals = [d.get(key, np.nan) for d in input_dict_list]
        input_data_vars[key] = (episode_dim, vals)
    # keep run_idx integer; carry the labels alongside as 'eq_label'
    compiled_inputs_xr = xr.Dataset(input_data_vars)
    compiled_inputs_xr = compiled_inputs_xr.assign_coords(
        eq_label=(episode_dim, eq_labels_arr)
    )
    inputs_zarr_path = os.path.join(master_working_dir, "compiled_inputs_xr.zarr")
    if debug_len == 0:
        compiled_inputs_xr.to_zarr(inputs_zarr_path, mode="w", consolidated=True)
        print(f"[multi_compile] Wrote compiled_inputs_xr to {inputs_zarr_path}")

    if broken_runs:
        broken_runs = sorted(set(broken_runs))
        print(f"[multi_compile] {len(broken_runs)} runs excluded: {broken_runs}")
        if debug_len == 0:
            with open(os.path.join(master_working_dir, "broken_runs.pkl"), "wb") as f:
                pkl.dump(broken_runs, f)

    # Persist eq_labels as a pickle fallback too (belt-and-braces: the Zarr now
    # carries them via 'eq_label', but eq_labels.pkl remains a cheap,
    # engine-independent backup).
    with open(os.path.join(master_working_dir, "eq_labels.pkl"), "wb") as f:
        pkl.dump(eq_labels, f)

    debug_suffix = (f" (debug_len={debug_len}, processed {n_effective}/{n_runs})"
                    if debug_len else "")
    print(f"[multi_compile] Compiled {n_added} runs into {zarr_path}{debug_suffix}")
    return compiled_xr, compiled_inputs_xr

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

def build_tensorized_dataset(  # noqa: PLR0912
    process_fn: Callable[[Any], xr.Dataset],
    identifiers: Iterable[Any],
    zarr_path: os.PathLike,
    episode_dim: str,
    time_dim: str | None = None,
    extend_existing: bool = False,
    episodes_per_chunk: int | None = None,
    mb_per_chunk: int | None = 10,
) -> xr.Dataset:
    """Build a tensorized multi-episode dataset that can then be used for training or evaluation.
    The user provides a function that processes data for a single episode, which returns an xarray Dataset for a single episode. This function will then build up the multi-episode dataset. This function was built with the intention of only needing to load a single episode at a time, allowing us to build up a dataset that is too large to fit in memory.

    Args:
        process_fn (Callable[[Any], xr.Dataset]): A function that takes in a episode identifier (e.g. a file path, a pulse number, or even a xarray Dataset) and returns an xarray Dataset for a single episode.
        identifiers (Iterable[Any]): An iterable that will be looped over, with each item being passed to process_fn to build up the dataset.
        zarr_path (os.PathLike): path to the zarr store where the dataset will be saved.
        episode_dim (str): The name of the episode dimension in the dataset.
        time_dim (Optional[str], optional): The name of the time dimension in the dataset. If None, no dimension receives the special time-size tracking optimization and all non-episode dimensions are reconciled uniformly. Defaults to None.
        extend_existing (bool, optional): If zarr_path already exists and this is true, we will try to extend the existing zarr_path. Defaults to False.
        episodes_per_chunk (Optional[int], optional): The number of episodes per chunk in storage. If this is not None, then this function will rechunk the built Zarr store once all the files are added. Defaults to None.
        mb_per_chunk (Optional[int], optional): If specified, the resulting Zarr stored will be chunked along the episodes dimension with size max(1, int(mb_per_chunk / mean_mb_per_episode)). Defaults to 10.

    Raises:
        ValueError: If zarr_path already exists and extend_existing is False, this function will raise an error.

    Returns:
        xr.Dataset: The xarray Dataset that was built up.
    """

    if not zarr_path.endswith(".zarr"):
        raise ValueError(f"Provided zarr_path must end with .zarr, but got {zarr_path}")

    if os.path.exists(zarr_path) and extend_existing is False:
        raise ValueError(
            f"Zarr store at {zarr_path} already exists and extend_existing is False. Please remove it or set extend_existing to True."
        )
    if mb_per_chunk is not None and episodes_per_chunk is not None:
        raise ValueError("Please specify either mb_per_chunk or episodes_per_chunk, not both.")

    def get_and_preprocess(identifier: Any) -> xr.Dataset | None:
        try:
            return process_fn(identifier)
        except Exception as e:
            # Warn the user.
            loguru.logger.warning(f"Error processing {identifier}: {e}")
            return None

    atleast_one_success = False  # Keep track of whether at least one dataset was successfully processed.
    store_time_dim_size = None  # Keep track of the size of the time dimension in the zarr store.

    # Iterate over the identifiers and process them one by one to build the dataset.
    for i, it in enumerate(tqdm(identifiers, desc="Building the dataset...")):
        ds = get_and_preprocess(it)
        if ds is not None:
            success = add_to_zarr_store(ds, zarr_path, episode_dim, time_dim=time_dim, store_time_dim_size=store_time_dim_size)

            # Only track the time dimension size if a time_dim was provided.
            if success and time_dim is not None:
                if not store_time_dim_size:
                    store_time_dim_size = xr.open_zarr(zarr_path, consolidated=True).sizes[time_dim]
                else:
                    store_time_dim_size = max(store_time_dim_size, ds.sizes[time_dim])

            atleast_one_success = True if success else atleast_one_success

        del ds
        if (i > 0) and (i % GC_INTERVAL == 0):
            loguru.logger.debug(f"Forcing garbage collection after {i} processed shots")
            jax.clear_caches()
            gc.collect()

    # Exit if no datasets were successfully processed.
    if not atleast_one_success:
        loguru.logger.warning(f"No successful datasets were processed. Returning an empty dataset for {zarr_path}.")
        if extend_existing and os.path.exists(zarr_path):
            return xr.open_zarr(zarr_path, consolidated=None)  # Allow for both consolidated and non-consolidated zarr stores.
        else:
            return xr.Dataset()

    # Now that the store has been built, we can consolidate the metadata and rechunk if necessary.
    loguru.logger.info(f"Successfully processed at least some of the files. Chunking and consolidating metadata for {zarr_path}.")
    ds = xr.open_zarr(zarr_path, consolidated=True)

    if mb_per_chunk is not None:
        # Compute the number of episodes per chunk based on the average size of per-episode datasets.
        n_episodes = ds.sizes[episode_dim]
        bytes_per_episode = ds.isel({episode_dim: 0}).nbytes
        mean_mb_per_episode = bytes_per_episode / (1024 * 1024)
        episodes_per_chunk = min(max(1, int(mb_per_chunk / mean_mb_per_episode)), n_episodes)
        loguru.logger.info(
            f"Determined {episodes_per_chunk} episodes per chunk based on provided mb_per_chunk={mb_per_chunk} and the computed mean_mb_per_episode={mean_mb_per_episode:.2f}."
        )

    if episodes_per_chunk is not None:
        loguru.logger.info(f"Chunking the dataset with {episodes_per_chunk} episodes per chunk.")

        # We essentially only want chunking across episodes. So other dimensions should be covered by a single chunk.
        chunk_spec = {episode_dim: episodes_per_chunk} | {k: ds.sizes[k] for k in ds.dims if k != episode_dim}

        ds = zarr_chunk(ds, chunk_spec=chunk_spec)

        # Save the chunked dataset to a temporary path and then rename it to the final zarr path.
        tmp_path = zarr_path + ".tmp"
        ds.to_zarr(tmp_path, mode="w", consolidated=True)
        shutil.rmtree(zarr_path)  # Remove the old zarr store if it exists.
        os.rename(tmp_path, zarr_path)

    loguru.logger.info(f"Successfully processed at least some of the files. Consolidating metadata for {zarr_path}.")
    zarr.consolidate_metadata(zarr_path)
    return xr.open_zarr(zarr_path, consolidated=True)


def extend_zarr_along_dim(zarr_path: os.PathLike, dim: str, n_extend: int) -> None:
    """Extend an existing zarr store along a specified dimension by padding it with nans."""
    ds = xr.open_zarr(zarr_path, consolidated=True)
    ds = ds.pad({dim: (0, n_extend)})
    ds_padding = ds.isel({dim: slice(-n_extend, None)})
    ds_padding = ds_padding.compute()  # Pull dataset into memory to avoid dask/chunking issues
    ds_padding.to_zarr(zarr_path, mode="a-", append_dim=dim, consolidated=True, align_chunks=True)


def add_to_zarr_store(  # noqa: PLR0912
    ds: xr.Dataset, zarr_path: os.PathLike, episode_dim: str, time_dim: str | None = None, store_time_dim_size: int | None = None
) -> bool:
    """Helper function to add a single xarray Dataset to a zarr store. Requires that the zarr store either doesn't exist or already contains all the dimensions in the provided Dataset."""

    # We need to reset all the non-index coordinates to make sure they get vary across episodes.
    print(f"[pre_reset]")
    dims_str = "\n    ".join(f"{k}: {v}" for k, v in ds.sizes.items())
    print(f"  Dimensions:\n    {dims_str}")
    print()
    coords_str = "\n    ".join(ds.coords)
    print(f"  Coordinates:\n    {coords_str}")
    print()
    vars_str = "\n    ".join(ds.data_vars)
    print(f"  Variables:\n    {vars_str}")
    ds = ds.reset_coords()

    if time_dim is not None and time_dim in ds.coords:
        ds = ds.drop_vars(time_dim)

    # If the episode dimension is not present, expand the dataset to include it.
    if episode_dim not in ds.dims:
        ds = ds.set_coords(episode_dim)
        ds = ds.expand_dims(episode_dim)

    if ds.sizes[episode_dim] > 1:
        raise ValueError(
            f"The dataset has more than one episode ({ds.sizes[episode_dim]}). "
            f"Please ensure that the dataset is for a single episode before adding it to the Zarr store."
        )

    # If only some of the variables have the episode dimension, we can't just expand the whole dataset.
    # We need to ensure that all variables have the episode dimension.
    for var in ds.data_vars:
        if episode_dim not in ds[var].dims:
            # If the variable does not have the episode dimension, we need to add it.
            ds[var] = ds[var].expand_dims(episode_dim)

    if not os.path.exists(zarr_path):
        loguru.logger.info(f"Zarr store at {zarr_path} does not exist. Creating a new one.")
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
        return True
    else:
        ds_store = xr.open_zarr(zarr_path, consolidated=True)

        # Handle dimension size changes for all dimensions except episode_dim
        pad_dims = {}
        extend_dims = {}

        # Check all other dimensions that exist in both datasets
        considered_dims = set(ds.dims) - {episode_dim}
        for dim in considered_dims:
            if dim in ds_store.dims:
                # Use the tracked time-dim size only when a time_dim was supplied and matches.
                if time_dim is not None and dim == time_dim and store_time_dim_size is not None:
                    store_dim_size = store_time_dim_size
                else:
                    store_dim_size = ds_store.sizes[dim]

                ds_dim_size = ds.sizes[dim]

                if ds_dim_size < store_dim_size:
                    pad_dims[dim] = (0, store_dim_size - ds_dim_size)
                elif ds_dim_size > store_dim_size:
                    extend_dims[dim] = ds_dim_size - store_dim_size
            else:
                raise ValueError(f"Dimension {dim} is not present in the zarr store.")

        # Extend the zarr store for any dimensions that need to grow
        for dim, n_extend in extend_dims.items():
            extend_zarr_along_dim(zarr_path, dim, n_extend)

        # Pad the dataset for any dimensions that are too small
        if pad_dims:
            ds = ds.pad(pad_dims)

        # Append the new dataset to the existing zarr store.
        # a- means we append only to variables that have episode_dim.
        ds.to_zarr(zarr_path, mode="a-", append_dim=episode_dim, consolidated=True)
        # According to xarray docs, stale consolidated metadata can cause issues.
        # Should re-consolidate between each addition.
        zarr.consolidate_metadata(zarr_path)
        return True


def zarr_chunk(ds: xr.Dataset, chunk_spec: dict) -> xr.Dataset:
    """Chunk the dataset according to the provided chunk specification.
    For some reason we need to delete the chunks encoding from the dataset before saving it to zarr.
    See the stackoverflow issue:
        https://stackoverflow.com/questions/67476513/zarr-not-respecting-chunk-size-from-xarray-and-reverting-to-original-chunk-size
    """
    ds = ds.chunk(chunk_spec)
    for var in ds.data_vars:
        if "chunks" in ds[var].encoding:
            del ds[var].encoding["chunks"]
    for coord_name in ds.coords:  # Also check coords if they are being saved as arrays
        if hasattr(ds[coord_name], "encoding") and "chunks" in ds[coord_name].encoding:
            del ds[coord_name].encoding["chunks"]
    return ds