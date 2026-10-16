import operator
import xarray as xr
import os
import operator
import time as _time
import numpy as np
import matplotlib.pyplot as plt
from tearing_physics_suite.tearing_physics_suite import add_unique_label
import sys
from matplotlib.ticker import MultipleLocator, NullFormatter, AutoMinorLocator
home_dir = os.environ['TPSHOME']

import operator
import numpy as np
import xarray as xr
import os

# Map operator strings to comparison functions
_OPS = {
    '<':  operator.lt,
    '<=': operator.le,
    '>':  operator.gt,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne,
}

# Variables that are cheap to co-load and are frequently extracted one after
# another. Requesting any one of them will load/cache ALL THREE, and the cache
# key is normalized so switching between them is a cache HIT (no recompute).
HIGH_USE_SURF_VARS = ("Delta_prime_surf", "w_marg_surf", "dwdtau_max_surf")

"""
SHARED_WHERE = {
    'psi_n_rational':                 [{'op': '<', 'val': 0.8}],
    'Delta_prime_ImRatio':            [{'op': '<', 'val': 1e-2}],
    'Delta_prime_reldiff_across_GPEC':[{'op': '<', 'val': 0.2}],
    'nzero': 0,
    'r_occ': 0,

    # cross-variable OR: keep rows where var_x < 2 OR var_y < 2
    '__cond_xy__': {
        'or': [
            {'var': 'var_x', 'op': '<', 'val': 2},
            {'var': 'var_y', 'op': '<', 'val': 2},
        ]
    },
}

'__cond__': {
        'and': [
            {'or': [
                {'var': 'var_x', 'op': '<', 'val': 2},
                {'var': 'var_y', 'op': '<', 'val': 2},
            ]},
            {'var': 'var_z', 'op': '>=', 'val': 0},
        ]
    },
"""

def _eval_where_expr(ds, expr):
    """
    Recursively turn a where-expression tree into a boolean DataArray mask.

    Leaf:   {'var': 'var_x', 'op': '<', 'val': 2}
            {'var': 'var_x', 'val': 0}          # implicit ==
    Branch: {'or':  [expr, expr, ...]}
            {'and': [expr, expr, ...]}
    """
    if isinstance(expr, dict) and ('or' in expr or 'and' in expr):
        combine = 'or' if 'or' in expr else 'and'
        subs = expr[combine]
        mask = None
        for s in subs:
            sm = _eval_where_expr(ds, s)
            if mask is None:
                mask = sm
            else:
                mask = (mask | sm) if combine == 'or' else (mask & sm)
        return mask
    # leaf node
    if isinstance(expr, dict) and 'var' in expr:
        var = expr['var']
        if 'op' in expr:
            return _OPS[expr['op']](ds[var], expr['val'])
        return ds[var] == expr['val']
    raise ValueError(f"Unrecognized where-expression: {expr!r}")

def _collect_expr_vars(expr, out):
    """Gather all variable names referenced anywhere in an expression tree."""
    if isinstance(expr, dict):
        if 'or' in expr:
            for s in expr['or']:
                _collect_expr_vars(s, out)
        elif 'and' in expr:
            for s in expr['and']:
                _collect_expr_vars(s, out)
        elif 'var' in expr:
            out.add(expr['var'])

def _where_referenced_vars(wod):
    """
    All variable names a where_option_dict actually touches, including
    variables buried inside logical (__*) expression trees. Used for both
    the zarr load subset and the cache signature.
    """
    if not wod:
        return []
    vars_ = set()
    for key, cond in wod.items():
        if isinstance(key, str) and key.startswith('__'):
            exprs = cond if isinstance(cond, list) else [cond]
            for expr in exprs:
                _collect_expr_vars(expr, vars_)
        else:
            vars_.add(key)
    return sorted(vars_)

def dist_extractor(master_working_dir, output_dir, var_name,
                   sel_option_dict=None,
                   isel_option_dict=None,
                   where_option_dict=None,
                   one_surf=True,
                   unique_surf_filter=True,
                   debug=False,
                   debug_n_per_time=None,   # randomly sample this many points per time slice
                   sample_dim='run_idx',    # dimension to subsample / split along
                   random_seed=None,        # reproducibility for the random sampling
                   timing=True,             # print timing diagnostics
                   load_into_memory=True,   # eagerly load the subset into RAM (big speed win)
                   use_cache=True,          # reuse a previously saved xri_small if inputs match
                   cache_dir=None):         # where to store cached small subsets

    import json, hashlib
    from time import perf_counter

    zarr_path = os.path.join(master_working_dir, "compiled_combined_xr.zarr")
    xri = xr.open_zarr(zarr_path, consolidated=True)
    print(xri)
    import sys
    sys.exit()

    os.makedirs(output_dir, exist_ok=True)
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "xri_small_cache")
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")

    rng = np.random.default_rng(random_seed)
    # cumulative timers across all slices
    t_totals = {
        'split_isel': 0.0, 'ida_time_compute': 0.0, 'sel': 0.0, 'isel': 0.0,
        'add_unique': 0.0, 'where': 0.0, 'sample': 0.0, 'extract_values': 0.0,
    }
    def _tic():
        return perf_counter()
    def _toc(key, t0, label=None, indent=1):
        dt = perf_counter() - t0
        t_totals[key] += dt
        if timing:
            print(f"[timing]{'  '*indent}{label or key}: {dt:.3f}s", flush=True)
        return dt
    t_start = _tic()

    # -----------------------------------------------------------------
    # CACHE KEY: everything that affects the DEFINITION of xri_small.
    # NOTE: only the where_option_dict *referenced variables* matter
    # (values are applied later, per-slice), and sel/isel/plot settings
    # do NOT affect the load. Logical (__*) trees are dug through so the
    # right variables are loaded and the signature stays unique.
    # -----------------------------------------------------------------

    # If var_name is one of the high-use surf vars, add the other two, and collapse
    # var_name into a shared group token.
    in_high_use_group = var_name in HIGH_USE_SURF_VARS
    where_vars = _where_referenced_vars(where_option_dict)
    if in_high_use_group:
        where_vars = sorted(set(where_vars) | set(HIGH_USE_SURF_VARS))

    signature = {
        'zarr_path':   os.path.abspath(zarr_path),
        'var_name':    'HIGH_USE_SURF_GROUP' if in_high_use_group else var_name,
        'where_keys':  where_vars,
        'sample_dim':  sample_dim,
        'schema_ver':  3,   # bumped: signature now folds in the high-use group
    }
    sig_str = json.dumps(signature, sort_keys=True, default=str)
    cache_key = hashlib.md5(sig_str.encode()).hexdigest()[:16]
    cache_file = os.path.join(cache_dir, f"xri_small_{cache_key}.nc")

    def _read_manifest():
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_manifest(m):
        try:
            with open(manifest_path, 'w') as f:
                json.dump(m, f, indent=2)
        except Exception as e:
            print(f"[cache] warning: could not write manifest: {e}", flush=True)

    # -----------------------------------------------------------------
    # RESOLVE WHICH CACHE FILE TO READ.
    #   1) exact signature match (fast path, same as before), else
    #   2) SUPERSET match: any previously-built subset whose where_keys
    #      is a superset of the keys we need. Its stored variables are a
    #      superset of ours, so it can be reused verbatim. We pick the
    #      *tightest* superset (fewest extra keys) to minimize wasted RAM.
    # -----------------------------------------------------------------
    def _find_superset_cache():
        """Return (key, file, sig) of the tightest usable superset cache, or None."""
        manifest = _read_manifest()
        want_keys = set(where_vars)
        best = None  # (n_extra, key, file, sig)
        for k, sig in manifest.items():
            if not isinstance(sig, dict):
                continue
            # Everything except where_keys must match exactly: these all
            # change *which rows/vars* get loaded or how they're split.
            if (sig.get('zarr_path')  != signature['zarr_path'] or
                sig.get('var_name')   != signature['var_name'] or
                sig.get('sample_dim') != signature['sample_dim'] or
                sig.get('schema_ver') != signature['schema_ver']):
                continue
            cached_keys = set(sig.get('where_keys', []))
            # Reusable iff the cache holds (at least) every var we reference.
            if not want_keys.issubset(cached_keys):
                continue
            cand_file = os.path.join(cache_dir, f"xri_small_{k}.nc")
            if not os.path.exists(cand_file):
                continue
            n_extra = len(cached_keys) - len(want_keys)
            if best is None or n_extra < best[0]:
                best = (n_extra, k, cand_file, sig)
        if best is None:
            return None
        return best[1], best[2], best[3]

    xri_small = None
    read_cache_key = cache_key
    read_cache_file = cache_file

    cache_hit = use_cache and os.path.exists(cache_file)
    if use_cache and not cache_hit:
        superset = _find_superset_cache()
        if superset is not None:
            read_cache_key, read_cache_file, sup_sig = superset
            cache_hit = True
            if timing:
                extra = sorted(set(sup_sig.get('where_keys', [])) - set(where_vars))
                print(f"[timing] SUPERSET CACHE HIT ({read_cache_key}): reusing "
                      f"cache built with extra where vars {extra} "
                      f"(needed {where_vars})", flush=True)

    # -----------------------------------------------------------------
    # CACHE HIT: skip the big zarr entirely, just read the small file.
    # -----------------------------------------------------------------
    if cache_hit:
        t0 = _tic()
        try:
            xri_small = xr.open_dataset(read_cache_file).load()
            if timing:
                print(f"[timing] CACHE HIT ({read_cache_key}): loaded small subset "
                      f"from {read_cache_file} in {perf_counter()-t0:.3f}s "
                      f"({xri_small.nbytes/1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"[cache] warning: failed to read cache {read_cache_file} "
                  f"({e}); rebuilding.", flush=True)
            xri_small = None
            cache_hit = False

    # -----------------------------------------------------------------
    # CACHE MISS: open the big zarr, build the minimal subset, load, save.
    # -----------------------------------------------------------------
    if xri_small is None:
        if timing and use_cache:
            print(f"[timing] CACHE MISS ({cache_key}): building subset from zarr",
                  flush=True)

        xri = xr.open_zarr(zarr_path, consolidated=True)

        t0 = _tic()
        needed = {var_name, 'IDA_time_idx', 'IDA_time'}
        if in_high_use_group:
            needed.update(HIGH_USE_SURF_VARS)   # co-load the other two
        if where_option_dict:
            needed.update(_where_referenced_vars(where_option_dict))
        for dim in ("r", "r_prime"):
            needed.update([f"{dim}_value", f"{dim}_occ"])
            if f"{dim}_unique" in xri.data_vars:
                needed.add(f"{dim}_unique")
        present = [v for v in needed if v in xri.data_vars]
        xri_small = xri[present]
        if sample_dim in xri_small.dims:
            xri_small = xri_small.chunk({sample_dim: -1})
        if timing:
            print(f"[timing] build subset ({len(present)} vars) + rechunk: "
                  f"{perf_counter()-t0:.3f}s", flush=True)

        # -------------------------------------------------------------
        # LOAD THE SUBSET INTO MEMORY ONCE
        # -------------------------------------------------------------
        if load_into_memory:
            t0 = _tic()
            try:
                print(f"[timing] subset size: {xri_small.nbytes/1e9:.3f} GB",
                      flush=True)
            except Exception:
                pass
            xri_small = xri_small.load()
            if timing:
                print(f"[timing] load subset into memory: "
                      f"{perf_counter()-t0:.3f}s", flush=True)

            # ---- save to cache so the next matching call is instant ----
            if use_cache:
                t0 = _tic()
                try:
                    tmp = cache_file + ".tmp"
                    xri_small.to_netcdf(tmp)
                    os.replace(tmp, cache_file)  # atomic-ish
                    m = _read_manifest()
                    m[cache_key] = signature
                    _write_manifest(m)
                    if timing:
                        print(f"[timing] wrote cache ({cache_key}) -> {cache_file} "
                              f"in {perf_counter()-t0:.3f}s "
                              f"({xri_small.nbytes/1e6:.1f} MB)", flush=True)
                except Exception as e:
                    print(f"[cache] warning: failed to write cache "
                          f"{cache_file} ({e}); continuing without caching.",
                          flush=True)

    # -----------------------------------------------------------------
    # Precompute the split index AND IDA_time ONCE
    # -----------------------------------------------------------------
    t0 = _tic()
    tidx_da = xri_small.IDA_time_idx
    tidx_dim = tidx_da.dims[0]
    tidx_arr = tidx_da.compute().values.ravel()
    ida_time_arr = xri_small.IDA_time.compute().values.ravel()
    if timing:
        print(f"[timing] IDA_time_idx + IDA_time compute (split index): "
              f"{perf_counter()-t0:.3f}s  (len={tidx_arr.size})", flush=True)

    # ... everything below here is unchanged (the per-slice loop, etc.) ...
    time_idxs = sorted(set(tidx_arr.tolist()))
    times = []
    value_matrix = []
    xri_list = []
    n_skipped = 0
    if timing:
        print(f"[timing] beginning loop over {len(time_idxs)} time indices",
              flush=True)
    for i_ti, time_idx in enumerate(time_idxs):
        t_slice = _tic()
        if timing:
            print(f"[timing] --- slice {i_ti+1}/{len(time_idxs)} "
                  f"(time_idx={time_idx}) ---", flush=True)
        t0 = _tic()
        pos = np.where(tidx_arr == time_idx)[0]
        xri_idx = xri_small.isel({tidx_dim: pos})
        _toc('split_isel', t0, 'split_isel')
        # --- IDA_time now taken from precomputed numpy array (no per-slice compute) ---
        t0 = _tic()
        slice_ida = ida_time_arr[pos]
        unique_times = np.unique(slice_ida[np.isfinite(slice_ida)])
        _toc('ida_time_compute', t0, 'ida_time_compute')
        if len(unique_times) != 1:
            print(f"IDA_time not unique for time_idx={time_idx}: {unique_times}")
            raise ValueError("F9ff")
        time_val = float(unique_times[0])
        try:
            # --- .sel downselect ---
            t0 = _tic()
            xri_idx_sel=xri_idx
            if debug: print(xri_idx_sel.coords)
            if sel_option_dict:
                for key, val in sel_option_dict.items():
                    xri_idx_sel = xri_idx_sel.sel({key: val})
            _toc('sel', t0, 'sel')
            # --- .isel downselect ---
            t0 = _tic()
            xri_idx_isel = xri_idx_sel
            if debug: print(xri_idx_isel.coords)
            if isel_option_dict:
                for key, val in isel_option_dict.items():
                    xri_idx_isel = xri_idx_isel.isel({key: val})
            if debug: print(xri_idx_isel.coords)
            _toc('isel', t0, 'isel')
            t0 = _tic()
            try:
                if 'r_unique' not in xri_idx_isel.data_vars:
                    xri_idx_isel = add_unique_label(xri_idx_isel)
            except:
                raise NotImplementedError("BRUH")
            _toc('add_unique', t0, 'add_unique')
            # --- .where downselect ---
            t0 = _tic()
            xri_idx_where = xri_idx_isel

            def _apply_mask(ds, mask):
                # fast path: a mask that varies only along the sample dim can
                # be applied with positional isel (much cheaper than where+drop,
                # which reindexes every variable).
                if set(mask.dims) == {sample_dim}:
                    keep = np.where(np.asarray(mask.values))[0]
                    if debug: print(f"[where]   (isel fast-path) keep={keep.size}")
                    return ds.isel({sample_dim: keep})
                mask = mask.compute()
                if debug: print(f"[where]   mask sum={int(mask.sum())}, "
                                f"size before drop={ds.sizes}")
                return ds.where(mask, drop=True)

            if where_option_dict:
                if debug: print(f"[where] starting size: {xri_idx_where.sizes}")
                for key, cond in where_option_dict.items():

                    # ---- logical expression entry (cross-variable AND/OR) ----
                    if isinstance(key, str) and key.startswith('__'):
                        exprs = cond if isinstance(cond, list) else [cond]
                        for expr in exprs:
                            if debug: print(f"[where] logical {key!r}: {expr!r}")
                            mask = _eval_where_expr(xri_idx_where, expr)
                            xri_idx_where = _apply_mask(xri_idx_where, mask)
                            if debug: print(f"[where]   size after={xri_idx_where.sizes}")
                        continue

                    # ---- ordinary single-variable entry (unchanged behavior) ----
                    conds = cond if isinstance(cond, list) else [cond]
                    for c in conds:
                        if isinstance(c, dict):
                            if debug: print(f"[where] key={key!r}, op={c['op']!r}, val={c['val']!r}")
                            mask = _OPS[c['op']](xri_idx_where[key], c['val'])
                        else:
                            if debug: print(f"[where] key={key!r}, == {c!r}")
                            mask = xri_idx_where[key] == c
                        xri_idx_where = _apply_mask(xri_idx_where, mask)
                        if debug: print(f"[where]   size after={xri_idx_where.sizes}")
            _toc('where', t0, 'where')
            if debug:
                import sys
                sys.exit()
        except (KeyError, IndexError) as e:
            print(f"[skip] time_idx={time_idx}: label/index not found -> {e}")
            n_skipped += 1
            continue
        # --- extract the variable, subsampling BEFORE compute if debugging ---
        t0 = _tic()
        da_out = xri_idx_where[var_name]
        if debug_n_per_time is not None and sample_dim in da_out.dims:
            n = da_out.sizes[sample_dim]
            if n > debug_n_per_time:
                idx = np.sort(rng.choice(n, size=int(debug_n_per_time), replace=False))
                da_out = da_out.isel({sample_dim: idx})
                xri_idx_where = xri_idx_where.isel({sample_dim: idx})
        _toc('sample', t0, 'sample')
        t0 = _tic()
        vals = da_out.values.tolist()
        _toc('extract_values', t0, 'extract_values')
        # Only append time + values if selection succeeded
        times.append(time_val)
        value_matrix.append(vals)
        xri_list.append(xri_idx_where)
        if timing:
            print(f"[timing]   slice total: {perf_counter()-t_slice:.3f}s",
                  flush=True)
    # --- all-skipped safeguard ---
    if time_idxs and n_skipped == len(time_idxs):
        print()
        raise ValueError(
            f"All {len(time_idxs)} time indices were skipped for var_name="
            f"'{var_name}'. This usually indicates a mistyped coordinate name "
            f"or a label/index that exists in no time slice.\n"
            f"  sel_option_dict={sel_option_dict}\n"
            f"  isel_option_dict={isel_option_dict}\n"
            f"  where_option_dict={where_option_dict}"
        )
    if n_skipped:
        print(f"[info] Completed with {n_skipped}/{len(time_idxs)} time indices skipped.")
    if timing:
        print(f"[timing] ===== cumulative totals across all slices =====",
              flush=True)
        for k, v in sorted(t_totals.items(), key=lambda kv: -kv[1]):
            print(f"[timing]   {k:<18s}: {v:.3f}s", flush=True)
        print(f"[timing]   {'TOTAL wall':<18s}: {perf_counter()-t_start:.3f}s",
              flush=True)
    return dict(
        output_dir=output_dir,
        var_name=var_name,
        times=times,
        value_matrix=value_matrix,
        xri_list=xri_list,
        sel_option_dict=sel_option_dict,
        isel_option_dict=isel_option_dict,
        where_option_dict=where_option_dict,
    )

def _converge_hist_range_single(vals, lo, hi, n_bins, hist_min_count, ylog):
    """Converge [lo, hi] onto the bins meeting hist_min_count for ONE slice.

    Returns the tight (unbuffered) converged (lo, hi), or None if this slice
    has no qualifying data.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if ylog:
        vals = vals[vals > 0]
    if vals.size == 0:
        return None

    while True:
        in_range = vals[(vals >= lo) & (vals <= hi)]
        if in_range.size == 0:
            return None

        if ylog:
            edges = np.logspace(np.log10(max(lo, 1e-300)), np.log10(hi), n_bins + 1)
        else:
            edges = np.linspace(lo, hi, n_bins + 1)

        # >>> NEW: guard against float round-trip (esp. logspace) pushing the
        # edges inside the data, which would silently drop the extreme points
        # and bias the low end downward
        edges[0]  = min(edges[0],  in_range.min())
        edges[-1] = max(edges[-1], in_range.max())
        # <<< END NEW

        counts, _ = np.histogram(in_range, bins=edges)
        good = np.where(counts >= hist_min_count)[0]
        if good.size == 0:
            return None

        first_bin, last_bin = good[0], good[-1]
        # converged: the outermost qualifying bins are already the edge bins
        if first_bin == 0 and last_bin == n_bins - 1:
            break

        # restrict to the qualifying bin span, then anchor to the actual points
        lo_edge, hi_edge = edges[first_bin], edges[last_bin + 1]
        kept = in_range[(in_range >= lo_edge) & (in_range <= hi_edge)]
        if kept.size == 0:
            return None
        new_lo, new_hi = kept.min(), kept.max()

        if np.isclose(new_lo, lo) and np.isclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi

    return lo, hi

def _converge_hist_range(y_per_time, lo, hi, n_bins, hist_min_count, range_buffer, ylog):
    best_lo, best_hi = None, None
    for vals in y_per_time:
        res = _converge_hist_range_single(vals, lo, hi, n_bins,
                                        hist_min_count, ylog)
        if res is None:
            continue
        s_lo, s_hi = res
        best_lo = s_lo if best_lo is None else min(best_lo, s_lo)
        best_hi = s_hi if best_hi is None else max(best_hi, s_hi)

    if best_lo is None:          # no slice converged
        c_lo, c_hi = lo, hi
    else:
        c_lo, c_hi = best_lo, best_hi

    if c_lo < lo:
        c_lo = lo
    if c_hi > hi:
        c_hi = hi
    return c_lo, c_hi

def dist_plotter(output_dir=None, var_name=None,
                 times=None, value_matrix=None, xri_list=None,
                 y_range=[None, None],
                 x_range=[None, None],  # x-axis limits; None => auto from times
                 n_bins=40,
                 hist_frac=0.65,   # max histogram width as fraction of time spacing
                 hist_offset=0.2,
                 angle_deg=0.0,   # shear angle for the 'angled' histogram
                 point_size=8,
                 save_name=None,
                 percent_bounds=[0, 100],
                 range_buffer=0.05,
                 sel_option_dict=None,
                 isel_option_dict=None,
                 where_option_dict=None,
                 ylog=False,
                 sample_size=1000,
                 hist_min_count=1,        # bins with fewer counts than this are not drawn
                 iterative_range=False,   # iteratively shrink range to bins meeting hist_min_count
                 y_plot_name=None,        # y-axis display name (latex ok); defaults to var_name
                 y_unit=None,             # y-axis unit string, appended after y_plot_name
                 x_plot_name='time',      # x-axis display name
                 x_unit='[ms]',           # x-axis unit string
                 shotnum=None,            # shot number for title
                 title_fontsize=None,     # font size for the title
                 xlabel_fontsize=None,    # font size for the x-axis label
                 ylabel_fontsize=None,    # font size for the y-axis label
                 debug_n_per_time=None,   # randomly sample this many points per time slice
                 include_scatter=True,
                 figsize=(12, 6),
                 time_id=False,
                 special_id='',
                 random_seed=None,
                 renorm=False, # Hists at different timeslices show true counts 
                 ):       # reproducibility for the random sampling
    if ylog==False and (var_name=='w_marg_surf' or 'w_marg' in var_name):
        ylog=True
    rng = np.random.default_rng(random_seed)
    times = np.asarray(times, dtype=float)
    # --- collapse each time's values into a clean 1D array of y-values ---
    y_per_time = []
    for vals in value_matrix:
        arr = np.atleast_1d(np.asarray(vals, dtype=float)).ravel()
        arr = arr[np.isfinite(arr)]
        # debug: randomly subsample points from this time slice
        if debug_n_per_time is not None and arr.size > debug_n_per_time:
            idx = rng.choice(arr.size, size=int(debug_n_per_time), replace=False)
            arr = arr[idx]
        if len(arr) > sample_size:
            arr = arr.flatten()[:sample_size]
        y_per_time.append(arr)

    # --- x_range: if either bound is None, auto-compute from times ---
    # resolved BEFORE y-range logic so y-range only considers in-x-range points
    x_lo, x_hi = x_range
    if times.size > 1:
        _t_span = times.max() - times.min()
    else:
        _t_span = 1.0
    if x_lo is None:
        x_lo = times.min() - 0.05 * _t_span
    if x_hi is None:
        x_hi = times.max() + 0.05 * _t_span

    # --- restrict the points used for y-range logic to those inside x_range ---
    in_x = (times >= x_lo) & (times <= x_hi)
    nonempty = [y_per_time[i] for i in range(len(y_per_time))
                if in_x[i] and y_per_time[i].size]
    all_vals = np.concatenate(nonempty) if nonempty else np.array([])

    # --- y_range: if either bound is None, use percentile bounds ---
    y_lo, y_hi = y_range
    if y_lo is None or y_hi is None:
        if all_vals.size:
            all_vals = np.asarray(all_vals)
            all_vals = all_vals[np.isfinite(all_vals)]  # drop inf/nan
            _vals_for_range = all_vals[all_vals > 0] if ylog else all_vals
            if _vals_for_range.size:
                auto_lo, auto_hi = np.percentile(_vals_for_range, percent_bounds)
            else:
                auto_lo, auto_hi = (1e-3, 1.0) if ylog else (0.0, 1.0)
        else:
            auto_lo, auto_hi = (1e-3, 1.0) if ylog else (0.0, 1.0)
        if y_lo is None:
            y_lo = auto_lo
        if y_hi is None:
            y_hi = auto_hi

    # --- optional: iteratively converge the range onto bins that meet hist_min_count ---
    # runs independently per time-slice; the widest converged range is shared by all
    if iterative_range and nonempty:
        print("Running _converge_hist_range")
        y_lo, y_hi = _converge_hist_range(nonempty, y_lo, y_hi, n_bins,
                                          hist_min_count, range_buffer, ylog)

    # shared bins across all times so heights are comparable
    if ylog:
        bin_edges = np.logspace(np.log10(max(y_lo, 1e-300)), np.log10(y_hi), n_bins + 1)
    else:
        bin_edges = np.linspace(y_lo, y_hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_h = (bin_edges[1:] - bin_edges[:-1]) * 0.9
    # horizontal scale for the histograms, based on time spacing
    if times.size > 1:
        dt = np.min(np.diff(np.sort(times)))
    else:
        dt = 1.0
    if not include_scatter:
        hist_offset=0.0
        hist_frac=0.9
    hist_width = hist_frac * dt
    fig, ax = plt.subplots(figsize=figsize)
    # Setting widths:
    count_inds = [0,1] if renorm else [1]
    maxest_count = 0
    for count_ind in count_inds:
        for i, t in enumerate(times):
            yv = y_per_time[i]
            if yv.size == 0:
                continue
            # raw scatter at this time
            if include_scatter and count_ind==1:
                ax.scatter(np.full_like(yv, t), yv, s=point_size,
                    color='r', alpha=0.35, zorder=3, edgecolors='none')
            # histogram along y, normalised so the tallest bin == hist_width
            counts, _ = np.histogram(yv, bins=bin_edges)
            if counts.max() == 0:
                continue
            counts = counts.astype(float)
            counts[counts < hist_min_count] = 0
            if counts.max() == 0:
                continue
            if count_ind == 0:
                maxest_count = max(maxest_count,counts.max())
            local_count = counts.max()
            widths = counts / max(local_count,maxest_count) * hist_width
            # no angled: shear each bar in x by its height above y_lo
            if count_ind==1:
                ax.barh(bin_centers, widths, left=t + hist_offset*dt, height=bin_h,
                    color='k', alpha=1.0, zorder=2,
                    edgecolor='none')
    if not ylog:
        ax.set_ylim(y_lo-range_buffer*(y_hi-y_lo), y_hi+range_buffer*(y_hi-y_lo))
    else:
        ax.set_ylim(y_lo*(1-range_buffer), y_hi*(1+range_buffer))
    ax.set_xlim(x_lo, x_hi)
    # --- primary axis: keep your auto-labeled major ticks, add 100 ms minor ticks ---
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='x', which='minor', length=4)   # medium ticks at 100 ms
    ax.tick_params(axis='x', which='major', length=5)    # (optional) longer majors

    # --- secondary axis just to carry the 10 ms ticks (no labels) ---
    #ax2 = ax.secondary_xaxis('bottom')
    #ax2.set_xlim(*ax.get_xlim())          # match limits (usually automatic, but explicit is safe)
    #ax2.xaxis.set_major_locator(MultipleLocator(20))
    #ax2.xaxis.set_major_formatter(NullFormatter())  # no text
    #ax2.tick_params(axis='x', which='major', length=2)  # tiny ticks at 10 ms
    #ax2.spines['bottom'].set_visible(False)  # hide the duplicate spine line
    #ax.xaxis.set_major_locator(MultipleLocator(100))
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.tick_params(axis='x', which='major', length=6)
    #ax.tick_params(axis='x', which='minor', length=2)
    if ylog:
        ax.set_yscale('log')
    _x_label = f"{x_plot_name} {x_unit}" if x_unit else x_plot_name
    _y_label_base = y_plot_name if y_plot_name is not None else var_name
    _y_label = f"{_y_label_base} {y_unit}" if y_unit else _y_label_base
    #_shot_str = f" in shot #{shotnum}" if shotnum is not None else ""
    _title = f"Shot {shotnum}" if shotnum is not None else ""
    ax.set_xlabel(_x_label, fontsize=xlabel_fontsize)
    ax.set_ylabel(_y_label, fontsize=ylabel_fontsize)
    ax.set_title(_title, fontsize=title_fontsize)
    fig.tight_layout()
    if save_name is None:
        full_name=gen_name(var_name,
                sel_option_dict,
                isel_option_dict,
                where_option_dict)
        save_name = f'{full_name}_{int(_time.time())}{special_id}.svg' if time_id else f'{full_name}{special_id}.svg'
    else:
        save_name = f'{save_name}_{int(_time.time())}{special_id}.svg' if time_id else f'{save_name}{special_id}.svg'
    if save_name is not None:
        out_path = os.path.join(output_dir, save_name)
        fig.savefig(out_path, format='svg')
        print(f'[saved] {out_path}')
    return fig, ax, (times, value_matrix, xri_list, [x_lo, x_hi])

# Substring -> abbreviation map used to shorten overly long generated names
# while keeping the information recoverable. Applied longest-key-first so that
# e.g. 'Delta_prime' is replaced before 'diff'. Extend as needed.
_NAME_ABBREV = {
    'Delta_prime':   'Dp',
    'across_GPEC':   'xGPEC',
    'reldiff':       'rdf',
    'ImRatio':       'ImR',
    'psi_n_rational':'psin',
    'Delta_prime_type':'Dptype',
    '_surf':         's',
    '_max':          'mx',
    'marg':          'mrg',
    'dwdtau':        'dwdt',
    'diff':          'df',
    'value':         'v',
    'nzero':         'nz',
}

def _abbrev(name):
    """Shorten a token using ``_NAME_ABBREV`` substring replacements.

    Replacements are applied longest-key-first so that longer, more specific
    substrings win over shorter ones. Non-string inputs are returned as-is.
    """
    if not isinstance(name, str):
        return name
    for long, short in sorted(_NAME_ABBREV.items(), key=lambda kv: -len(kv[0])):
        name = name.replace(long, short)
    return name

def _expr_to_name(expr, abbrev=False):
    """Render a where-expression tree into a compact, filename-safe string.

    Leaf:   {'var': 'var_x', 'op': '<', 'val': 2}  -> 'var_x<2'
            {'var': 'var_x', 'val': 0}             -> 'var_x==0'  (implicit ==)
    Branch: {'or':  [expr, expr, ...]}             -> '(a-OR-b)'
            {'and': [expr, expr, ...]}             -> '(a-AND-b)'

    When ``abbrev`` is True, variable names are shortened via ``_abbrev``.
    """
    if isinstance(expr, dict) and ('or' in expr or 'and' in expr):
        combine = 'or' if 'or' in expr else 'and'
        sep = '-OR-' if combine == 'or' else '-AND-'
        rendered = [_expr_to_name(s, abbrev) for s in expr[combine]]
        rendered = [r for r in rendered if r]
        if not rendered:
            return ''
        if len(rendered) == 1:
            return rendered[0]
        return '(' + sep.join(rendered) + ')'
    if isinstance(expr, dict) and 'var' in expr:
        var = _abbrev(expr['var']) if abbrev else expr['var']
        op  = expr.get('op', '==')
        return f"{var}{op}{expr['val']}"
    return ''

def gen_name(var_name,
                sel_option_dict,
                isel_option_dict,
                where_option_dict,
                max_len=100):
    """Build a descriptive, filename-safe name for a distribution figure.

    If the full name exceeds ``max_len`` characters, it is rebuilt with the
    ``_NAME_ABBREV`` translation table applied so the name stays short while
    remaining human-decodable. Set ``max_len=None`` to always use full names.
    """
    full = _gen_name(var_name, sel_option_dict, isel_option_dict,
                     where_option_dict, abbrev=False)
    if max_len is not None and len(full) > max_len:
        return _gen_name(var_name, sel_option_dict, isel_option_dict,
                         where_option_dict, abbrev=True)
    return full

def _gen_name(var_name,
                sel_option_dict,
                isel_option_dict,
                where_option_dict,
                abbrev=False):
    _ab = _abbrev if abbrev else (lambda x: x)
    sel   = sel_option_dict   or {}
    isel  = isel_option_dict  or {}
    where = where_option_dict or {}
    parts = [_ab(var_name)]
    # m<r_value>n<nn> from sel['nn'] + where['r_value']
    nn    = sel.get('nn')
    r_val = where.get('r_value')
    if nn is not None and r_val is not None:
        parts.append(f"m{int(r_val)}n{nn}")
    elif nn is not None:
        parts.append(f"n{nn}")
    elif r_val is not None:
        parts.append(f"m{int(r_val)}")
    # code (e.g. 'rdcon', 'stride')
    if 'code' in sel:
        parts.append(sel['code'])
    # isel: skip Delta_prime_type=0 (standard)
    for key, val in isel.items():
        if key == 'Delta_prime_type' and val == 0:
            continue
        parts.append(f"{_ab(key)}{val}")
    # where: skip r_value (already encoded above)
    #        skip psi_n_rational < 0.95 (standard)
    #        render compound/special keys (e.g. __cond_xy__) via their
    #        expression tree
    for key, cond in where.items():
        if isinstance(key, str) and key.startswith('__') and key.endswith('__'):
            exprs = cond if isinstance(cond, list) else [cond]
            for expr in exprs:
                rendered = _expr_to_name(expr, abbrev=abbrev)
                if rendered:
                    parts.append(rendered)
            continue
        if key == 'r_value' or (key == 'r_occ' and cond==0):
            continue
        conds = cond if isinstance(cond, list) else [cond]
        if key == 'psi_n_rational':
            non_std = [c for c in conds
                       if not (isinstance(c, dict)
                               and c.get('op') == '<'
                               and c.get('val') == 0.95)]
            for c in non_std:
                op_str = c['op'] if isinstance(c, dict) else '=='
                v_str  = c['val'] if isinstance(c, dict) else c
                parts.append(f"psin{op_str}{v_str}")
            continue
        for c in conds:
            if isinstance(c, dict):
                parts.append(f"{_ab(key)}{c['op']}{c['val']}")
            else:
                parts.append(f"{_ab(key)}{c}")
    return '_'.join(str(p) for p in parts)

def prof_plotter(output_dir, var_name, times, xri_list,
                 alpha=0.4, lw=0.8, save_fmt="svg",):
    prof_dir = os.path.join(output_dir, f"{var_name}_profile_figs")
    os.makedirs(prof_dir, exist_ok=True)
    for i, t in enumerate(times):
        xri_loc = xri_list[i]
        assert xri_loc.IDA_time_idx.min().values==xri_loc.IDA_time_idx.max().values
        run_idxs = xri_loc.run_idx.values
        fig, ax = plt.subplots(figsize=(8, 5))
        for j in range(len(run_idxs)):
            xvals = xri_loc.isel(run_idx=j).psi_n.values.ravel()
            yvals = xri_loc.isel(run_idx=j)[var_name].values.ravel()
            mask = ~np.isnan(yvals)
            #print(xvals,yvals)
            #print(xri_loc.isel(run_idx=j)[var_name])
            ax.plot(np.sqrt(xvals[mask]), yvals[mask], lw=lw, alpha=alpha, color='C0')
            #import sys
            #sys.exit()
        ax.set_xlabel('rho_n')
        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name}  t={t:.4g}  ({len(run_idxs)} runs)")
        fig.tight_layout() 
        fname = f"{var_name}_t{xri_loc.IDA_time_idx.min().values}.{save_fmt}"
        out_path = os.path.join(prof_dir, fname)
        fig.savefig(out_path, format=save_fmt)
        plt.close(fig)
        print(f"[saved] {out_path}")
