import operator
import xarray as xr
import os
import operator
import time as _time
import numpy as np
home_dir = os.environ['TPSHOME']
from tearing_physics_suite.tearing_physics_suite import add_unique_label

#xri = xr.open_dataset(os.path.join(
#    "/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/",
#    "test3/TPS_run_1/compiled_xr_meta.nc"))

master_working_dir=os.path.join(
    "/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/",
    "174956bigT250/TPS_run_2")

xri = xr.open_zarr(os.path.join(master_working_dir,
    "compiled_combined_xr.zarr")
    , consolidated=True)

eq_labels = xri["eq_label"].values.tolist()
eq_labels_arr = np.asarray(eq_labels, dtype="U")
xri = xri.assign_coords(eq_label=("run_idx", eq_labels_arr))


print(xri)


# Map operator strings to comparison functions
_OPS = {
    '<':  operator.lt,
    '<=': operator.le,
    '>':  operator.gt,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne,
}

def dist_extractor(output_dir, xri, var_name,
                   sel_option_dict=None,
                   isel_option_dict=None,
                   where_option_dict=None,
                   one_surf=True,
                   unique_surf_filter=True,
                   debug=False,
                   debug_n_per_time=None,   # randomly sample this many points per time slice
                   sample_dim='run_idx',    # dimension to subsample / split along
                   random_seed=None,        # reproducibility for the random sampling
                   timing=True):            # print timing diagnostics
    from time import perf_counter
    os.makedirs(output_dir, exist_ok=True)
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
    # BUILD A MINIMAL VARIABLE SUBSET (big speed win)
    # -----------------------------------------------------------------
    t0 = _tic()
    needed = {var_name, 'IDA_time_idx', 'IDA_time'}
    if where_option_dict:
        needed.update(where_option_dict.keys())
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

    # -----------------------------------------------------------------
    # Precompute the split index ONCE
    # -----------------------------------------------------------------
    t0 = _tic()
    tidx_da = xri_small.IDA_time_idx
    tidx_dim = tidx_da.dims[0]
    tidx_arr = tidx_da.compute().values.ravel()
    if timing:
        print(f"[timing] IDA_time_idx compute (split index): "
              f"{perf_counter()-t0:.3f}s  (len={tidx_arr.size})", flush=True)

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

        t0 = _tic()
        ida_times = xri_idx.IDA_time.compute().values.ravel()
        _toc('ida_time_compute', t0, 'ida_time_compute')
        unique_times = np.unique(ida_times[np.isfinite(ida_times)])
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
            if where_option_dict:
                if debug: print(f"[where] starting size: {xri_idx_where.sizes}")
                for key, cond in where_option_dict.items():
                    conds = cond if isinstance(cond, list) else [cond]
                    for c in conds:
                        if isinstance(c, dict):
                            if debug: print(f"[where] key={key!r}, op={c['op']!r}, val={c['val']!r}")
                            mask = _OPS[c['op']](xri_idx_where[key], c['val'])
                        else:
                            if debug: print(f"[where] key={key!r}, == {c!r}")
                            mask = xri_idx_where[key] == c
                        mask = mask.compute()
                        if debug: print(f"[where]   mask sum={int(mask.sum())}, size before drop={xri_idx_where.sizes}")
                        xri_idx_where = xri_idx_where.where(mask, drop=True)
                        if debug: print(f"[where]   size after drop={xri_idx_where.sizes}")
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

import numpy as np
import matplotlib.pyplot as plt

def dist_plotter(output_dir=None, var_name=None,
                 times=None, value_matrix=None, xri_list=None,
                 y_range=[None, None],
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
                 debug_n_per_time=None,   # randomly sample this many points per time slice
                 random_seed=None):       # reproducibility for the random sampling
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
    nonempty = [a for a in y_per_time if a.size]
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
        y_range = [y_lo, y_hi]
        if ylog:
            y_lo = y_lo * (1.0 - range_buffer)
            y_hi = y_hi * (1.0 + range_buffer)
        else:
            y_lo, y_hi = [y_lo-range_buffer*(y_hi-y_lo), y_hi+range_buffer*(y_hi-y_lo)]
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
    hist_width = hist_frac * dt
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, t in enumerate(times):
        yv = y_per_time[i]
        if yv.size == 0:
            continue
        # raw scatter at this time
        ax.scatter(np.full_like(yv, t), yv, s=point_size,
                   color='k', alpha=0.35, zorder=3)
        # histogram along y, normalised so the tallest bin == hist_width
        counts, _ = np.histogram(yv, bins=bin_edges)
        if counts.max() == 0:
            continue
        widths = counts / counts.max() * hist_width
        # no angled: shear each bar in x by its height above y_lo
        ax.barh(bin_centers, widths, left=t + hist_offset*dt, height=bin_h,
                color='C0', alpha=0.55, zorder=2,
                edgecolor='none')
    ax.set_ylim(y_lo, y_hi)
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel('IDA_time')
    ax.set_ylabel(var_name)
    ax.set_title(f'Distribution of {var_name} vs time')
    fig.tight_layout()
    if save_name is None:
        full_name=gen_name(var_name,
                sel_option_dict,
                isel_option_dict,
                where_option_dict)
        save_name = f'{full_name}_{int(_time.time())}.png'
    else:
        save_name = f'{save_name}_{int(_time.time())}.png'
    if save_name is not None:
        out_path = os.path.join(output_dir, save_name)
        fig.savefig(out_path, format='png')
        print(f'[saved] {out_path}')
    return fig, ax, (times, value_matrix, xri_list)

def gen_name(var_name,
                sel_option_dict,
                isel_option_dict,
                where_option_dict):
    sel   = sel_option_dict   or {}
    isel  = isel_option_dict  or {}
    where = where_option_dict or {}
    parts = [var_name]
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
        parts.append(f"{key}{val}")
    # where: skip r_value (already encoded above)
    #        skip psi_n_rational < 0.95 (standard)
    for key, cond in where.items():
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
                parts.append(f"{key}{c['op']}{c['val']}")
            else:
                parts.append(f"{key}{c}")
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

import numpy as np
import sys

# ---- shared configuration ------------------------------------------------
DATA_DIR = os.path.join(master_working_dir,'TPS_figs')

SHARED_WHERE = {
    'psi_n_rational': [{'op': '<', 'val': 0.95}],
    'r_occ': 0,
    'Delta_prime_ImRatio': [{'op': '<', 'val': 1e-2}],
    'nzero': 0,
}

SHARED_ISEL = {'Delta_prime_type': 0}

SHARED_PLOT_KW = dict(
    percent_bounds=[3, 97],
    n_bins=250,
    point_size=3,
    #angle_deg=0.0,
    range_buffer=0.1,
)

# ---- per-case parameters: (nn, r_value, y_range) -------------------------
VAR_NAME = 'Delta_prime_surf'
cases = [
    {'nn': 1, 'r_value': 2.0, 'y_range': [None, None]},
    {'nn': 2, 'r_value': 3.0, 'y_range': [-50, 60]},
    {'nn': 3, 'r_value': 4.0, 'y_range': [-100, 10]},
    {'nn': 4, 'r_value': 5.0, 'y_range': [-100, 10]},
]
OVAR_NAME = 'w_marg_surf'
Ocases = [
    {'nn': 1, 'r_value': 2.0, 'y_range': None , 'ylog':True},
    {'nn': 2, 'r_value': 3.0, 'y_range': None , 'ylog':True},
    {'nn': 3, 'r_value': 4.0, 'y_range': None , 'ylog':True},
    {'nn': 4, 'r_value': 5.0, 'y_range': None , 'ylog':True},
]
OVAR_NAME = 'dwdtau_max_surf'
Ocases = [
    {'nn': 1, 'r_value': 2.0, 'y_range': [None, None]},
    {'nn': 2, 'r_value': 3.0, 'y_range': [None, None]},
    {'nn': 3, 'r_value': 4.0, 'y_range': [None, None]},
    {'nn': 4, 'r_value': 5.0, 'y_range': [None, None]},
]

# ---- driver loop ---------------------------------------------------------
results = []
for case in cases:
    plot_input = dist_extractor(
        DATA_DIR,
        xri,
        VAR_NAME,
        sel_option_dict={'code': 'rdcon', 'nn': case['nn']},
        where_option_dict={'r_value': case['r_value'], **SHARED_WHERE},
        isel_option_dict=SHARED_ISEL,
        debug=False,
    )
    total_finite = sum(
        np.isfinite(np.asarray(vals, dtype=float)).sum()
        for vals in plot_input['value_matrix']
    )
    print(f"[nn={case['nn']}, r={case['r_value']}] "
          f"Total finite values in value_matrix: {total_finite}", flush=True)
    plot_kw = dict(SHARED_PLOT_KW)
    if case.get('y_range') is not None:
        plot_kw['y_range'] = case['y_range']
    if case.get('ylog'):
        plot_kw['ylog'] = True
    fig, ax, outputs = dist_plotter(**plot_input, **plot_kw)
    results.append((fig, ax, outputs))

print("finito", flush=True)