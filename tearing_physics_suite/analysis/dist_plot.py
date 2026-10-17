"""dist_plotter (old_plot_scripts 2d6516e) on one shot: per time slice, a histogram of one surface variable
over the slice's runs (bouquet draws), with the raw points.

    python -m tearing_physics_suite.analysis.dist_plot --source SRC --shot N --out DIR
"""
import argparse
import operator
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .shot_data import open_shot, select

_OPS = {'<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge, '==': operator.eq,
        '!=': operator.ne}
# (variable, n, m); rdcon, single-helicity Delta', MRE terms at the default k1, C0
CASES = tuple((v, n, m) for v in ('Delta_prime_surf', 'dwdtau_max_surf', 'w_marg_surf')
              for n, m in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)))
SEL = {'code': 'rdcon', 'Delta_prime_type': 'single helicity', 'k1': 1.7, 'C0': 0.6}


def _where_vars(key, cond):
    """Variables one where-entry reads."""
    if not key.startswith('__'):
        return {key}
    out = set()
    for e in (cond if isinstance(cond, list) else [cond]):
        stack = [e]
        while stack:
            x = stack.pop()
            if 'var' in x:
                out.add(x['var'])
            stack += x.get('or', []) + x.get('and', [])
    return out


def _expr_mask(ds, e):
    """{'var', 'op', 'val'} (no op: ==), or {'or' | 'and': [expr, ...]}."""
    if 'or' in e or 'and' in e:
        masks = [_expr_mask(ds, s) for s in e.get('or', e.get('and'))]
        out = masks[0]
        for m in masks[1:]:
            out = (out | m) if 'or' in e else (out & m)
        return out
    return _OPS[e.get('op', '==')](ds[e['var']], e['val'])


def _mask(ds, key, cond):
    """One where-entry: {var: val}, {var: [{'op', 'val'}, ...]} or {'__name': expr | [expr, ...]}."""
    if key.startswith('__'):
        exprs = cond if isinstance(cond, list) else [cond]
        out = _expr_mask(ds, exprs[0])
        for e in exprs[1:]:
            out = out & _expr_mask(ds, e)
        return out
    out = None
    for c in (cond if isinstance(cond, list) else [cond]):
        m = _OPS[c['op']](ds[key], c['val']) if isinstance(c, dict) else ds[key] == c
        out = m if out is None else out & m
    return out


def dist_values(ds, var, sel=None, isel=None, where=None):
    """(t_ms, [finite values per t_ms]) of ds[var] after sel / isel (keys ds lacks are ignored) and the
    where-filters (dist_extractor's syntax, _mask); a filter on a variable ds lacks is skipped with a warning."""
    ds = ds.sel({k: v for k, v in (sel or {}).items() if k in ds.dims})
    ds = ds.isel({k: v for k, v in (isel or {}).items() if k in ds.dims})
    da = ds[var]
    for key, cond in (where or {}).items():
        missing = _where_vars(key, cond) - set(ds.variables)
        if missing:
            warnings.warn(f'{var}: where {key!r} skipped, {sorted(missing)} not in the data')
            continue
        da = da.where(_mask(ds, key, cond))
    t = np.asarray(ds['t_ms'].values, float)
    times = np.unique(t[np.isfinite(t)])
    vals = [np.asarray(da.isel(run_idx=np.flatnonzero(t == tk)).values, float).ravel() for tk in times]
    return times, [v[np.isfinite(v)] for v in vals]


def dist_plot(times, values, var, y_range=(None, None), x_range=(None, None), n_bins=40, hist_frac=0.65,
              hist_offset=0.2, point_size=8, percent_bounds=(0, 100), range_buffer=0.05, ylog=None,
              sample_size=1000, hist_min_count=1, include_scatter=True, renorm=False, y_label=None,
              title='', figsize=(12, 6), random_seed=None, onset=None, activity_onset=None):
    """dist_plotter: a horizontal histogram per time (shared bins; log for w_marg) right of that time's
    points (at most sample_size, randomly sampled). renorm: one count scale for all times, else each
    time's own. A time with a single run gets a hollow marker. onset / activity_onset [ms]: first onset of the
    labelled mode (same m/|n|) / of the labelled activity (same |n|, same m parity), vertical lines. Returns the
    figure."""
    ylog = 'w_marg' in var if ylog is None else ylog
    rng = np.random.default_rng(random_seed)
    times = np.asarray(times, float)
    ys = [v if v.size <= sample_size else rng.choice(v, sample_size, replace=False) for v in values]
    marks = [t for t in (onset, activity_onset) if t is not None]
    tx = np.append(times, marks)
    span = np.ptp(tx) if tx.size > 1 else 1.0
    x_lo = tx.min() - 0.05 * span if x_range[0] is None else x_range[0]
    x_hi = tx.max() + 0.05 * span if x_range[1] is None else x_range[1]
    inx = [y for t, y in zip(times, ys) if x_lo <= t <= x_hi and y.size]
    pts = np.concatenate(inx) if inx else np.array([])
    pts = pts[pts > 0] if ylog else pts
    auto = np.percentile(pts, percent_bounds) if pts.size else ((1e-3, 1.0) if ylog else (0.0, 1.0))
    y_lo, y_hi = (auto[i] if y_range[i] is None else y_range[i] for i in (0, 1))
    edges = (np.logspace(np.log10(max(y_lo, 1e-300)), np.log10(y_hi), n_bins + 1) if ylog
             else np.linspace(y_lo, y_hi, n_bins + 1))
    dt = np.min(np.diff(np.sort(times))) if times.size > 1 else 1.0
    if not include_scatter:
        hist_offset, hist_frac = 0.0, 0.9
    counts = [np.histogram(y, bins=edges)[0].astype(float) for y in ys]
    for c in counts:
        c[c < hist_min_count] = 0
    top = max((c.max() for c in counts if c.size), default=0)

    fig, ax = plt.subplots(figsize=figsize)
    single = True
    for t, y, c in zip(times, ys, counts):
        if y.size == 1:   # one run (e.g. baseline only): a hollow marker, no histogram
            ax.scatter([t], y, s=4 * point_size, facecolors='none', edgecolors='k', zorder=3,
                       label='single run' if single else None)
            single = False
            continue
        if include_scatter and y.size:
            ax.scatter(np.full_like(y, t), y, s=point_size, color='r', alpha=0.35, zorder=3, edgecolors='none')
        on = c > 0   # empty bins are not drawn (a zero-width bar still renders as a hairline)
        if on.any():
            ax.barh(0.5 * (edges[:-1] + edges[1:])[on], c[on] / (top if renorm else c.max()) * hist_frac * dt,
                    left=t + hist_offset * dt, height=np.diff(edges)[on] * 0.9, color='k', edgecolor='none',
                    zorder=2)
    if ylog:
        ax.set_yscale('log')
        ax.set_ylim(y_lo * (1 - range_buffer), y_hi * (1 + range_buffer))
    else:
        ax.set_ylim(y_lo - range_buffer * (y_hi - y_lo), y_hi + range_buffer * (y_hi - y_lo))
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel('t [ms]')
    ax.set_ylabel(y_label or var)
    ax.set_title(title)
    if onset is not None:
        ax.axvline(onset, color='tab:blue', ls='--', lw=1.5, zorder=4, label=f'mode onset (m/|n|) {onset:g} ms')
    if activity_onset is not None:
        ax.axvline(activity_onset, color='tab:orange', ls=':', lw=1.8, zorder=4,
                   label=f'activity onset (|n|, m parity) {activity_onset:g} ms')
    if not single or marks:
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig


def plot_shot(source, shot, out_dir, cases=CASES, t_ms=None, draws=None, fmt='pdf', onsets=None, activity=None,
              **kw):
    """One dist_plot per case (variable, n, m) present in the shot's runs, unique surfaces only (r_occ 0),
    to <out_dir>/<shot>_<var>_m<m>n<n>.<fmt>; y-range from the 2-98th percentiles unless kw says.
    onsets: {(m, |n|): onset t_ms}, activity: {(|n|, m % 2): onset t_ms}, drawn on the matching plots.
    Returns the paths."""
    kw.setdefault('percent_bounds', (2, 98))
    variables = sorted({c[0] for c in cases} | {'r_value', 'r_occ'})
    ds = select(open_shot(source, shot, variables), t_ms, draws)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for var, n, m in cases:
        if var not in ds or n not in ds['nn'].values:
            continue
        times, vals = dist_values(ds, var, sel=dict(SEL, nn=n), where={'r_value': m, 'r_occ': 0})
        if not any(v.size for v in vals):
            continue
        fig = dist_plot(times, vals, var, title=f'shot {shot}  {m}/{n}  {var}  (rdcon, single helicity; '
                                                f'{ds.sizes["run_idx"]} runs)',
                        onset=(onsets or {}).get((m, abs(n))), activity_onset=(activity or {}).get((abs(n), m % 2)),
                        **kw)
        paths.append(os.path.join(out_dir, f'{shot}_{var}_m{m}n{n}.{fmt}'))
        fig.savefig(paths[-1])
        plt.close(fig)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--source', required=True, help='study dir, zarr store, or a glob of run dirs')
    p.add_argument('--shot', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--vars', default='', help='comma-separated subset of the CASES variables')
    p.add_argument('--fmt', default='pdf', choices=('pdf', 'svg'))
    p.add_argument('--onsets', default='', help='mode onset times, m/|n|:t_ms, comma-separated (e.g. 2/1:1650,3/2:1400)')
    p.add_argument('--activity', default='', help='activity onset times, |n|/odd|even:t_ms (e.g. 1/odd:1600,2/even:1450)')
    a = p.parse_args(argv)
    cases = [c for c in CASES if not a.vars or c[0] in a.vars.split(',')]
    onsets = {tuple(int(x) for x in mn.split('/')): float(t) for mn, t in (o.split(':') for o in a.onsets.split(',') if o)}
    activity = {(int(n), ('even', 'odd').index(par)): float(t)
                for nk, t in (o.split(':') for o in a.activity.split(',') if o) for n, par in [nk.split('/')]}
    for path in plot_shot(a.source, a.shot, a.out, cases=cases, fmt=a.fmt, onsets=onsets, activity=activity):
        print(path)


if __name__ == '__main__':
    main()
