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
from .tm_labels import close_gaps, read_labels, spans

_OPS = {'<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge, '==': operator.eq,
        '!=': operator.ne}
# (variable, n, m); rdcon, single-helicity Delta', MRE terms at the default k1, C0
CASES = tuple((v, n, m) for v in ('Delta_prime_surf', 'dwdtau_max_surf', 'w_marg_surf')
              for n, m in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)))
SEL = {'code': 'rdcon', 'Delta_prime_type': 'single helicity', 'k1': 1.7, 'C0': 0.6}
BAND_COLORS = ('#e6e6e6', '#efe6d8', '#d3e0d6')   # no label, activity only, mode


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
              title='', figsize=(12, 6), random_seed=None, bands=None, amp=None, axes=None):
    """dist_plotter: a horizontal histogram per time (shared bins; log for w_marg) right of that time's
    points (at most sample_size, randomly sampled). renorm: one count scale for all times, else each
    time's own. A time with a single run gets a hollow marker. bands: [(t0, t1, label, color)], shaded behind
    the points and named in a strip below; amp: [(t_ms, y, label, lw)], a panel below (dB). axes: draw into these
    (main[, strip][, amp]) instead of a new figure, e.g. to stack other panels on the same time axis. Returns the
    figure."""
    ylog = 'w_marg' in var if ylog is None else ylog
    rng = np.random.default_rng(random_seed)
    times = np.asarray(times, float)
    ys = [v if v.size <= sample_size else rng.choice(v, sample_size, replace=False) for v in values]
    tx = times
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

    rows = [6] + [0.5] * bool(bands) + [2.5] * bool(amp)
    if axes is None:
        fig, axs = plt.subplots(len(rows), 1, figsize=(figsize[0], figsize[1] * sum(rows) / 6), sharex=True,
                                gridspec_kw=dict(height_ratios=rows, hspace=0.05), squeeze=False)
    else:
        axs = np.asarray(axes).reshape(-1, 1)
        fig = axs[0, 0].figure
    ax = axs[0, 0]
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
    ax.set_ylabel(y_label or var)
    ax.set_title(title)
    if not single:
        ax.legend(loc='best', fontsize=8)
    if bands:
        strip = axs[1, 0]
        for t0, t1, label, color in bands:
            ax.axvspan(t0, t1, color=color, lw=0, zorder=0)
            strip.axvspan(t0, t1, color=color, lw=0)
            lo, hi = max(t0, x_lo), min(t1, x_hi)
            if hi - lo > 0.12 * (x_hi - x_lo):
                strip.text(0.5 * (lo + hi), 0.5, label, ha='center', va='center', fontsize=9, style='italic')
        strip.set_yticks([])
        strip.set_ylabel('labels', rotation=0, ha='right', va='center', fontsize=8)
    if amp:
        pa = axs[-1, 0]
        inx = [(i, *a) for i, a in enumerate(amp) if np.isfinite(a[1][(a[0] >= x_lo) & (a[0] <= x_hi)]).any()]
        for i, t, y, label, lw in inx:   # one colour per trace across the cases
            pa.plot(t, y, lw=lw, label=label, color=f'C{i % 10}')
        if inx:
            yy = np.concatenate([y[(t >= x_lo) & (t <= x_hi) & np.isfinite(y)] for _, t, y, _, _ in inx])
            pa.set_ylim(yy.min() - 2, yy.max() + 2)
            pa.legend(loc='upper left', fontsize=7, ncol=4)
        pa.set_ylabel('mode power [dB]')
    if axes is None:
        axs[-1, 0].set_xlabel('t [ms]')
        fig.tight_layout()
    return fig


def label_layers(labels, m, n, gap_ms=20.0, amp_panel=False):
    """dist_plot's bands (no label / activity of the same |n| and m parity only / the m/|n| mode, gaps shorter
    than gap_ms closed) and, with amp_panel, its amp traces (every labelled mode, m/|n| thick) from read_labels."""
    t = labels['t_ms']
    off = np.zeros(t.shape, bool)
    on = close_gaps(labels['mode'].get((m, n), off), t, gap_ms)
    act = close_gaps(labels['activity'].get((n, m % 2), off), t, gap_ms)
    names = ('no label', f'n={n} m {("even", "odd")[m % 2]} activity', f'{m}/{n} mode')
    bands = [(t0, t1, names[k], BAND_COLORS[k]) for t0, t1, k in spans(np.where(on, 2, np.where(act, 1, 0)), t)]
    amp = [(t, y, name, 1.8 if name in (f'({m},{n})', f'({m},-{n})') else 0.8) for name, y in labels['amp_db'].items()]
    return bands, (amp if amp_panel else None)


def plot_shot(source, shot, out_dir, cases=CASES, t_ms=None, draws=None, fmt='pdf', labels=None, gap_ms=20.0,
              amp_panel=False, label_var='ntm_mask', **kw):
    """One dist_plot per case (variable, n, m) present in the shot's runs, unique surfaces only (r_occ 0),
    to <out_dir>/<shot>_<var>_m<m>n<n>.<fmt>; y-range from the 2-98th percentiles unless kw says.
    labels: the shot's TARS <shot>_labeled.nc (or a file of its layout, mode mask label_var), shaded per case
    (label_layers). Returns the paths."""
    kw.setdefault('percent_bounds', (2, 98))
    variables = sorted({c[0] for c in cases} | {'r_value', 'r_occ'})
    ds = select(open_shot(source, shot, variables), t_ms, draws)
    os.makedirs(out_dir, exist_ok=True)
    lab = read_labels(labels, label_var) if labels else None
    paths = []
    for var, n, m in cases:
        if var not in ds or n not in ds['nn'].values:
            continue
        times, vals = dist_values(ds, var, sel=dict(SEL, nn=n), where={'r_value': m, 'r_occ': 0})
        if not any(v.size for v in vals):
            continue
        fig = dist_plot(times, vals, var, title=f'shot {shot}  {m}/{n}  {var}  (rdcon, single helicity; '
                                                f'{ds.sizes["run_idx"]} runs)',
                        **dict(zip(('bands', 'amp'), label_layers(lab, m, abs(n), gap_ms, amp_panel)) if lab else {}),
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
    p.add_argument('--labels', default='', help="the shot's TARS <shot>_labeled.nc: shade the mode / activity times")
    p.add_argument('--label-var', default='ntm_mask', help='mode mask in --labels (ntm_mask_fused: eq_stab fused)')
    p.add_argument('--gap-ms', type=float, default=20.0, help='label gaps shorter than this are closed')
    p.add_argument('--amp-panel', action='store_true', help='add a panel of the labelled modes\' power [dB]')
    a = p.parse_args(argv)
    cases = [c for c in CASES if not a.vars or c[0] in a.vars.split(',')]
    for path in plot_shot(a.source, a.shot, a.out, cases=cases, fmt=a.fmt, labels=a.labels or None, gap_ms=a.gap_ms,
                          amp_panel=a.amp_panel, label_var=a.label_var):
        print(path)


if __name__ == '__main__':
    main()
