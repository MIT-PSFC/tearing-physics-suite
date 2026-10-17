"""TARS tearing-mode labels (one <shot>_labeled.nc per shot) as time traces, for dist_plot."""
import numpy as np
import xarray as xr


def read_labels(path):
    """One shot's labels on the file's time bins: t_ms; mode {(m, |n|): on} (ntm_mask) and activity
    {(|n|, m parity): on} (ntm_activity_mask), on any frequency, both signs of n merged; amp_db {mode name:
    10 log10(ntm_sxx)} of the labelled modes (NaN where the mode is off)."""
    with xr.open_dataset(path) as d:
        t_ms = np.round(d['time'].values * 1e3, 3)
        mode = d['ntm_mask'].any('frequency').transpose('time_idx', 'mode_idx').values
        act = d['ntm_activity_mask'].any('frequency').transpose('time_idx', 'activity_idx').values
        m, n, names = d['mode_m'].values, d['mode_n'].values, d['mode_name'].values
        an, ap = d['activity_n'].values, d['activity_parity'].values
        sxx = d['ntm_sxx'].transpose('time_idx', 'mode_idx').values.astype(float)
    modes, activity, amp_db = {}, {}, {}
    for i in range(len(m)):
        k = (int(m[i]), abs(int(n[i])))
        modes[k] = modes.get(k, False) | mode[:, i]
        if mode[:, i].any():
            with np.errstate(divide='ignore', invalid='ignore'):
                amp_db[str(names[i])] = np.where(mode[:, i] & (sxx[:, i] > 0), 10 * np.log10(sxx[:, i]), np.nan)
    for i in range(len(an)):
        k = (abs(int(an[i])), int(ap[i]))
        activity[k] = activity.get(k, False) | act[:, i]
    return dict(t_ms=t_ms, mode=modes, activity=activity, amp_db=amp_db)


def close_gaps(on, t_ms, gap_ms):
    """on with its off stretches shorter than gap_ms between two on bins filled."""
    out = np.array(on, bool)
    idx = np.flatnonzero(out)
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a + 1 and t_ms[b] - t_ms[a] <= gap_ms:
            out[a:b] = True
    return out


def spans(state, t_ms):
    """[(t0, t1, value)] of the runs of equal value in state (t1 = the next run's first time)."""
    cut = np.flatnonzero(np.diff(state)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(state)]
    edge = np.r_[t_ms, t_ms[-1] + (t_ms[-1] - t_ms[-2] if len(t_ms) > 1 else 1.0)]
    return [(edge[a], edge[b], state[a]) for a, b in zip(starts, ends)]
