"""i-file (eq_type='ldp_i', inverse.f) vs g-file (eq_type='efit', direct.f): GPEC GSE, Delta', file size.

usage: python run_ifile_comparison.py BINDIR TRUTH OUTROOT [key=value ...]  (extra &EQUIL_CONTROL settings)
All files come from the same TokaMaker truth equilibrium (make_truth_equilibrium.py). Each i-file
is run as written by OFT (F, p only -> profile_source falls back to values) and with appended
FF', p' records (profile_source = integrate). psihigh puts the last surface at true psi_N = 0.985.
Writes OUTROOT/ifile_results.json.
"""
import os, sys, json, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../spline_improvements/scripts'))
from gpec_runs import run_stride, read_delta_prime, read_gsec, gse_metrics
from ifile_tools import append_derivatives

BIN, TRUTH, ROOT = sys.argv[1:4]
EXTRA = dict(a.split('=', 1) for a in sys.argv[4:])
META = json.load(open(os.path.join(TRUTH, 'truth_meta.json')))
PSI_TRUE = 0.985


def cases():
    out = []
    for g in sorted(glob.glob(os.path.join(TRUTH, 'g*.geqdsk'))):
        out.append((os.path.basename(g)[:-7], g, 'efit', 'integrate', PSI_TRUE / (1 - META['psi_pad'])))
    os.makedirs(os.path.join(ROOT, 'ifiles'), exist_ok=True)
    truth = np.load(os.path.join(TRUTH, 'truth_profiles.npz'))
    for f in sorted(glob.glob(os.path.join(TRUTH, 'i*.ifile'))):
        name = os.path.basename(f)[:-6]
        pad = float(name.split('_pad')[1]) if '_pad' in name else 0.01
        aug = os.path.join(ROOT, 'ifiles', name + '_ffp.ifile')
        append_derivatives(f, aug, truth, META['psi_bounds'])
        out.append((name, f, 'ldp_i', 'values', PSI_TRUE / (1 - pad)))
        out.append((name + '_ffp', aug, 'ldp_i', 'integrate', PSI_TRUE / (1 - pad)))
    return out


def one(c):
    name, path, eq_type, method, psihigh = c
    out = os.path.join(ROOT, name)
    ok = run_stride(BIN, out, path, eq_type=f'"{eq_type}"', profile_source=f'"{method}"',
                    psihigh=f'{psihigh:.6f}', **EXTRA)
    r = {'file': path, 'bytes': os.path.getsize(path), 'eq_type': eq_type, 'profile_source': method, 'ok': ok}
    if ok:
        q, dp, total = read_delta_prime(out)
        m = gse_metrics(read_gsec(out))
        r.update(q=q.tolist(), dp_diag=dp.tolist(), total_energy=total,
                 **{k: v for k, v in m.items() if not k.endswith('profile')})
    return name, r


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(int(os.environ.get('NPROC', '6'))) as ex:
        for name, r in ex.map(one, cases()):
            res[name] = r
            print(f"{name:24s} {r['bytes']:9d} B  " + ("FAILED" if not r['ok'] else
                  f"dP'(2/1)={r['dp_diag'][0]:8.4f} gse_local_med={r['local_med']:.2e} max={r['local_max']:.2e} "
                  f"int_med={r['int_med']:.2e}"), flush=True)
    json.dump(res, open(os.path.join(ROOT, 'ifile_results.json'), 'w'), indent=1)
