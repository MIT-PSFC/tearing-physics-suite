"""i-file (eq_type='ldp_i', inverse.f) vs g-file (eq_type='efit', direct.f): GPEC GSE, Delta', file size.

usage: python run_ifile_comparison.py BINDIR TRUTH IFILE_DIR OUTROOT [key=value ...]  (extra &EQUIL_CONTROL settings)
g-files from TRUTH (make_truth_equilibrium.py), i-files from IFILE_DIR (write_ifiles.py; env IFILES selects names,
e.g. i129x257,i257x513). All run with profile_source = integrate; an i-file without FF', p' records gets them
appended from the truth profiles first. psihigh puts the last surface at true psi_N = 0.985.
Writes OUTROOT/ifile_results.json.
"""
import os, sys, json, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../spline_improvements/scripts'))
from gpec_runs import run_stride, read_delta_prime, read_gsec, gse_metrics
from ifile_tools import read_ifile, append_derivatives

BIN, TRUTH, IDIR, ROOT = sys.argv[1:5]
EXTRA = dict(a.split('=', 1) for a in sys.argv[5:])
META = json.load(open(os.path.join(TRUTH, 'truth_meta.json')))
PSIHIGH = 0.985 / (1 - META['psi_pad'])
SEL = os.environ.get('IFILES')


def cases():
    out = [(os.path.basename(g)[:-7], g, 'efit', os.path.getsize(g)) for g in sorted(glob.glob(os.path.join(TRUTH, 'g*.geqdsk')))]
    truth = np.load(os.path.join(TRUTH, 'truth_profiles.npz'))
    os.makedirs(os.path.join(ROOT, 'ifiles'), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(IDIR, 'i*.ifile'))):
        name = os.path.basename(f)[:-6]
        if SEL and name not in SEL.split(','):
            continue
        size = os.path.getsize(f)
        if 'ffp' not in read_ifile(f):
            aug = os.path.join(ROOT, 'ifiles', name + '.ifile')
            append_derivatives(f, aug, truth, META['psi_bounds'])
            f = aug
        out.append((name, f, 'ldp_i', size))
    return out


def one(c):
    name, path, eq_type, size = c
    out = os.path.join(ROOT, name)
    ok = run_stride(BIN, out, path, eq_type=f'"{eq_type}"', profile_source='"integrate"',
                    psihigh=f'{PSIHIGH:.6f}', **EXTRA)
    r = {'bytes': size, 'eq_type': eq_type, 'ok': ok}
    if ok:
        q, dp, total = read_delta_prime(out)
        m = gse_metrics(read_gsec(out))
        r.update(dp_diag=dp.tolist(), **{k: v for k, v in m.items() if not k.endswith('profile')})
    return name, r


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(int(os.environ.get('NPROC', '4'))) as ex:
        for name, r in ex.map(one, cases()):
            res[name] = r
            print(f"{name:10s} {r['bytes']:9d} B  " + ('FAILED' if not r['ok'] else
                  f"dP'(2/1)={r['dp_diag'][0]:8.4f} gse_local_med={r['local_med']:.2e} int_med={r['int_med']:.2e}"),
                  flush=True)
    json.dump(res, open(os.path.join(ROOT, 'ifile_results.json'), 'w'), indent=1)
