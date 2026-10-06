"""i-file (eq_type ldp_i, inverse.f) vs g-file (eq_type efit, direct.f) of the same equilibrium:
STRIDE Delta', GPEC GSE and file size, with profile_source = integrate.

usage: python run_ifile_vs_gfile.py BINDIR TRUTH OUTROOT [key=value ...]  (extra &EQUIL_CONTROL settings)
TRUTH is make_truth_equilibrium.py's OUT; psihigh puts the last surface at the same true psi_N = 0.985.
Writes OUTROOT/ifile_vs_gfile.json.
"""
import os, sys, json, glob
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from gpec_runs import run_stride, read_delta_prime, read_gsec, gse_metrics

BIN, TRUTH, ROOT = sys.argv[1:4]
EXTRA = dict(a.split('=', 1) for a in sys.argv[4:])
PSIHIGH = 0.985 / (1 - json.load(open(os.path.join(TRUTH, 'truth_meta.json')))['psi_pad'])
FILES = sorted(glob.glob(os.path.join(TRUTH, 'g*.geqdsk')) + glob.glob(os.path.join(TRUTH, 'i*.ifile')))


def one(path):
    name = os.path.basename(path).rsplit('.', 1)[0]
    eq_type = 'efit' if path.endswith('.geqdsk') else 'ldp_i'
    out = os.path.join(ROOT, name)
    ok = run_stride(BIN, out, path, eq_type=f'"{eq_type}"', profile_source='"integrate"',
                    psihigh=f'{PSIHIGH:.6f}', **EXTRA)
    r = {'bytes': os.path.getsize(path), 'eq_type': eq_type, 'ok': ok}
    if ok:
        q, dp, total = read_delta_prime(out)
        m = gse_metrics(read_gsec(out))
        r.update(dp_diag=dp.tolist(), **{k: v for k, v in m.items() if not k.endswith('profile')})
    return name, r


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(int(os.environ.get('NPROC', '6'))) as ex:
        for name, r in ex.map(one, FILES):
            res[name] = r
            print(f"{name:10s} {r['bytes']:9d} B  " + ('FAILED' if not r['ok'] else
                  f"dP'(2/1)={r['dp_diag'][0]:.4f} gse_local_med={r['local_med']:.2e} int_med={r['int_med']:.2e}"),
                  flush=True)
    json.dump(res, open(os.path.join(ROOT, 'ifile_vs_gfile.json'), 'w'), indent=1)
