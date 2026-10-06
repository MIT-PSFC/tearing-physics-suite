"""Compare profile_source settings on direct (g-file, efit) and inverse (i-file, ldp_i) equilibria:
STRIDE Delta' and GPEC GSE.

usage: python run_profile_source_comparison.py BINDIR TRUTH OUTROOT [key=value ...]  (extra &EQUIL_CONTROL settings)
Cases: the TokaMaker truth g-files (129/257/513) and i-files (65x129/129x257/257x513), with psihigh set so the
last surface is at the same true psi_N = 0.985; GPEC's TkMkr D3D-like example; and g147131 (classic EFIT;
env G147131, e.g. inputs/g147131.02300_DIIID_KEFIT, skipped if unset).
env METHODS (default values,integrate).
Writes OUTROOT/profile_source_results.json.
"""
import os, sys, json, glob
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from gpec_runs import run_stride, read_delta_prime, read_gsec, gse_metrics, example_dir

BIN, TRUTH, ROOT = sys.argv[1:4]
EXTRA = dict(a.split('=', 1) for a in sys.argv[4:])
PSI_TRUE = 0.985
PSIHIGH = PSI_TRUE / (1 - json.load(open(os.path.join(TRUTH, 'truth_meta.json')))['psi_pad'])
EQS = {os.path.basename(f).rsplit('.', 1)[0]: (f, 'efit' if f.endswith('.geqdsk') else 'ldp_i', PSIHIGH)
       for f in sorted(glob.glob(os.path.join(TRUTH, 'g*.geqdsk')) + glob.glob(os.path.join(TRUTH, 'i*.ifile')))}
EQS['TkMkr'] = (os.path.join(example_dir(BIN), 'TkMkr_D3Dlike_Hmode.geqdsk'), 'efit', 0.995)
if os.environ.get('G147131'):
    EQS['g147131'] = (os.path.abspath(os.environ['G147131']), 'efit', 0.993)
METHODS = os.environ.get('METHODS', 'values,integrate').split(',')


def one(args):
    eq, method = args
    path, eq_type, psihigh = EQS[eq]
    out = os.path.join(ROOT, f'{eq}_{method}')
    ok = run_stride(BIN, out, path, eq_type=f'"{eq_type}"', profile_source=f'"{method}"',
                    psihigh=f'{psihigh:.6f}', **EXTRA)
    if not ok:
        return eq, method, None
    q, dp, total = read_delta_prime(out)
    m = gse_metrics(read_gsec(out))
    return eq, method, {'bytes': os.path.getsize(path), 'q': q.tolist(), 'dp_diag': dp.tolist(),
                        'total_energy': total, **{k: v for k, v in m.items() if not k.endswith('profile')}}


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    jobs = [(e, m) for e in EQS for m in METHODS]
    res = {}
    with ProcessPoolExecutor(int(os.environ.get('NPROC', '6'))) as ex:
        for eq, method, r in ex.map(one, jobs):
            res.setdefault(eq, {})[method] = r
            print(eq, method, 'FAILED' if r is None else
                  f"dP'(2/1)={r['dp_diag'][0]:.4f} gse_local_med={r['local_med']:.2e} gse_int_med={r['int_med']:.2e}",
                  flush=True)
    json.dump(res, open(os.path.join(ROOT, 'profile_source_results.json'), 'w'), indent=1)
