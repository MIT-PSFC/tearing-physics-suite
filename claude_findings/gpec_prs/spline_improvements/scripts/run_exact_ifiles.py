"""A6 follow-up: GPEC on i-files whose R,Z are exact FEM psi crossings (diag_ifile_noise.py output).

usage: python run_exact_ifiles.py BINDIR OUTROOT IFILE [IFILE ...]  (env MPSI sets mpsi, default 128)
"""
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from gpec_runs import run_stride, read_delta_prime, read_gsec, gse_metrics

BIN, ROOT, FILES = sys.argv[1], sys.argv[2], sys.argv[3:]
MPSI = os.environ.get('MPSI', '128')


def one(path):
    name = os.path.basename(path)[:-6]
    pad = float(name.split('_pad')[1].split('_')[0]) if '_pad' in name else 0.01
    out = os.path.join(ROOT, name + f'_m{MPSI}')
    ok = run_stride(BIN, out, path, eq_type='"ldp_i"', profile_source='"integrate"',
                    psihigh=f'{0.985 / (1 - pad):.6f}', mpsi=MPSI)
    if not ok:
        return name, None
    q, dp, total = read_delta_prime(out)
    m = gse_metrics(read_gsec(out))
    return name, {'bytes': os.path.getsize(path), 'dp_diag': dp.tolist(), 'total_energy': total,
                  **{k: v for k, v in m.items() if not k.endswith('profile')}}


if __name__ == '__main__':
    os.makedirs(ROOT, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(int(os.environ.get('NPROC', '6'))) as ex:
        for name, r in ex.map(one, FILES):
            res[name] = r
            print(f'{name:28s} mpsi={MPSI} ' + ('FAILED' if r is None else
                  f"dP'(2/1)={r['dp_diag'][0]:8.4f} gse_local_med={r['local_med']:.2e} int_med={r['int_med']:.2e}"),
                  flush=True)
    json.dump(res, open(os.path.join(ROOT, f'exact_results_m{MPSI}.json'), 'w'), indent=1)
