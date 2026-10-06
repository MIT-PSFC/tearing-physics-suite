"""Where does the noise in OFT i-file R,Z(psi,theta) come from: surface tracing or the FEM psi itself?

Re-solves the truth equilibrium, then for each point of an i-file finds the exact psi = psi_k crossing
along the same geometric-angle ray from the axis by Newton iteration on TokaMaker's FEM psi
(get_field_eval 'psi'/'dPSI'). Reports |rho_file - rho_newton| and the independent GS residual of
the file and of the Newton-refined points.

usage: python diag_ifile_noise.py TRUTH IFILE [IFILE ...]   (TRUTH: make_truth_equilibrium.py OUT)
Writes <name>_exact.ifile (Newton-refined R,Z plus truth FF', p' records) in the working directory.
"""
import os, sys, json
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../spline_improvements/scripts'))
from make_truth_equilibrium import solve_truth
from ifile_tools import read_ifile, write_ifile, gs_residual

TRUTH = sys.argv[1]
meta = json.load(open(os.path.join(TRUTH, 'truth_meta.json')))
mygs, _ = solve_truth(meta['source'], meta['mesh'])
fpsi, fgrad = mygs.get_field_eval('psi'), mygs.get_field_eval('dPSI')
truth = np.load(os.path.join(TRUTH, 'truth_profiles.npz'))
pb = meta['psi_bounds']


def refine(d, its=8):
    """Newton along fixed rays for every surface except the axis."""
    ra, za = d['R'][0, 0], d['Z'][0, 0]
    dr, dz = d['R'][1:] - ra, d['Z'][1:] - za
    rho = np.hypot(dr, dz)
    c, s = dr / rho, dz / rho
    target = np.repeat(d['psi'][1:, None], rho.shape[1], axis=1)
    for _ in range(its):
        pts = np.column_stack([(ra + rho * c).ravel(), (za + rho * s).ravel()])
        f = fpsi.eval(pts)[:, 0].reshape(rho.shape) - target
        g = fgrad.eval(pts)[:, :2]
        dfdrho = (g[:, 0] * c.ravel() + g[:, 1] * s.ravel()).reshape(rho.shape)
        rho = rho - f / dfdrho
    out = dict(d)
    out['R'] = d['R'].copy(); out['Z'] = d['Z'].copy()
    out['R'][1:] = ra + rho * c
    out['Z'][1:] = za + rho * s
    return out, np.abs(np.hypot(dr, dz) - rho), np.abs(f).max()


for path in sys.argv[2:]:
    d = read_ifile(path)
    x = (pb[1] - d['psi']) / (pb[1] - pb[0])
    ffp = CubicSpline(truth['psi_N'], truth['FFp'])(x)
    pp = CubicSpline(truth['psi_N'], truth['pp'])(x)
    e, drho, fres = refine(d)
    sel = (x > 0.1) & (x < 0.9)
    r0, r1 = gs_residual(d, ffp, pp), gs_residual(e, ffp, pp)
    print(f'{os.path.basename(path)}: |rho_file - rho_exact| median {np.median(drho):.1e} max {drho.max():.1e} m '
          f'(Newton |psi err| {fres:.1e}); GS residual median file {np.median(r0[sel]):.1e} -> exact-crossing '
          f'{np.median(r1[sel]):.1e}', flush=True)
    e['ffp'], e['pp'] = ffp, pp
    write_ifile(os.path.basename(path).replace('.ifile', '_exact.ifile'), e)
