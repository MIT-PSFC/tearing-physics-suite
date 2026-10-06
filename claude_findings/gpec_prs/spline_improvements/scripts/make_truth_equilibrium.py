"""Solve a DIII-D-like TokaMaker equilibrium with known profiles and write it in every format we compare.

The FF' and p' shapes, Ip, F0, and boundary come from SOURCE (inputs/D3Dlike_Hmode_baseline.geqdsk,
a TokaMaker D3D-like H-mode); MESH is the TokaMaker DIII-D mesh (inputs/DIIID_mesh.h5).
The shapes are interpolated with a C2 cubic spline onto NODES (default 4001) points and given
to TokaMaker as piecewise-linear ('linterp') profiles, so F'' is smooth to within ~1/NODES**2.
TokaMaker's F, F', p, p' are exact at any psi: they are the ground truth for the g-file and
i-file round trips.

Writes into OUT:
  truth_profiles.npz        psi_N, F, F', FF', p, p' (dF/dpsi_N etc., psi in Wb/rad) on a fine grid
  g{N}.geqdsk               save_eqdsk at nr = nz = N
  i{NPSI}x{NTHETA}.ifile    save_ifile (with FF', p' records) at 65x129, 129x257, 257x513
  truth_meta.json           psi bounds, F0, padding, stats, SOURCE and MESH paths

usage: python make_truth_equilibrium.py SOURCE MESH OUT   (env NTHREADS, NODES)
"""
import os, sys, json
import numpy as np
from scipy.interpolate import CubicSpline
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

PSI_PAD = 1e-3


def solve_truth(source, mesh, nthreads=4):
    """Solve the truth equilibrium; returns (TokaMaker, source g-file dict)."""
    g = read_eqdsk(source)
    psi_N = np.linspace(0.0, 1.0, g['nr'])
    F0 = abs(g['rcentr'] * g['bcentr'])

    myOFT = OFT_env(nthreads=nthreads)
    mygs = TokaMaker(myOFT)
    mp, ml, mr, cd, cnd = load_gs_mesh(mesh)
    mygs.setup_mesh(mp, ml, mr)
    mygs.setup_regions(cond_dict=cnd, coil_dict=cd)
    mygs.setup(order=3, F0=F0)
    mygs.settings.maxits = 800
    mygs.settings.pm = False
    mygs.update_settings()
    mygs.set_coil_vsc({'F9A': 1.0, 'F9B': -1.0})
    reg = [mygs.coil_reg_term({n: 1.0}, target=0.0, weight=1.0) for n in mygs.coil_sets]
    reg.append(mygs.coil_reg_term({'#VSC': 1.0}, target=0.0, weight=1e-2))
    mygs.set_coil_reg(reg_terms=reg)

    # boundary: isoflux at 32 even angles about the centroid, saddle at the lower X-point
    bR, bZ = g['rzout'][:, 0], g['rzout'][:, 1]
    R0, Z0 = 0.5 * (bR.max() + bR.min()), bZ.mean()
    ang = np.arctan2(bZ - Z0, bR - R0)
    sel = [int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - t)))))) for t in np.linspace(-np.pi, np.pi, 32, endpoint=False)]
    mygs.set_isoflux(np.column_stack([bR[sel], bZ[sel]]), weights=200 * np.ones(len(sel)))
    ix = int(np.argmin(bZ))
    mygs.set_saddle_constraints(np.array([[bR[ix], bZ[ix]]]), np.array([200.0]))

    # profile shapes from the file; TokaMaker rescales them to Ip and p_axis
    xn = np.linspace(0.0, 1.0, int(os.environ.get('NODES', '4001')))
    mygs.set_profiles(ffp_prof={'type': 'linterp', 'x': xn, 'y': CubicSpline(psi_N, g['ffprim'] / g['ffprim'][0])(xn)},
                      pp_prof={'type': 'linterp', 'x': xn, 'y': CubicSpline(psi_N, g['pprime'] / g['pprime'][0])(xn)})
    mygs.set_targets(Ip=abs(g['ip']), pax=g['pres'][0])
    a = 0.5 * (bR.max() - bR.min())
    mygs.init_psi(R0, Z0, a, (bZ.max() - bZ.min()) / (2 * a), 0.4)
    mygs.solve()
    return mygs, g


if __name__ == '__main__':
    SOURCE, MESH, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
    os.makedirs(OUT, exist_ok=True)
    mygs, g = solve_truth(SOURCE, MESH, int(os.environ.get('NTHREADS', '4')))
    stats = mygs.get_stats(lcfs_pad=PSI_PAD)
    print({k: stats[k] for k in ('Ip', 'q_0', 'q_95', 'l_i', 'beta_n') if k in stats}, flush=True)

    # truth profiles on a fine grid; get_profiles returns F'(psi) with psi in Wb/rad
    x = np.linspace(0.0, 1.0, 20001)
    _, F, Fp, p, pp = mygs.get_profiles(psi=x)
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
    np.savez(os.path.join(OUT, 'truth_profiles.npz'), psi_N=x, F=F, Fp=Fp, FFp=F * Fp, p=p, pp=pp,
             psi_bounds=np.array(mygs.psi_bounds), psi_range=psi_range)

    for n in (129, 257, 513):
        mygs.save_eqdsk(os.path.join(OUT, f'g{n}.geqdsk'), nr=n, nz=n, lcfs_pad=PSI_PAD, truncate_eq=True)
    for npsi, nth in ((65, 129), (129, 257), (257, 513)):
        mygs.save_ifile(os.path.join(OUT, f'i{npsi}x{nth}.ifile'), npsi=npsi, ntheta=nth, lcfs_pad=PSI_PAD)
    json.dump({'psi_bounds': list(mygs.psi_bounds), 'F0': abs(g['rcentr'] * g['bcentr']), 'psi_pad': PSI_PAD,
               'source': SOURCE, 'mesh': MESH,
               'stats': {k: float(v) for k, v in stats.items() if np.isscalar(v)}},
              open(os.path.join(OUT, 'truth_meta.json'), 'w'), indent=1)
    print('wrote', OUT, flush=True)
