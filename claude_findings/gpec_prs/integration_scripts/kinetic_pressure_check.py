"""B.5 (Issue 3.3): TPS's kinetic interpolants against the pressure OFT solved with, on a bouquet archive.

For each draw: p_kin = e (n_e T_e + n_i T_i) (bouquet's pressure_thermal convention: impurity and
fast-ion pressure are in pressure - pressure_thermal) from TPS's splines (pchip, as bouquet's own
pp_prof; and Akima, TPS's previous choice) against the archived thermal pressure (the TokaMaker
solve's pressure without fast ions) on the equilibrium psi_N grid.

usage: python kinetic_pressure_check.py ARCHIVE.h5 [n_draws]
"""
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
import bouquet as bq

E = 1.602176634e-19
ar = bq.BouquetArchive(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for key in ar.scan_keys:
    for d in ar[key].selected[:n]:
        pr, at = d.profiles, d.attrs
        x = pr.get('psi_N_kinetic', pr['psi_N'])
        xf = np.linspace(x[0], x[-1], 4001)
        pk = {}
        for name, I in (('pchip', PchipInterpolator), ('akima', Akima1DInterpolator)):
            s = lambda y: I(x, y)(xf)
            pk[name] = E * (s(pr['n_e']) * s(pr['T_e']) + s(pr['n_i']) * s(pr['T_i']))
        # archived thermal pressure: the solve's p minus fast ions, on the equilibrium psi_N grid
        xe = pr['psi_N']
        sel = (xe >= x[0]) & (xe <= x[-1])
        pth = pr['pressure_thermal'][sel]
        pke = {k: np.interp(xe[sel], xf, v) for k, v in pk.items()}
        pmax = np.max(pth)
        msg = (f"{key}/{d.count}: |pchip-akima|max/pmax = {np.max(np.abs(pke['pchip']-pke['akima']))/pmax:.1e}, "
               f"|p_kin - p_thermal|max/pmax: pchip {np.max(np.abs(pke['pchip']-pth))/pmax:.1e}, "
               f"akima {np.max(np.abs(pke['akima']-pth))/pmax:.1e}")
        print(msg, flush=True)
