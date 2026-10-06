"""Write the reference equilibrium as i-files over a grid of resolutions with this OpenFUSIONToolkit's save_ifile.

usage: python write_ifiles.py TRUTH OUT [NPSIxNTHETA ...]   (TRUTH: make_truth_equilibrium.py OUT)
Default grid: npsi in 65, 129, 257 x ntheta in 65, 129, 257, 513; lcfs_pad as in TRUTH.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../spline_improvements/scripts'))
from make_truth_equilibrium import solve_truth

TRUTH, OUT = sys.argv[1:3]
GRID = [tuple(int(n) for n in a.split('x')) for a in sys.argv[3:]] or \
    [(npsi, nth) for npsi in (65, 129, 257) for nth in (65, 129, 257, 513)]
meta = json.load(open(os.path.join(TRUTH, 'truth_meta.json')))
mygs, _ = solve_truth(meta['source'], meta['mesh'], int(os.environ.get('NTHREADS', '4')))
os.makedirs(OUT, exist_ok=True)
for npsi, nth in GRID:
    mygs.save_ifile(os.path.join(OUT, f'i{npsi}x{nth}.ifile'), npsi=npsi, ntheta=nth, lcfs_pad=meta['psi_pad'])
print('wrote', OUT, flush=True)
