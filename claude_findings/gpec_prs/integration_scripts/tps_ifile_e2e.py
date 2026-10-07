"""B.5: the same TokaMaker equilibrium through TPS run_resistive_calculation as a g-file and as an i-file.

usage: python tps_ifile_e2e.py GPEC_DIR TRUTH_DIR OUT_DIR
GPEC_DIR is a GPEC tree with bin/ (e.g. the OFT_interface worktree); TRUTH_DIR holds g257.geqdsk and
i129x257.ifile written by make_truth_equilibrium.py with an OFT that has GPECf_interface. The i-file is
picked up as eq_type 'ldp_i' from its extension. psihigh is set for the same true psi_N in both.
"""
import os, sys, json
import numpy as np
from tearing_physics_suite.wrappers import run_codes as tfw

gpec_dir, truth, out = sys.argv[1:4]
pad_g = json.load(open(os.path.join(truth, 'truth_meta.json')))['psi_pad']
cases = {'geqdsk': ('g257.geqdsk', 0.985 / (1 - pad_g)), 'ifile': ('i129x257.ifile', 0.985 / (1 - 0.01))}
res = {}
for name, (fname, psihigh) in cases.items():
    wd = os.path.join(out, name)
    os.makedirs(wd, exist_ok=True)
    rd, st, _, rd_ran, st_ran = tfw.run_resistive_calculation(
        os.path.join(truth, fname), 1, working_dir=wd, gpec_dir=gpec_dir, run_rdcon=True, run_stride=True,
        run_pest3=False, psihigh=psihigh, output_location=wd, verbose=False)[:5]
    res[name] = {'rdcon_ran': rd_ran, 'stride_ran': st_ran,
                 'rdcon_dp21': float(rd.Delta_prime.isel(r=0, r_prime=0, i=0)) if rd_ran else None,
                 'stride_dp21': float(st.Delta_prime.isel(r=0, r_prime=0, i=0)) if st_ran else None}
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(out, 'tps_ifile_e2e.json'), 'w'), indent=1)
