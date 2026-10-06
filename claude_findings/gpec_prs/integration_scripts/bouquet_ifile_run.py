"""B.5: a real bouquet run with write_ifile=True, read back through TPS for both equilibrium sources.

usage: python bouquet_ifile_run.py OUT_DIR GPEC_DIR [n_draws]
Uses bouquet's D3D-like example (g-file, p-file, mesh). Needs bouquet (OFT_inverse) and an OFT with
GPECf_interface on PYTHONPATH, and TPS importable.
"""
import os, sys, json
import bouquet as bq
from tearing_physics_suite.profile_read import read_bouquet_archive
from tearing_physics_suite import fortran_wrappers as tfw

out, gpec_dir = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ex = os.path.join(os.path.dirname(bq.__file__), '..', 'examples', 'D3D-like')
os.makedirs(out, exist_ok=True)
os.chdir(out)
b = bq.Bouquet.from_geqdsk(os.path.join(ex, 'D3Dlike_Hmode_baseline.geqdsk'),
                           profiles=os.path.join(ex, 'D3Dlike_Hmode_baseline.peqdsk'),
                           mesh=os.path.join(ex, 'DIIID_mesh.h5'), nthreads=4,
                           header=os.path.join(out, 'ifrun'), n_draws=n)
b.config.generation.write_ifile = True
b.setup_solver()
b.prepare_baseline()
b.generate(n=n)

res = {}
for src in ('geqdsk', 'ifile'):
    files, profs = read_bouquet_archive(os.path.join(out, 'ifrun.h5'), selection='all', eq_source=src,
                                        eqdsk_out_dir=os.path.join(out, 'TPS_' + src))
    pad = b.config.source.psi_pad if hasattr(b.config.source, 'psi_pad') else 1e-3
    wd = os.path.join(out, 'run_' + src)
    os.makedirs(wd, exist_ok=True)
    rd, st, _, rd_ran, st_ran, _, _, _ = tfw.run_resistive_calculation(
        files[0], 1, working_dir=wd, gpec_dir=gpec_dir, run_rdcon=True, run_stride=True, run_pest3=False,
        psihigh=0.985 / (1 - pad), output_location=wd, verbose=False)
    res[src] = {'file': files[0], 'bytes': os.path.getsize(files[0]),
                'rdcon_dp21': float(rd.Delta_prime.isel(r=0, r_prime=0, i=0)) if rd_ran else None,
                'stride_dp21': float(st.Delta_prime.isel(r=0, r_prime=0, i=0)) if st_ran else None}
    print(src, res[src], flush=True)
json.dump(res, open(os.path.join(out, 'bouquet_ifile_run.json'), 'w'), indent=1)
