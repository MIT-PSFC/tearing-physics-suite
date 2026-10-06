"""Run STRIDE on an equilibrium file with equil.in overrides, and read back Delta' and the GSE diagnostic."""
import os, re, shutil, subprocess
import numpy as np
import xarray as xr
from scipy.io import FortranFile

TEMPLATE = os.path.join(os.environ.get('WT', ''), 'spline_improvements/docs/examples/DIIID_ideal_example')


def set_namelist(path, group, **kv):
    """Set (or add) entries of a Fortran namelist group in place."""
    s = open(path).read()
    m = re.search(r'&' + group + r'\b(.*?)\n\s*/', s, re.I | re.S)
    body = m.group(1)
    for k, v in kv.items():
        pat = re.compile(r'^(\s*)' + re.escape(k) + r'\s*=.*$', re.I | re.M)
        line = f'{k}={v}'
        body = pat.sub(lambda mm: mm.group(1) + line, body) if pat.search(body) else body + '\n    ' + line
    open(path, 'w').write(s[:m.start(1)] + body + s[m.end(1):])


def run_stride(bindir, out, eqfile, template=TEMPLATE, threads=2, **equil):
    """Run stride in a fresh directory `out`; equil overrides go into &EQUIL_CONTROL (strings pre-quoted)."""
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in os.listdir(template):
        if f.endswith('.in'):
            shutil.copy(os.path.join(template, f), out)
    shutil.copy(eqfile, out)
    set_namelist(os.path.join(out, 'equil.in'), 'EQUIL_CONTROL', eq_filename=f'"{os.path.basename(eqfile)}"', **equil)
    set_namelist(os.path.join(out, 'equil.in'), 'EQUIL_OUTPUT', gse_flag='t')
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with open(os.path.join(out, 'stride.log'), 'w') as log:
        subprocess.run([os.path.join(bindir, 'stride')], cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env)
    return 'Normal termination' in open(os.path.join(out, 'stride.log')).read()


def read_delta_prime(out):
    """q_rational, diagonal Delta' (real), total free-boundary energy."""
    d = xr.open_dataset(os.path.join(out, 'stride_output_n1.nc'))
    dp = d.Delta_prime.isel(i=0).values
    return d.q_rational.values, np.diag(dp).copy(), float(d.attrs['total1'])


def read_gsec(out):
    """GPEC's GSE diagnostic (gsec.bin): psi_n, theta-grid arrays of flux terms, source, residual, error."""
    f = FortranFile(os.path.join(out, 'gsec.bin'), 'r')
    f.read_ints('<i4')
    mpsi, mtheta = f.read_ints('<i4')
    shape = (mtheta + 1, mpsi + 1)
    rz = f.read_reals('<f4')
    names = ['fsx', 'fsy', 'source', 'total', 'error', 'errlog']
    res = {n: f.read_reals('<f4').reshape(shape).T for n in names}
    n = shape[0] * shape[1]
    res['r'] = rz[:n].reshape(shape).T
    res['z'] = rz[n:].reshape(shape).T
    res['psi_n'] = xr.open_dataset(os.path.join(out, 'stride_output_n1.nc')).psi_n.values
    return res


def gse_metrics(g, lo=0.05, hi=0.95):
    """Surface-wise GSE: max over theta of the normalized local residual, and the
    theta-integrated residual relative to the theta-integrated |source|; summarized over lo < psi_n < hi."""
    sel = (g['psi_n'] > lo) & (g['psi_n'] < hi)
    local = g['error'].max(axis=1)
    integ = np.abs(g['total'].sum(axis=1)) / np.abs(g['source']).sum(axis=1)
    return {'local_max': float(local[sel].max()), 'local_med': float(np.median(local[sel])),
            'int_max': float(integ[sel].max()), 'int_med': float(np.median(integ[sel])),
            'local_profile': local, 'int_profile': integ}
