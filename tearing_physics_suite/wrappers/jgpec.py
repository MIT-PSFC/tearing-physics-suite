"""Delta' from jGPEC (Julia GPEC, package GeneralizedPerturbedEquilibrium) as TPS codes jGPEC_galerkin / jGPEC_riccati.

jGPEC runs in one persistent Julia process per Python process (JuliaServer), so Julia compiles once per
worker. Inputs are mapped from the RDCON/STRIDE inputs so the domain and surfaces match RDCON.
"""
import atexit
import os
import selectors
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr

from tearing_physics_suite.utils import eq_stem
from tearing_physics_suite.wrappers.d3d_wall import write_jgpec_wall_file
from tearing_physics_suite.wrappers.gpec_common import save_output

SOLVERS = ('galerkin', 'riccati')
SERVER_JL = Path(__file__).with_name('jgpec_server.jl')
JULIA_INSTALL = '/fusion/projects/codes/julia/julia-1.11.7/bin/julia'
# Defaults where unset (`module load julia` is not used: its purge() would drop the compiler modules)
_JULIA_DEFAULTS = {
    'JULIA_DEPOT_PATH': '/fusion/projects/tmdb/julia_depot_benjamins:',
    'JULIA_CPU_TARGET': 'generic',
    'OPENBLAS_NUM_THREADS': '1',
}
EQUILIBRIUM_KEYS = {'eq_filename', 'eq_type', 'jac_type', 'grid_type', 'psilow', 'psihigh', 'mpsi', 'mtheta',
                    'newq0', 'etol', 'force_termination', 'psi_accuracy', 'profile_source', 'use_galgrid', 'r0exp',
                    'b0exp', 'imas_cocos', 'jac_custom_power_b', 'jac_custom_power_bp', 'jac_custom_power_r',
                    'jac_custom_power_rc'}
WALL_KEYS = {'shape', 'a', 'aw', 'bw', 'cw', 'dw', 'tw', 'equal_arc_wall'}


def jgpec_home():
    """jGPEC repository (Julia project): $JGPEC_HOME, else /fusion/projects/tmdb/src/GPEC."""
    return os.environ.get('JGPEC_HOME', '/fusion/projects/tmdb/src/GPEC')


def julia_exe():
    """Julia executable: $JGPEC_JULIA, else $JULIA_BINDIR/julia, else JULIA_INSTALL.

    (The module's bin/julia on PATH is a wrapper that needs the module's shell functions.)
    """
    if os.environ.get('JGPEC_JULIA'):
        return os.environ['JGPEC_JULIA']
    if os.environ.get('JULIA_BINDIR'):
        return os.path.join(os.environ['JULIA_BINDIR'], 'julia')
    return JULIA_INSTALL


def julia_env(threads=1):
    """Environment for the Julia process (module defaults filled in where unset)."""
    env = dict(os.environ)
    for k, v in _JULIA_DEFAULTS.items():
        env.setdefault(k, v)
    env['JULIA_NUM_THREADS'] = str(threads)
    env.pop('LD_LIBRARY_PATH', None)  # module libraries (e.g. PCRE) break Julia's bundled ones
    env.pop('JULIA_PROJECT', None)  # the module points it at the global environment; --project is passed
    return env


def precompile_jgpec(timeout=7200):
    """Instantiate and precompile jGPEC once (call before starting a pool of workers)."""
    cmd = [julia_exe(), f'--project={jgpec_home()}', '--startup-file=no', '-e',
           'using Pkg; Pkg.instantiate(); using GeneralizedPerturbedEquilibrium']
    subprocess.run(cmd, env=julia_env(), check=True, timeout=timeout)


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------
def _unquote(v):
    return str(v).strip().strip("'\"")


def _true(v):
    return str(v).strip().strip("'\".").lower() in ('t', 'true')


def _toml_value(v):
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_toml_value(x) for x in v) + ']'
    return '"' + str(v).replace('"', '\\"') + '"'


def gpec_toml_sections(eq_filename, nn, solver, inputs, **jgpec_kwargs):
    """{section: {key: value}} for gpec.toml, from the RDCON/STRIDE input dict (rdcon_stride_input_dict).

    jgpec_kwargs: '<key>_jgpec' sets <key> in [Equilibrium], [Wall] or [ForceFreeStates] (by key).
    """
    if solver not in SOLVERS:
        raise ValueError(f"jGPEC solver must be one of {SOLVERS}, not {solver!r}")
    vac = _true(inputs['vac_flag'])
    if solver == 'riccati' and not vac:
        raise ValueError("jGPEC Riccati gives no Delta' with a conducting wall on the plasma (vac_flag='f'); "
                         "use jgpec_solvers=('galerkin',).")
    profile_source = {'integrate': 'derivatives', 'values': 'values'}[_unquote(inputs.get('profile_source', 'integrate'))]
    eq = dict(eq_filename=eq_filename, eq_type=_unquote(inputs['eq_type']), jac_type=_unquote(inputs['jac_type']),
              grid_type=_unquote(inputs['grid_type']), psilow=float(inputs['psilow']), psihigh=float(inputs['psihigh']),
              mpsi=int(inputs['mpsi']), mtheta=int(inputs['mtheta']), newq0=float(inputs['newq0']),
              etol=float(inputs['etol']), profile_source=profile_source, force_termination=False)
    a_wall, ishape = float(inputs.get('a_wall', 0.0)), int(inputs.get('ishape', 6))
    if vac and ishape not in (6, 8):
        raise NotImplementedError(f"jGPEC wall for ishape={ishape}: only conformal (6) and DIII-D (8) walls are mapped.")
    # a >= 10: no wall in VACUUM. 'd3d': run_jgpec_solvers writes the DIII-D wall file (wrappers/d3d_wall.py).
    wall = dict(shape='nowall') if not vac or a_wall >= 10 else \
        dict(shape='conformal', a=a_wall) if ishape == 6 else dict(shape='d3d', equal_arc_wall=True)
    ffs = dict(vac_flag=vac, qlow=float(inputs['qlow']), qhigh=float(inputs['qhigh']), sing_start=int(inputs['sing_start']),
               nn_low=int(nn), nn_high=int(nn), delta_mlow=int(inputs['delta_mlow']), delta_mhigh=int(inputs['delta_mhigh']),
               mthvac=int(inputs['mthvac']), singfac_min=float(inputs['singfac_min']), ucrit=float(inputs['ucrit']),
               eulerlagrange_tolerance=float(min(inputs['tol_r'], inputs['tol_nr'])),
               set_psilim_via_dmlim=_true(inputs['sas_flag']), dmlim=float(inputs['dmlim']),
               integrator=solver, kinetic_factor=0.0, local_stability_flag=False, psiedge=1.0,
               force_termination=True, HDF5_filename='gpec.h5')
    if solver == 'galerkin':
        ffs.update(gal_nx=int(inputs['nx']), gal_nq=int(inputs['nq']), gal_pfac=float(inputs['pfac']),
                   gal_dx0=float(inputs['dx0']), gal_dx1=float(inputs['dx1']), gal_dx2=float(inputs['dx2']),
                   gal_cutoff=int(inputs['cutoff']), gal_tol=float(inputs['gal_tol']),
                   gal_dx1dx2_flag=_true(inputs['dx1dx2_flag']), gal_sing_order=int(inputs['sing_order']),
                   gal_sing_order_ceiling=_true(inputs['sing_order_ceiling']), gal_solver=_unquote(inputs['solver']),
                   gal_rpec_flag=False)  # rpec drops the vacuum edge
    sections = {'Equilibrium': eq, 'Wall': wall, 'ForceFreeStates': ffs}
    for k, v in jgpec_kwargs.items():
        key = k.removesuffix('_jgpec')
        sec = 'Equilibrium' if key in EQUILIBRIUM_KEYS else 'Wall' if key in WALL_KEYS else 'ForceFreeStates'
        sections[sec][key] = v
    return sections


def write_gpec_toml(run_dir, sections):
    """Write run_dir/gpec.toml from {section: {key: value}}."""
    lines = []
    for sec, kv in sections.items():
        lines.append(f'[{sec}]')
        lines += [f'{k} = {_toml_value(v)}' for k, v in kv.items()]
        lines.append('')
    Path(run_dir, 'gpec.toml').write_text('\n'.join(lines))


# ------------------------------------------------------------------------------------------------
# Persistent Julia server
# ------------------------------------------------------------------------------------------------
class JuliaServer:
    """One Julia process running jgpec_server.jl; run(dir) runs jGPEC on dir/gpec.toml."""

    def __init__(self, threads=1, log_path=None, startup_timeout=7200):
        self.threads, self.startup_timeout = threads, startup_timeout
        self.log_path = log_path or os.path.join(os.getcwd(), f'jgpec_server_{os.getpid()}.log')
        self.proc = None

    def start(self):
        log = open(self.log_path, 'a')
        self.proc = subprocess.Popen(
            [julia_exe(), f'--project={jgpec_home()}', '--startup-file=no', f'-t{self.threads}', str(SERVER_JL)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=julia_env(self.threads))
        log.close()
        self._readline_until('TPS_READY', self.startup_timeout)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _readline_until(self, prefix, timeout):
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        end = time.monotonic() + timeout
        try:
            while True:
                left = end - time.monotonic()
                if left <= 0 or not sel.select(left):
                    self.close(kill=True)
                    raise TimeoutError(f'jGPEC server: no {prefix} within {timeout} s (log {self.log_path})')
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'jGPEC server exited (log {self.log_path})')
                if line.startswith(prefix):
                    return line.strip()
        finally:
            sel.close()

    def run(self, run_dir, timeout=7200):
        """Run jGPEC in run_dir; returns (ok, message). Restarts the server if it died."""
        if not self.alive():
            self.start()
        self.proc.stdin.write(str(run_dir) + '\n')
        self.proc.stdin.flush()
        reply = self._readline_until('TPS_DONE', timeout).split(' ', 2)
        return reply[1] == 'ok', (reply[2] if len(reply) > 2 else '')

    def close(self, kill=False):
        if self.proc is None:
            return
        if self.alive():
            if kill:
                self.proc.kill()
            else:
                try:
                    self.proc.stdin.write('TPS_QUIT\n')
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    self.proc.kill()
        self.proc = None


_SERVER = None


def get_server(threads=1):
    """This process's JuliaServer (started on first use, closed at exit)."""
    global _SERVER
    if _SERVER is None:
        _SERVER = JuliaServer(threads=threads)
        atexit.register(_SERVER.close)
    return _SERVER


# ------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------
def read_jgpec_h5(path):
    """Delta' outputs of a jGPEC gpec.h5 in the RDCON schema, on the surfaces jGPEC solved across.

    Delta_prime, A_prime, B_prime, Gamma_prime (i, r_prime, r) with i = 0 real, 1 imaginary (A/B/Gamma:
    Galerkin only); psi_n_rational, q_rational (r); r = r_prime = m. Attributes n, psilim, qlim.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        ss = f['SingularSurfaces']
        if 'Delta_prime_matrix' not in ss:
            raise KeyError(f"{path}: no SingularSurfaces/Delta_prime_matrix (no Delta' computed)")
        psi = np.atleast_1d(ss['rational_psi'][()])
        q = np.atleast_1d(ss['rational_q'][()])
        m = np.atleast_2d(ss['rational_m'][()])[0]  # (mode, surface) as read
        n = int(np.atleast_2d(ss['rational_n'][()])[0, 0])
        # h5py reads Julia's column-major matrices transposed: [r_prime, r] as RDCON's netCDF
        mats = {name: ss[key][()] for name, key in (('Delta_prime', 'Delta_prime_matrix'), ('A_prime', 'pest3_A'),
                                                     ('B_prime', 'pest3_B'), ('Gamma_prime', 'pest3_Gamma')) if key in ss}
        info = {k: f['Info'][k][()] for k in ('psilim', 'qlim', 'mlow', 'mhigh')}
    keep = _solved_surfaces(psi, m, mats['Delta_prime'].shape[0], info, path)
    r = m[keep].astype(float)
    ds = xr.Dataset(
        {name: (('i', 'r_prime', 'r'), np.stack([mat.real, mat.imag])) for name, mat in mats.items()},
        coords={'i': [0, 1], 'r_prime': r, 'r': r})
    ds['psi_n_rational'] = ('r', psi[keep])
    ds['q_rational'] = ('r', q[keep])
    ds.attrs.update(n=n, psilim=float(info['psilim']), qlim=float(info['qlim']))
    return ds


def _solved_surfaces(psi, m, nsolved, info, path):
    """Indices of the rational surfaces the Delta' matrix is over: all (Riccati), or Galerkin's in-domain,
    in-band subset (psi < psilim, mlow <= m <= mhigh; jGPEC gal_resonant_surfaces)."""
    if nsolved == len(psi):
        return np.arange(len(psi))
    keep = np.flatnonzero((psi < info['psilim']) & (m >= info['mlow']) & (m <= info['mhigh']))
    if len(keep) != nsolved:
        raise ValueError(f"{path}: Delta' matrix has {nsolved} surfaces; cannot identify them among {len(psi)}")
    return keep


def check_jgpec_request(solvers, vac_flag, run_rdcon):
    """Validate jgpec_solvers before any code runs (Riccati has no Delta' with a wall on the plasma)."""
    solvers = (solvers,) if isinstance(solvers, str) else tuple(solvers)
    if not solvers or set(solvers) - set(SOLVERS):
        raise ValueError(f"jgpec_solvers must be a non-empty subset of {SOLVERS}, not {solvers!r}")
    if not run_rdcon:
        raise ValueError("run_jgpec needs run_rdcon: jGPEC surfaces are aligned to RDCON's.")
    if 'riccati' in solvers and not _true(vac_flag):
        if solvers == ('riccati',):
            raise ValueError("jGPEC Riccati gives no Delta' with a conducting wall on the plasma (vac_flag='f'); "
                             "use jgpec_solvers=('galerkin',).")
        print("WARNING: vac_flag='f': jGPEC Riccati is skipped (jGPEC_riccati is all NaN).")
    return solvers


def _nan_like(rdcon_xr):
    """All-NaN Delta' dataset on RDCON's surfaces (for a solver that cannot run)."""
    nan = xr.full_like(rdcon_xr['Delta_prime'], np.nan, dtype=float)
    return xr.Dataset({'Delta_prime': nan, 'psi_n_rational': rdcon_xr['psi_n_rational'],
                       'q_rational': rdcon_xr['q_rational']})


def run_jgpec_solvers(eq_filename, nn, working_dir, inputs, rdcon_xr, solvers=('galerkin',), threads=1,
                      timeout=7200, output_location=None, output_prefix='', override_save=True, verbose=False,
                      **jgpec_kwargs):
    """Run each jGPEC solver in working_dir/jgpec_<solver> and read its Delta'.

    inputs: the RDCON/STRIDE input dict; rdcon_xr: RDCON's output (for the all-NaN Riccati placeholder).
    Returns ({'jGPEC_<solver>': Dataset or None}, {'jGPEC_<solver>': ran}). Datasets are on jGPEC's own
    surfaces; physics.combine.align_surfaces puts them on RDCON's (after the Delta' coupling).
    """
    xrs, ran = {}, {}
    for solver in solvers:
        code = f'jGPEC_{solver}'
        xrs[code], ran[code] = None, False
        if solver == 'riccati' and not _true(inputs['vac_flag']):
            xrs[code] = _nan_like(rdcon_xr) if rdcon_xr is not None else None
            continue
        run_dir = os.path.join(working_dir, f'jgpec_{solver}')
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir)
        shutil.copy(eq_filename, run_dir)
        sections = gpec_toml_sections(os.path.basename(eq_filename), nn, solver, inputs, **jgpec_kwargs)
        if sections['Wall'].get('shape') == 'd3d':
            sections['Wall']['shape'] = write_jgpec_wall_file(os.path.join(run_dir, 'd3d_wall.dat'),
                                                              int(sections['ForceFreeStates']['mthvac']))
        write_gpec_toml(run_dir, sections)
        if verbose: print(f"Running jGPEC {solver} in {run_dir}...")
        try:
            ok, msg = get_server(threads).run(run_dir, timeout)
        except (TimeoutError, RuntimeError, OSError) as e:
            ok, msg = False, str(e)
        h5 = os.path.join(run_dir, 'gpec.h5')
        if ok:
            try:
                xrs[code], ran[code] = read_jgpec_h5(h5), True
            except (OSError, KeyError, ValueError) as e:
                msg = str(e)
        if not ran[code]:
            print(f"WARNING: jGPEC {solver} failed: {msg}")
        elif output_location is not None:
            os.makedirs(output_location, exist_ok=True)
            save_output(code, xrs[code], os.path.join(output_location, f'{output_prefix}{eq_stem(eq_filename)}_{code}_n{nn}.nc'),
                        override_save, verbose)
    return xrs, ran
