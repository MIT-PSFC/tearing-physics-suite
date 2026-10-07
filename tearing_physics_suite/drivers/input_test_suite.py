# Python functions to test the numerical and physics stability of delta' calculations for a given equilibrium
# Tests scan all input parameters using the scan_1D_input function. Refer to input_test_runner.py to run multiple tests at once.

# README:
#   All computations are done with the default wrapper inputvalues unless otherwise specified.
#   All computation is internal mode (vac_flag = 'f') unless otherwise specified.
#   All computations are done with nn=1 unless otherwise specified.
# quick_test = True <=> deviate slightly from default values to make sure no large changes are detected.
# quick_test = False <=> Proper scoping over reasonable the reasonable range for a given input parameter.
#
# Each scan is one ScanSpec row in SCANS; run it with run_scan(name, eq_filename, ...).

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from tearing_physics_suite.drivers.input_scans import scan_1D_input
from tearing_physics_suite.utils import create_dense_log_paramvals, tps_home

BANNER = "#" * 105


@dataclass(frozen=True)
class ScanSpec:
    """One 1D input scan.

    input_name: scanned input, or a function of the scan values returning it.
    subdir: output folder under results_dir/1D_scans; title: banner printed with the result.
    fixed: kwargs always passed to scan_1D_input (passing them again raises TypeError).
    defaults: kwargs passed unless the caller overrides them.
    prefix: output_prefix for this scan (e.g. 'no_wall_'), after any caller prefix.
    superquick_vals: values used when superquick=True (only scans that support it).
    debug_fixed: kwargs replacing fixed ones when debug=True (only scans that support it).
    full_note: printed when the full default scan is run.
    """
    name: str
    input_name: str | Callable
    quick_vals: list
    full_vals: list
    subdir: str
    title: str
    fixed: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    nn: int = 1
    prefix: str = ''
    superquick_vals: list | None = None
    debug_fixed: dict | None = None
    full_note: str = ''


def _pest3_nx_name(vals):
    # If scan_vals is provided, check first element to determine whether we're scanning over nx_pest or nx_string_pest
    return 'nx_string_pest' if isinstance(vals[0], str) else 'nx_pest'


_TRUNC_NOTE = ("Warning: it's likely the q-surface being truncated will jump by one over the course of this scan. "
               "Read {} & fiddle with psihigh to stop this happening.")
WALL = dict(vac_flag='f')
NO_WALL = dict(vac_flag='t')
NO_PEST = dict(run_pest3=False)
RDCON_ONLY = dict(run_pest3=False, run_stride=False)

# Order matches the original scan_functions list (the default run order).
_SPECS = [
    # NUMERICAL STABILITY TESTS
    # Varying number of fourier modes used in analysis:
    ScanSpec('fourier_mode_scan', 'delta_mhigh', [7, 8], list(range(9)), 'fourier_mode_scan', 'Fourier mode scan results:',
             fixed=dict(set_delta_mlow_to_delta_mhigh=True, **WALL)),
    # Varying solver:
    ScanSpec('RDCON_solver_scan', 'solver', ["'LU'", "'cholesky'"], ["'LU'", "'cholesky'"], 'solver_scan',
             'RDCON solver scan results:', fixed=dict(**WALL, **RDCON_ONLY)),
    # Varying order of asymptotic expansion.
    # Note: PEST3 doesn't have an option to vary sing_order, so we don't run it here.
    ScanSpec('sing_order_scan', 'sing_order', [5, 6], list(range(7)), 'sing_order_scan',
             'Asymptotic expansion order scan results:', fixed=dict(**WALL, **NO_PEST)),
    # Varying asmpyotic matching point:
    ScanSpec('matching_point_scan', 'dx0', [1e-3, 2e-3],
             create_dense_log_paramvals(start=1e-1, end=1e-6, points_per_decade=2), 'matching_point_scan',
             'Matching point scan results:', fixed=dict(set_dx1dx2_dx0_mult=2, set_singfac_min_to_dx=True, **WALL, **NO_PEST)),
    # Varying psilow truncation point. delta_prime_variability-applicable in most cases.
    ScanSpec('psilow_truncation_scan', 'psilow', [1e-4, 2e-4],
             create_dense_log_paramvals(start=1e-1, end=1e-5, points_per_decade=2), 'psilow_truncation_scan',
             'Psilow truncation scan results:', fixed=dict(**WALL, **NO_PEST)),
    # Varying equilibrium integration tolerance:
    ScanSpec('equilibrium_integrator_tolerance_scan', 'etol', [1e-9, 1e-10], [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
             'equilibrium_integrator_tolerance_scan', 'Equilibrium integrator tolerance scan results:', fixed=dict(**NO_PEST, **WALL)),
    # Varying mtheta:
    ScanSpec('mtheta_scan', 'mtheta', [129, 130, 257, 513], [70, 90, 129, 257, 513, 1025], 'mtheta_scan',
             'mtheta scan results:', fixed=dict(**WALL, pest_pull_mtheta=True)),
    ScanSpec('mtheta_scan_no_wall', 'mtheta', [129, 130, 257, 513], [70, 90, 129, 257, 513, 1025], 'mtheta_scan',
             'mtheta scan results (no wall):', fixed=dict(**NO_WALL, pest_pull_mtheta=True), prefix='no_wall_'),
    # Varying mpsi:
    ScanSpec('mpsi_scan', 'mpsi', [257, 258, 513], [70, 90, 129, 257, 513, 1025], 'mpsi_scan', 'mpsi scan results:',
             fixed=WALL),
    ScanSpec('mpsi_scan_no_wall', 'mpsi', [257, 258, 513], [70, 90, 129, 257, 513, 1025], 'mpsi_scan',
             'mpsi scan results (no wall):', fixed=NO_WALL, prefix='no_wall_'),
    # Varying psihigh truncation within q surface:
    ScanSpec('edge_truncation_within_surface_scan', 'dmlim', [0.15, 0.2],
             [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'edge_truncation_within_surface_scan',
             'Psihigh truncation within q surface scan results:', fixed=WALL, defaults=dict(psihigh=0.998),
             full_note=_TRUNC_NOTE.format('edge_truncation_within_surface_scan')),
    ScanSpec('edge_truncation_within_surface_scan_no_wall', 'dmlim', [0.15, 0.2],
             [1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'edge_truncation_within_surface_scan',
             'Psihigh truncation within q surface scan results (no wall):', fixed=NO_WALL, prefix='no_wall_',
             full_note=_TRUNC_NOTE.format('edge_truncation_within_surface_scan_no_wall')),
    # Varying number of finite elements: RDCON
    ScanSpec('rdcon_finite_element_scan', 'nx', [257, 128], [60, 70, 80, 90, 100, 120, 257, 512], 'nx_rdcon_scan',
             'RDCON finite element scan results:', fixed=dict(**RDCON_ONLY, **WALL)),
    # Varying number of finite elements: PEST3.
    # The correct Delta prime value is calculated from extrapolating a straight line in nx_pest^(-2) to nx_pest = infinity (nx_pest^(-2) = 0).
    # HOWEVER, above a certain value of nx_pest (~ > 140 < 200), the Delta prime values no longer obey a straight
    # line relationship w.r.t nx_pest^(-2). (Higher res. equilibria push this point to higher nx_pest values - Dylan Brennan). You want
    # to use values of nx_pest that are below this point to ensure convergence is correctly calculated.
    ScanSpec('pest3_finite_element_scan', _pest3_nx_name,
             ['-k"100 50 80 140"', '-k"90 46 70 130"', '-k"70 30 50 100"'],
             [30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 257, 300, 400, 512], 'nx_pest3_scan',
             'PEST3 finite element scan results:',
             fixed=dict(run_pest3=True, run_stride=True, run_rdcon=False, gal_flag='t', ode_flag='f', **WALL),
             debug_fixed=dict(gal_flag='f')),  # the old debug branch also set an unused run_stride=False
    # Varying integration tolerance near rational surfaces:
    ScanSpec('integrator_tolerance_scan', 'tol_r', [1e-9, 1e-10], [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
             'integrator_tolerance_scan', 'Integrator tolerance scan results:',
             fixed=dict(set_int_tolerances_equal=True, **NO_PEST, **WALL)),
    # Varying magnetic coordinate type.
    # Note: PEST3 doesn't have an option to vary jac_type, so we don't run it here.
    ScanSpec('mag_coord_type_scan', 'jac_type', ["'hamada'", "'pest'"], ["'hamada'", "'boozer'", "'pest'", "'equal_arc'"],
             'mag_coord_type_scan', 'Magnetic coordinate type scan results:', fixed=WALL),
    # Varying grid_type.
    # Note: PEST3 doesn't have an option to vary grid_type, so we don't run it here.
    ScanSpec('grid_type_scan', 'grid_type', ["'ldp'", "'pow1'"], ["'ldp'", "'pow1'", "'pow2'", "'rho'", "'original'"],
             'grid_pack_scan', 'Grid type scan results:', fixed=dict(**WALL, **NO_PEST)),
    # Varying vacuum theta spline density:
    ScanSpec('vacuum_mtheta_scan', 'mthvac', [960, 1200],
             [80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000], 'mthvac_scan',
             'Vacuum theta spline density scan results:', fixed=dict(**NO_WALL, a_wall=0.1, **NO_PEST)),
    # Varying cutoff: n = 1, n = 4
    ScanSpec('RDCON_cutoff_scan_n1', 'cutoff', [9, 10], list(range(1, 15)), 'RDCON_cutoff_scan',
             'RDCON cutoff scan results: n = 1', fixed=dict(**WALL, **RDCON_ONLY)),
    ScanSpec('RDCON_cutoff_scan_n4', 'cutoff', [9, 10], list(range(1, 17)), 'RDCON_cutoff_scan',
             'RDCON cutoff scan results: n = 4', fixed=dict(**WALL, **RDCON_ONLY), nn=4),
    # Varying axis_mid_pt_skew:
    ScanSpec('STRIDE_axis_mid_pt_skew_scan', 'axis_mid_pt_skew', [11.0, 12.0, 13.0],
             [2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 18.0, 20.0], 'STRIDE_axis_mid_pt_skew_scan',
             'STRIDE axis_mid_pt_skew scan results:', fixed=dict(**WALL, **NO_PEST, run_rdcon=False)),
    # large_sol_extent_pest
    ScanSpec('large_sol_extent_pest_scan', 'large_sol_extent_pest', [0.8, 0.9],
             [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], 'large_sol_extent_pest_scan',
             'PEST3 large_sol_extent_pest scan results:',
             fixed=dict(**WALL, run_pest3=True, run_stride=False, run_rdcon=True, gal_flag='f', ode_flag='f')),
    # Varying nq:
    ScanSpec('RDCON_nq_scan', 'nq', [5, 6], list(range(1, 8)), 'RDCON_nq_scan', 'RDCON nq scan results:',
             fixed=dict(**WALL, **RDCON_ONLY)),
    # sing1_flag
    ScanSpec('sing1_flag_scan', 'sing1_flag', ['t', 'f'], ['t', 'f'], 'sing1_flag_scan', 'sing1_flag scan results:',
             fixed=dict(**WALL, **NO_PEST, run_stride=True, run_rdcon=True)),
    # regrid_flag - probably broken
    ScanSpec('regrid_flag_scan', 'regrid_flag', ['t', 'f'], ['t', 'f'], 'regrid_flag_scan', 'regrid_flag scan results:',
             fixed=dict(**WALL, **NO_PEST, run_stride=True, run_rdcon=True)),
    # PHYSICS PARAMETER DEPENDENCIES
    # varying q-surface truncation. delta_prime_variability may break.
    ScanSpec('edge_truncation_q_scan', 'qhigh', [6.2, 7.2], [2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
             'edge_truncation_q_scan', 'Psihigh truncation scan results:', fixed=dict(psihigh=0.9999, ode_flag='f', **WALL, sas_flag='f')),
    ScanSpec('edge_truncation_q_scan_no_wall', 'qhigh', [6.2, 7.2], [2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
             'edge_truncation_q_scan', 'Psihigh truncation scan results (no wall):',
             fixed=dict(psihigh=0.9999, ode_flag='f', **NO_WALL, sas_flag='f'), prefix='no_wall_'),
    # varying wall radius:
    ScanSpec('wall_radius_scan', 'a_wall', [0, 0.1, 21],
             [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21], 'wall_scan',
             'Wall radius scan results:', superquick_vals=[0]),
]
SCANS = {s.name: s for s in _SPECS}

# RDCON/STRIDE inputs that AREN'T scanned in this file: (& reasons why):
#   pfac - varying nx in rdcon will determine whether we need more finite elements near the rational surfaces
#   sing_start - cutting out the effect of various rational surfaces will affect the result, especially
#                if q_0 < 1, and the plasma is ideal unstable - delta' at other surfaces doesn't mean anything
#   crossover - effect will allow speedup by varying tol_r, tol_nr
#   ucrit - doesn't effect resistive calculations in stride or rdcon
#   nIntervalsTot - is autmatically increased by stride to minimally cover the number of singular intervals
#                   seems to affect threading/speed


def _banner(text):
    print(BANNER, BANNER, text, BANNER, BANNER, sep='\n')


def run_scan(name, eq_filename, results_dir=None, quick_test=True, verbose=True, output_prefix='', **kwargs):
    """Run the 1D scan SCANS[name] on eq_filename; returns (result, message) from scan_1D_input.

    kwargs: scan_vals overrides the default values; superquick/debug where the scan supports them;
    anything else is forwarded to scan_1D_input.
    """
    spec = SCANS[name]
    if results_dir is None:
        results_dir = os.path.join(tps_home(), 'tests/test_results')
    superquick = kwargs.pop('superquick', False) if spec.superquick_vals is not None else False
    debug = kwargs.pop('debug', False) if spec.debug_fixed is not None else False
    scan_vals = kwargs.pop('scan_vals', None)
    if scan_vals is None:
        if superquick:
            scan_vals = spec.superquick_vals
        elif quick_test:
            scan_vals = spec.quick_vals
        else:
            scan_vals = spec.full_vals
            if spec.full_note:
                print(spec.full_note)
    input_name = spec.input_name(scan_vals) if callable(spec.input_name) else spec.input_name
    call_kwargs = {**spec.defaults, **{k: kwargs.pop(k) for k in spec.defaults if k in kwargs},
                   **spec.fixed, **(spec.debug_fixed if debug else {})}
    if output_prefix + spec.prefix:
        call_kwargs['output_prefix'] = output_prefix + spec.prefix
    result, message = scan_1D_input(input_name, scan_vals, eq_filename, spec.nn,
                                    output_location=os.path.join(results_dir, '1D_scans', spec.subdir),
                                    **call_kwargs, **kwargs)  # a repeated fixed kwarg raises TypeError
    if verbose:
        _banner(f'{spec.title}\n{message}')
    return result, message


# Things that might vary for certain equilibria
key_numerical_tests_n4 = (
    'mtheta_scan',
    'mpsi_scan',
    'edge_truncation_within_surface_scan_no_wall',
    'pest3_finite_element_scan',
    'mag_coord_type_scan',
    'grid_type_scan',
    'vacuum_mtheta_scan',
    'RDCON_cutoff_scan_n4',
)

key_numerical_tests = key_numerical_tests_n4[:-1]

# Will have a big effect (physics motivated)
physics_tests = (
    'edge_truncation_q_scan',
    'edge_truncation_q_scan_no_wall',
    'wall_radius_scan',
)

# Things I expect to barely change things
extra_numerical_tests = (
    'fourier_mode_scan',
    'sing_order_scan',
    'matching_point_scan',
    'psilow_truncation_scan',
    'equilibrium_integrator_tolerance_scan',
    'mtheta_scan_no_wall',
    'mpsi_scan_no_wall',
    'edge_truncation_within_surface_scan',
    'rdcon_finite_element_scan',
    'integrator_tolerance_scan',
    'RDCON_cutoff_scan_n1',
    'RDCON_solver_scan',
    'STRIDE_axis_mid_pt_skew_scan',
    'large_sol_extent_pest_scan',
    'RDCON_nq_scan',
    'sing1_flag_scan',
    'regrid_flag_scan',
)

# The name of every scan, in default run order:
scan_functions = tuple(SCANS)
