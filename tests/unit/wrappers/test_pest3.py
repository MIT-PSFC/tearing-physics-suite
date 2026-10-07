# Unit tests for wrappers.pest3 that need no PEST3 binary (pytest).
import os
import stat

import numpy as np
import pytest
import xarray as xr

import tearing_physics_suite.wrappers.pest3 as p3w
from tearing_physics_suite.wrappers.pest3 import pest3_rescale_deltaprimes

FIXTURES = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fixtures')
PEST3_RAW = os.path.join(FIXTURES, 'pest3_raw')


def _fake_pest3(tmp_path, body='exit 0'):
    """Directory holding a stand-in pest3x shell script, plus a dummy equilibrium file."""
    exe_dir = tmp_path / 'bin'
    exe_dir.mkdir()
    exe = exe_dir / 'pest3x'
    exe.write_text('#!/bin/sh\n' + body + '\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    eq = tmp_path / 'g000001.00001'
    eq.write_text('dummy')
    return str(exe_dir), str(eq)


def test_stale_pest3_nc_not_read(tmp_path):
    """A pest3.nc left by an earlier run must not be read when the new run writes none."""
    exe_dir, eq = _fake_pest3(tmp_path)
    work = tmp_path / 'work'
    work.mkdir()
    xr.Dataset({'qa': ('x', [1.0, 2.0])}).to_netcdf(work / 'pest3.nc')

    cwd = os.getcwd()
    try:
        pest3_xr, pest3_ran, _ = p3w.PEST3_resistive_calculation(
            eq, 1, working_dir=str(work), pest3_dir=exe_dir, verbose=False,
            save_input=False, clean_netcdf=False, vacuum_source_pest='pest3')
    finally:
        os.chdir(cwd)
    assert pest3_xr is None
    assert not pest3_ran
    assert not (work / 'pest3.nc').exists()


def _run_loop(monkeypatch, qa_last, ran=True, truncimax=4):
    """Run pest3_special_truncation_loop with PEST3 calls replaced by a fixed qa result."""
    monkeypatch.setattr(p3w, 'pest3_special_truncation_single', lambda *a, **k: (0.9, True))

    def fake_run(*args, **kwargs):
        if not ran:
            return None, False, {}
        return xr.Dataset({'qa': ('x', [1.0, qa_last])}), True, {}
    monkeypatch.setattr(p3w, 'PEST3_resistive_calculation', fake_run)
    return p3w.pest3_special_truncation_loop('eq', 1, 5.0, {}, verbose=False, truncimax=truncimax)


def test_truncation_loop_not_converged(monkeypatch):
    """No convergence returns the last lower bracket and False (was a NameError)."""
    psihigh, ok = _run_loop(monkeypatch, qa_last=3.0, truncimax=4)
    assert not ok
    # qa stays below qlim, so the lower bracket rises from 0.9 by bisection toward 1.
    assert psihigh == pytest.approx(1.0 - 0.1 / 2**4)


def test_truncation_loop_converged(monkeypatch):
    psihigh, ok = _run_loop(monkeypatch, qa_last=5.005)
    assert ok
    assert psihigh == pytest.approx(0.9)


def test_truncation_loop_failed_run(monkeypatch):
    """A failed PEST3 run returns (0, False) instead of indexing None."""
    assert _run_loop(monkeypatch, qa_last=3.0, ran=False) == (0, False)


def _run_gpec_vac(tmp_path, body, vac_in=True):
    exe_dir, eq = _fake_pest3(tmp_path, body)
    work = tmp_path / 'work'
    work.mkdir()
    if vac_in:
        (work / 'vac.in').write_text('&vac\n/\n')
    cwd = os.getcwd()
    try:
        return p3w.PEST3_resistive_calculation(
            eq, 1, working_dir=str(work), pest3_dir=exe_dir, verbose=False,
            save_input=False, clean_netcdf=False, mthvac_pest=480), work
    finally:
        os.chdir(cwd)


def test_gpec_vacuum_needs_gpec_build(tmp_path):
    """The default GPEC vacuum refuses a pest3x linked without it."""
    with pytest.raises(RuntimeError, match='without GPEC vacuum'):
        _run_gpec_vac(tmp_path, 'exit 0')


def test_gpec_vacuum_needs_vac_in(tmp_path):
    with pytest.raises(FileNotFoundError, match='vac.in'):
        _run_gpec_vac(tmp_path, '# vacuum_mod_MOD_mscvac\nexit 0', vac_in=False)


def test_gpec_vacuum_passes_V_flag(tmp_path):
    """-V<mthvac_pest> reaches pest3x, and the input dict records the source."""
    (_, _, inputs), work = _run_gpec_vac(tmp_path, '# vacuum_mod_MOD_mscvac\necho "$@" > args.txt')
    assert ' -V480 ' in ' ' + (work / 'args.txt').read_text() + ' '
    assert inputs['vacuum_source_pest'] == 'gpec'
    assert inputs['mthvac_pest'] == 480


# pest3_clean_netcdf before the rename/pad rewrite (refactor_changes.md), verbatim.
def _old_pest3_clean_netcdf(ps3, debug=True, drop_soln_info=True, q_rationals=None, r=None, r_prime=None):
    """Standardize PEST3 NetCDF output dimensions, variables, and unit conventions.

    Renames dimensions, rescales Delta' values to GPEC units via
    pest3_rescale_deltaprimes, and optionally aligns surface labels with
    GPEC rational surfaces.

    Parameters
    ----------
    ps3 : xr.Dataset
        Raw PEST3 output dataset.
    debug : bool
        Print debug information.
    drop_soln_info : bool
        Drop full-solution variables (x1frbo_re, etc.) to avoid dimension conflicts.
    q_rationals : np.ndarray or None
        GPEC rational surface q values for alignment.
    r, r_prime : np.ndarray or None
        GPEC surface label coordinates for alignment.

    Returns
    -------
    xr.Dataset
        Processed dataset with standardized dimensions and rescaled Delta' values.
    """

    #########################################################################################################
    # Standardize dimensions and variable names in PEST3 output:
    #########################################################################################################
    # Vacuum matrix is for checks only; keep its source as attributes.
    if 'vacuum_source' in ps3:
        ps3.attrs['vacuum_source'] = 'gpec' if int(ps3.vacuum_source) == 1 else 'pest3'
        ps3.attrs['mthvac'] = int(ps3.mthvac)
    ps3 = ps3.drop_vars([v for v in ('vacmat', 'vacmti', 'vacuum_source', 'mthvac') if v in ps3])

    #Names of dims:
    missing_m_from_gpec = []
    missing_m_from_pest = []
    if len(ps3.cmatch.dims) > 0:
        surfdim=ps3.cmatch.dims[0]
        unknowndim=ps3.x1frbo_re.dims[1]
        if q_rationals is not None:
            # We convert q_rationals to m values:
            m_gpec = np.round(np.squeeze(q_rationals*ps3.n.values))
            m_pest = np.round(np.squeeze(ps3.qslay.values*ps3.n.values))

            #Check no values in m_pest are missing from m_gpec:
            missing_m_from_gpec = [m for m in m_pest if m not in m_gpec]
            missing_m_from_pest = [m for m in m_gpec if m not in m_pest]

            assert len(missing_m_from_gpec) == 0, "Some m values from GPEC are not in PEST3 output. Check truncation logic."

            if len(missing_m_from_pest) > 0:
                assert min(missing_m_from_pest) > max(m_pest), "Some interior m values from PEST3 are not in GPEC output. Check truncation logic."
    else: #No rational surfaces calculated
        surfdim=None
        unknowndim=ps3.x1frbo_re.dims[0]
    profdim_1=ps3.psinod.dims[0]
    profdim_2=ps3.qa.dims[0]
    thetadim=ps3.xjacob.dims[1]
    #Rename dimensions to standardize & remove duplicates
    for varname, da in ps3.data_vars.items():
        new_dims = []
        for dim_i in da.dims:
            if (dim_i == surfdim) and ('r_temp' in new_dims):
                new_dims.append('r_prime_temp')
            elif dim_i == surfdim:
                new_dims.append('r_temp')
            elif dim_i == profdim_1:
                new_dims.append('psinod_dim')
            elif dim_i == profdim_2:
                new_dims.append('qprof_dim')
            elif dim_i == thetadim:
                new_dims.append('theta_dim')
            elif dim_i == unknowndim:
                new_dims.append('ukn_dim')
            else:
                new_dims.append(dim_i)
        if debug: print(len(da.dims),new_dims)
        if len(da.dims) > 0:
            tempvals = da.values
            temp_da = xr.DataArray(tempvals, dims=tuple(new_dims))
            ps3[varname] = temp_da

    for varname, da in ps3.data_vars.items():
        if drop_soln_info:
            # Drop variables with more than 3 dimensions (e.g. x1frbo_re, x1frbo_im):
            if len(da.dims) > 2:
                if debug: print(f"Dropping variable {varname} with {len(da.dims)} dimensions.")
                ps3 = ps3.drop_vars(varname)
            elif len(da.dims) == 2 and 'ukn_dim' in da.dims:
                if debug: print(f"Dropping variable {varname} with {len(da.dims)} dimensions.")
                ps3 = ps3.drop_vars(varname)
            elif len(da.dims) == 2 and not ('r_temp' in da.dims and 'r_prime_temp' in da.dims):
                if debug: print(f"Dropping variable {varname} with {len(da.dims)} dimensions.")
                ps3 = ps3.drop_vars(varname)

    assert 'r' not in ps3.data_vars, "PEST3 output already has a variable named 'r'. Check PEST3 output and cleaning logic."
    assert 'r_prime' not in ps3.data_vars, "PEST3 output already has a variable named 'r_prime'. Check PEST3 output and cleaning logic."
    if 'r_temp' not in ps3.dims:
        print("PEST3 data variables: ", ps3.dims)
        raise ValueError("PEST3 output does not have a variable named 'r_temp'. Check PEST3 output and cleaning logic.")

    for varname, da in ps3.data_vars.items():
        if len(missing_m_from_pest) > 0:
            # We expand ps3[varname] such that ps3[varname].r matches input DataArray r:
            if 'r_temp' in da.dims and 'r_prime_temp' not in da.dims:
                tempvals = da.values
                # Add extra nans to the end:
                tempvals_new = np.full((len(r)), np.nan)
                tempvals_new[:len(da.r_temp)] = tempvals
                temp_da = xr.DataArray(tempvals_new, dims=('r'))
                # Set the coordinates of temp_da to match r:
                temp_da.coords['r'] = r
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da
            elif 'r_temp' in da.dims and 'r_prime_temp' in da.dims:
                # We expand ps3[varname] such that ps3[varname].r matches input DataArray r and ps3[varname].r_prime matches input DataArray r_prime:
                tempvals = da.values
                # Add extra nans to the end:
                tempvals_new = np.full((len(r), len(r_prime)), np.nan)
                tempvals_new[:len(da.r_temp), :len(da.r_prime_temp)] = tempvals
                temp_da = xr.DataArray(tempvals_new, dims=('r', 'r_prime'))
                # Set the coordinates of temp_da to match r and r_prime:
                temp_da.coords['r'] = r
                temp_da.coords['r_prime'] = r_prime
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da
        else:
            # Just rename 'r_temp' to 'r' and 'r_prime_temp' to 'r_prime':
            if 'r_temp' in da.dims and 'r_prime_temp' not in da.dims:
                tempvals = da.values
                temp_da = xr.DataArray(tempvals, dims=('r'))
                # Set the coordinates of temp_da to match r:
                temp_da.coords['r'] = r
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da
            elif 'r_temp' in da.dims and 'r_prime_temp' in da.dims:
                tempvals = da.values
                temp_da = xr.DataArray(tempvals, dims=('r', 'r_prime'))
                # Set the coordinates of temp_da to match r and r_prime:
                temp_da.coords['r'] = r
                temp_da.coords['r_prime'] = r_prime
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da

    if len(ps3.cmatch.dims) > 0:
        return pest3_rescale_deltaprimes(ps3)
    return ps3


def _surfaces(rdcon_file, n_surf=None):
    """q_rationals, r, r_prime as run_resistive_calculation takes them from RDCON (optionally the first n_surf)."""
    rd = xr.open_dataset(os.path.join(FIXTURES, rdcon_file) if os.sep not in rdcon_file else rdcon_file)
    if n_surf is not None:
        rd = rd.isel(r=slice(n_surf), r_prime=slice(n_surf))
    return dict(q_rationals=rd.q_rational.values, r=rd.r, r_prime=rd.r_prime)


def _check_same(pest3_file, drop_soln_info=True, **surf):
    """New and old cleaning give identical datasets (or the same exception) on a raw pest3.nc."""
    def run(f):
        try:
            return f(xr.open_dataset(pest3_file), debug=False, drop_soln_info=drop_soln_info, **surf)
        except Exception as e:
            return e
    old, new = run(_old_pest3_clean_netcdf), run(p3w.pest3_clean_netcdf)
    if isinstance(old, Exception):
        assert type(new) is type(old)
        return None
    xr.testing.assert_identical(new, old)
    if surf.get('r') is not None:
        # Saved file (scipy, as PEST3_resistive_calculation saves it) reads back the same.
        xr.testing.assert_identical(xr.open_dataset(new.to_netcdf()), xr.open_dataset(old.to_netcdf()))
    return new


# 3 surfaces (m=2-4), GPEC vacuum fields; 3 MB, kept outside the repo (TPS_PEST3_RAW, else the Group 1 work dir)
P985 = os.path.join(os.environ.get('TPS_PEST3_RAW', '/fusion/projects/tmdb/src/tps_G1_work/test_data'),
                    'p985_nowall_gpec_pest3.nc')
needs_p985 = pytest.mark.skipif(not os.path.exists(P985), reason=f'no raw PEST3 file {P985}')
P985_RDCON = os.path.join(PEST3_RAW, 'p985_nowall_gpec_rdcon_n1.nc')  # same case, m=2-4
FULL_RDCON = 'g147131.02300_DIIID_KEFIT_rdcon_n1.nc'  # same equilibrium, untruncated: m=2-6


@needs_p985
@pytest.mark.parametrize('drop_soln_info', [True, False])
@pytest.mark.parametrize('case', ['own_rdcon', 'missing_m_from_pest', 'missing_m_from_gpec', 'no_rdcon'])
def test_clean_netcdf_identical_to_old(case, drop_soln_info):
    surf = {'own_rdcon': lambda: _surfaces(P985_RDCON),
            'missing_m_from_pest': lambda: _surfaces(FULL_RDCON),
            'missing_m_from_gpec': lambda: _surfaces(P985_RDCON, n_surf=2),
            'no_rdcon': lambda: {}}[case]()
    new = _check_same(P985, drop_soln_info, **surf)
    if case == 'missing_m_from_pest' and drop_soln_info:
        assert new.sizes['r'] == 5 and np.isnan(new.Delta_prime.isel(i=0, r=-1)).all()


@pytest.mark.parametrize('drop_soln_info', [True, False])
def test_clean_netcdf_no_surfaces_raises_as_old(drop_soln_info):
    assert _check_same(os.path.join(PEST3_RAW, 'no_surfaces_pest3.nc'), drop_soln_info) is None


@needs_p985
def test_clean_netcdf_does_not_mutate_input():
    ps3 = xr.open_dataset(P985)
    p3w.pest3_clean_netcdf(ps3, debug=False, **_surfaces(P985_RDCON))
    assert ps3.attrs == {} and 'vacmat' in ps3
