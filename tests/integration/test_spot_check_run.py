"""multi_run_ with spot-checks on real runs: 2 cases, both checked with one short scan (mpsi_scan), n=1."""
import os
import shutil
import tempfile

import numpy as np
import pytest
import xarray as xr
from conftest import MRE, REDUCED

from tearing_physics_suite.drivers.multi_run import multi_compile_zarr, multi_run_
from tearing_physics_suite.drivers.profile_read import read_kin_file
from tearing_physics_suite.drivers.spot_check import SpotCheck

pytestmark = [pytest.mark.fortran, pytest.mark.parallel]


def _run(eq_file, out, spot):
    # Worker dirs on node-local disk (see tests/golden/cases.py); results copied to out.
    local = tempfile.mkdtemp(prefix='tps_spot_', dir=os.environ.get('TMPDIR', '/tmp'))
    profs = [read_kin_file(eq_file + '.kin') for _ in range(2)]
    _, _, errors = multi_run_([eq_file] * 2, profs, local, nvec=[1], run_pest3=False, vac_flag='f', ode_flag='f',
                              psi_surfs_of_interest=[0.95], spot_check=spot, **MRE, **REDUCED)
    ds, _ = multi_compile_zarr([eq_file] * 2, local)
    ds = ds.load()
    shutil.copytree(local, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns('worker_*', '.nfs*'))
    shutil.rmtree(local, ignore_errors=True)
    return ds, errors


def test_spot_check_run(eq_file, tmp_path):
    spot_ds, errors = _run(eq_file, tmp_path / 'spot', SpotCheck(fraction=1.0, scans=('mpsi_scan',)))
    plain_ds, _ = _run(eq_file, tmp_path / 'plain', None)
    assert not errors
    spots = xr.open_zarr(tmp_path / 'spot' / 'spot_checks' / 'spot_checks.zarr').load()
    assert spots.sizes['spot'] == 2 and set(spots.scan.values) == {'mpsi_scan'}
    assert spots.sizes['scan_value'] == 3 and set(spots.code.values) == {'rdcon', 'stride'}
    assert np.isin(spots.rel_exceeded_psi95.values, [0, 1]).all()
    assert (tmp_path / 'spot' / 'spot_checks' / 'report.txt').exists()
    flag = spot_ds.spot_check_flag.sel(code=['rdcon', 'stride'])
    assert np.isin(flag.values, [0, 1]).all()
    xr.testing.assert_identical(spot_ds.drop_vars('spot_check_flag'), plain_ds)
