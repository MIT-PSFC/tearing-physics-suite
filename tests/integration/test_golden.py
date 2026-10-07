"""Rerun the golden cases (tests/golden/cases.py) and diff against the reference (TPS_GOLDEN_DIR).

Exact match by default; set TPS_GOLDEN_RTOL / TPS_GOLDEN_ATOL to loosen.
"""
import os

import pytest

from golden.cases import run_case
from golden.compare import compare_dirs

RTOL = float(os.environ.get('TPS_GOLDEN_RTOL', 0))
ATOL = float(os.environ.get('TPS_GOLDEN_ATOL', 0))


@pytest.mark.fortran
@pytest.mark.parametrize('case', [
    'nonlinear_wall_rotation', 'nonlinear_nowall', 'linear_wall',
    pytest.param('multi_run_zarr', marks=pytest.mark.parallel)])
def test_golden_case(case, golden_dir, tmp_path):
    run_case(case, str(tmp_path))
    diffs = compare_dirs(str(golden_dir), str(tmp_path), RTOL, ATOL, names=[case])
    assert not diffs, '\n'.join(diffs[:50])
