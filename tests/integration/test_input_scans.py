"""Quick input scans through the runner (two short scans)."""
import pytest

from tearing_physics_suite.drivers.input_test_runner import run_multiple_scans

pytestmark = [pytest.mark.fortran, pytest.mark.slow]


def test_quick_scans(eq_file, tmp_path):
    results, messages, names, failed = run_multiple_scans(
        eq_file, scan_namelist=['edge_truncation_q_scan', 'rdcon_finite_element_scan'],
        results_dir=str(tmp_path), quick_test=True, verbose=True, debug=True)
    assert not failed, failed
    assert len(results) == 2
