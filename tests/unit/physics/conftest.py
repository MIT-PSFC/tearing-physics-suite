"""Per-code datasets with MRE surface terms, built from the stored n=1 outputs (no Fortran)."""
import pytest

from tearing_physics_suite.physics.combine import add_code_dim, code_delta_primes
from tearing_physics_suite.physics.cross_field_transport import (
    chi_para_lmfp_no_w_on_modes,
    chi_para_lmfp_noisland_on_modes,
    chi_para_smfp_on_modes,
    chi_perp_on_modes,
)
from tearing_physics_suite.physics.surface_terms import deltaprime_crit_on_modes, mre_terms_on_modes


@pytest.fixture(scope='session')
def code_datasets(code_fixture, kin_file):
    """{code: dataset with a 'code' dim and Delta_prime_surf}; rdcon also has the MRE surface terms."""
    from tearing_physics_suite.drivers.profile_read import read_kin_file
    p = read_kin_file(kin_file)
    rd = mre_terms_on_modes(code_fixture('rdcon'), p['ni_spline'], p['ne_spline'], p['te_keV_spline'],
                            p['ti_keV_spline'], average_ion_mass=2.5)
    for f in (chi_para_lmfp_no_w_on_modes, chi_para_lmfp_noisland_on_modes, chi_para_smfp_on_modes):
        rd = f(rd)
    rd = deltaprime_crit_on_modes(chi_perp_on_modes(rd, energy_confinement_time=0.12))
    raw = {'rdcon': rd, 'stride': code_fixture('stride'), 'pest3': code_fixture('pest3')}
    return {c: code_delta_primes(add_code_dim(ds, c), c) for c, ds in raw.items()}
