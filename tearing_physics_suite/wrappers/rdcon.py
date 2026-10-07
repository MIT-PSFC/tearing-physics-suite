"""Run RDCON (resistive DCON, Galerkin Delta'). Inputs are written by gpec_inputs.write_rdcon_in."""
from tearing_physics_suite.wrappers.gpec_common import run_gpec_code, stage_executable


def run_rdcon(gpec_dir, working_dir, nn, fresh_start=True, save_terminal_output=True, verbose=False):
    """Stage and run rdcon in working_dir (cwd), returning (rdcon_xr or None, ran)."""
    stage_executable(gpec_dir, 'rdcon', working_dir)
    return run_gpec_code('rdcon', working_dir, nn, fresh_start, save_terminal_output, verbose)
