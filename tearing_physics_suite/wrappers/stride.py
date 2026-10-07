"""Run STRIDE (Riccati/BVP Delta'). Inputs are written by gpec_inputs.write_stride_in."""
from tearing_physics_suite.wrappers.gpec_common import run_gpec_code, stage_executable


def run_stride(gpec_dir, working_dir, nn, fresh_start=True, save_terminal_output=True, verbose=False):
    """Stage and run stride in working_dir (cwd), returning (stride_xr or None, ran)."""
    stage_executable(gpec_dir, 'stride', working_dir)
    return run_gpec_code('stride', working_dir, nn, fresh_start, save_terminal_output, verbose)
