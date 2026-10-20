"""fsqueue worker names (drivers/fsqueue.py, a copy of eq_stab common/fsqueue.py)."""
import multiprocessing
import signal
from types import SimpleNamespace

from tearing_physics_suite.drivers import fsqueue as fq


class _Proc:
    """Stands in for a worker process: never started, exits 0."""
    def __init__(self, target, args, name):
        self.exitcode = 0

    def start(self):
        pass

    def join(self):
        pass

    def is_alive(self):
        return False


def test_launch_tag_has_slurm_step(tmp_path, monkeypatch):
    """Sessions run as concurrent job steps on one node get distinct workers: <host>.<job>.s<step>.<procid>.w<i>."""
    monkeypatch.setattr(multiprocessing, 'get_context', lambda _: SimpleNamespace(Process=_Proc))
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    monkeypatch.setattr(fq.socket, 'gethostname', lambda: 'nid1')
    for k, v in dict(SLURM_JOB_ID='7', SLURM_STEP_ID='3', SLURM_PROCID='0').items():
        monkeypatch.setenv(k, v)
    assert list(fq.launch(str(tmp_path), 2, None)) == ['nid1.7.s3.0.w0', 'nid1.7.s3.0.w1']
    monkeypatch.setenv('SLURM_STEP_ID', 'batch')  # not a number: no step in the tag
    assert list(fq.launch(str(tmp_path), 1, None)) == ['nid1.7.0.w0']
