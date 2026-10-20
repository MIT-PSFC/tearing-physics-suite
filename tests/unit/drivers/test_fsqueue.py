"""fsqueue worker names (drivers/fsqueue.py, a copy of eq_stab common/fsqueue.py)."""
import json
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

FAST = dict(STALE_S=2.0, HEARTBEAT_S=0.3, RESCAN_S=0.5)


def test_steal_race_one_winner(tmp_path):
    """Two workers saw the same stale claim; one steals it before the other renames: the second must not take it."""
    a, b = fq.Queue(tmp_path, **FAST), fq.Queue(tmp_path, **FAST)
    a.add_units([{'id': 'u000', 'est': 0}])
    a.next_unit(fq.Cursor(), 'old')
    a.expire('claims', 'u000')                             # the owner died: stale for everyone
    real = b.stale
    calls = []

    def stale_then_a_steals(kind, name):
        if not calls:
            calls.append(1)
            assert real(kind, name)
            assert a.acquire('claims', 'u000', 'a')         # a wins in b's window
            return True
        return real(kind, name)
    b.stale = stale_then_a_steals
    assert not b.acquire('claims', 'u000', 'b')
    assert json.load(open(a.path('claims', 'u000', 'owner.json')))['owner'] == 'a'
