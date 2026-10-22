"""fsqueue (drivers/fsqueue.py, a copy of eq_stab common/fsqueue.py): worker names, claims, deadline."""
import json
import multiprocessing
import os
import signal
import time
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
    monkeypatch.setenv('SLURM_RESTART_COUNT', '1')  # requeued after preemption: new names
    assert list(fq.launch(str(tmp_path), 1, None)) == ['nid1.7.r1.0.w0']

FAST = dict(STALE_S=2.0, HEARTBEAT_S=0.3, RESCAN_S=0.5)


def _units(n, prefix='u'):
    return [{'id': f'{prefix}{i:03d}', 'est': i % 7} for i in range(n)]


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


def test_claims_follow_manifest_order(tmp_path):
    """Workers claim from the head of the order (eq_stab 83ce571)."""
    q = fq.Queue(tmp_path, **FAST)
    units = [dict(u, est=1) for u in _units(20)]
    q.add_units(units)
    seen = []
    for w in ('a', 'b', 'c'):
        fq.work(q, w, lambda uid, u: seen.append(uid), idle_exit_s=0, poll_s=0.01, log=lambda *a: None)
        q.add_units(_units(3, prefix=f'x{w}'))
    assert seen[:20] == [u['id'] for u in units]


def test_worker_claims_only_what_it_accepts(tmp_path):
    """work(accept=...): a worker passes over the units it does not accept and leaves them to others."""
    q = fq.Queue(tmp_path, **FAST)
    q.add_units([dict(id=f'u{i}', est=1, need=need) for i, need in enumerate((5, 30, 10))])
    seen = []
    fq.work(q, 'small', lambda uid, u: seen.append(uid), idle_exit_s=0, poll_s=0.01, accept=lambda u: u['need'] < 16,
            log=lambda *a: None)
    assert seen == ['u0', 'u2'] and q.done_ids() == {'u0', 'u2'}


def test_live_deadline_is_reread(tmp_path, monkeypatch):
    """A --time-min job's limit is raised after it starts: workers follow squeue's end, and a failed query keeps the
    last good end instead of SLURM_JOB_END_TIME's first, short one (eq_stab 179bf09, 4741ba4)."""
    monkeypatch.setenv('SLURM_JOB_ID', f'test{os.getpid()}')
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(time.time() - 1))
    ends = [time.time() + 3600, None]
    monkeypatch.setattr(fq, '_job_end_squeue', lambda: ends.pop(0) if ends else None)
    d = fq.JobDeadline(0.0, every_s=0.0)
    try:
        assert d() > time.time() + 3000 and d() > time.time() + 3000
        q = fq.Queue(tmp_path, **FAST)
        q.add_units(_units(1))
        seen = []
        assert fq.work(q, 'w', lambda uid, u: seen.append(uid), deadline=d, idle_exit_s=0, poll_s=0.01,
                       log=lambda *a: None) == 'drained' and seen == ['u000']
    finally:
        if os.path.exists(d._cache()):
            os.remove(d._cache())


def test_held_claims_are_not_retried(tmp_path, monkeypatch):
    """A unit another worker holds is skipped from one claims listing: no mkdir (hot directory on DVS) and no
    attempt/failure listings for it; a stale one is still stolen (eq_stab a54bdae)."""
    q = fq.Queue(tmp_path, **FAST)
    q.add_units([dict(u, est=1) for u in _units(6)])
    held = [q.next_unit(fq.Cursor(start=0), 'a') for _ in range(4)]
    tried = []
    real = q.acquire
    monkeypatch.setattr(q, 'acquire', lambda kind, name, owner: tried.append(name) or real(kind, name, owner))
    got = q.next_unit(fq.Cursor(start=0), 'b')
    assert got == 'u004' and tried == ['u004'] and got not in held
    q.expire('claims', held[0])
    assert q.next_unit(fq.Cursor(start=0), 'c') == held[0]


def test_idle_worker_waits_to_steal_a_dead_claim(tmp_path):
    """A worker started beside a dead job's claim waits until it may steal it (STALE_S from first sight) instead of
    quitting after a shorter idle_exit_s (tps3 10-21: --idle-exit 120 < STALE_S 300; eq_stab 3df78d4)."""
    q = fq.Queue(tmp_path, **FAST)
    q.add_units(_units(1))
    assert q.next_unit(fq.Cursor(), 'dead') == 'u000'           # claimed, never beaten again
    seen = []
    assert fq.work(q, 'w', lambda uid, u: seen.append(uid), idle_exit_s=0.5, poll_s=0.1,
                   log=lambda *a: None) == 'idle' and seen == ['u000']
