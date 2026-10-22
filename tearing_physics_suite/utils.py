# Utilities for tearing physics suite

import contextlib
import math
import os
import signal
import subprocess
import time

import numpy as np


def trim_nans(delta_primes,delta_prime_errs=None):
    """Remove trailing NaN rows/columns from a square Delta' matrix.

    Assumes NaNs only appear in the final rows and columns, and that the
    underlying non-NaN sub-matrix is also square.

    Parameters
    ----------
    delta_primes : np.ndarray
        Square Delta' matrix, possibly with NaN padding.
    delta_prime_errs : np.ndarray or None
        Matching error matrix; trimmed identically if provided.

    Returns
    -------
    delta_primes : np.ndarray
        Trimmed square matrix.
    nans_in_col1 : int
        Number of NaN rows/columns removed.
    delta_prime_errs : np.ndarray or None
        Trimmed error matrix (or None).
    """
    assert delta_primes.shape[0] == delta_primes.shape[1], "Input matrix is not square."
    if delta_prime_errs is not None:
        assert delta_primes.shape[0] == delta_prime_errs.shape[0] == delta_prime_errs.shape[1], "Delta prime errors have different shape to Delta primes."
    dp_col1 = delta_primes[:,0]
    nans_in_col1=0
    for i in dp_col1:
        if np.isnan(i):
            nans_in_col1+=1
    dp_row1 = delta_primes[0,:]
    nans_in_row1=0
    for i in dp_row1:
        if np.isnan(i):
            nans_in_row1+=1
    assert nans_in_col1 == nans_in_row1, "Input matrix has inconsistent number of nans in rows and columns."
    #We cut the matrix to remove nans:
    if nans_in_col1 > 0:
        delta_primes = delta_primes[:-nans_in_col1,:-nans_in_row1]
        if delta_prime_errs is not None:
            delta_prime_errs = delta_prime_errs[:-nans_in_col1,:-nans_in_row1]
    # Now delta_primes is a square matrix with no nans. Check no nans present:
    if np.isnan(delta_primes).any():
        print(delta_primes)
        raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")
    if delta_prime_errs is not None:
        if np.isnan(delta_prime_errs).any():
            print(delta_prime_errs)
            raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")
    return delta_primes, nans_in_col1, delta_prime_errs

def create_dense_log_paramvals(start=1e-1, end=1e-7, points_per_decade=10, extra_density_regions=None):
    """Create densely packed parameter values in logarithmic scale.

    Parameters
    ----------
    start : float
        Starting value.
    end : float
        Ending value.
    points_per_decade : int
        Base number of points per decade.
    extra_density_regions : list of tuple or None
        (min_val, max_val, extra_points) for regions needing more density.

    Returns
    -------
    list of float
        Parameter values in descending order.
    """

    # Calculate the log range
    log_start = math.log10(start)
    log_end = math.log10(end)

    # Base logarithmic spacing
    num_decades = abs(log_start - log_end)
    base_points = int(num_decades * points_per_decade) + 1

    # Create base logarithmic array using list comprehension
    paramvals_base = []
    for i in range(base_points):
        log_val = log_start + (log_end - log_start) * i / (base_points - 1)
        paramvals_base.append(10**log_val)

    # Add extra density in specific regions if specified
    extra_vals = []
    if extra_density_regions:
        for min_val, max_val, extra_points in extra_density_regions:
            log_min = math.log10(min_val)
            log_max = math.log10(max_val)
            for i in range(extra_points):
                log_val = log_min + (log_max - log_min) * i / (extra_points - 1)
                extra_vals.append(10**log_val)

    # Combine all values and remove duplicates
    all_vals = paramvals_base + extra_vals
    paramvals_unique = list(set(all_vals))

    # Sort in descending order (from 1e-1 to 1e-7)
    paramvals_sorted = sorted(paramvals_unique, reverse=True)

    return paramvals_sorted


_EQ_SUFFIXES = ('.geqdsk', '.eqdsk', '.gfile', '.ifile')


def eq_stem(eq_filename):
    """Basename of eq_filename with a known equilibrium suffix removed (other dots kept)."""
    name = os.path.basename(eq_filename)
    for suffix in _EQ_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _get_num_cpus():
    """Number of worker processes: $TPS_NUM_WORKERS, else this node's SLURM allocation, else the affinity mask."""
    for var in ['TPS_NUM_WORKERS', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE']:
        val = os.environ.get(var)
        if val:
            return int(val.split('(')[0].split(',')[0])  # SLURM_JOB_CPUS_PER_NODE: '16(x2)' or '8,4'
    return len(os.sched_getaffinity(0))


def tps_threads():
    """STRIDE and Julia GPEC threads per TPS worker: $TPS_THREADS (job shape; no effect on results), else 1."""
    return int(os.environ.get('TPS_THREADS') or 1)


def session_mem_gib():
    """The session's memory in GiB (Slurm's --mem=<n>G): $TPS_MEM_GIB, else None (no limit)."""
    return float(os.environ.get('TPS_MEM_GIB') or 0) or None


def fits_memory(need_gib, mem_gib=None):
    """True if a run predicted to peak at need_gib fits in mem_gib (default: the session's; None: no limit), with 10 %
    left for the TPS worker and model error."""
    mem_gib = mem_gib or session_mem_gib()
    return mem_gib is None or need_gib <= 0.9 * mem_gib


def work_root():
    """Root for per-worker scratch dirs: $TPS_WORK_ROOT, else $TMPDIR, else /tmp (node-local where possible)."""
    return os.environ.get('TPS_WORK_ROOT') or os.environ.get('TMPDIR') or '/tmp'


CODE_WALL_S = {}  # '<code>_n<n>' (and 'jGPEC_start'): wall seconds of the code runs in this process (parallel: per draw)
CODE_KILLED = {}  # code: [n] of its runs killed by a signal (e.g. out of memory) in this process (parallel: per draw)
CODE_SKIPPED = {}  # code: [n] of its runs not started: predicted not to fit the session's memory (also in CODE_KILLED)


@contextlib.contextmanager
def code_timer(key):
    """Add the block's wall time to CODE_WALL_S[key], also when it raises."""
    t = time.perf_counter()
    try:
        yield
    finally:
        CODE_WALL_S[key] = CODE_WALL_S.get(key, 0.0) + time.perf_counter() - t


@contextlib.contextmanager
def node_slot(name, n, poll=5.0):
    """Hold one of ``n`` per-node slots called ``name`` (flock on files under work_root()); falsy n: no limit.

    E.g. at most n concurrent jGPEC solves per node ($TPS_JGPEC_SLOTS), whose memory sets workers per node.
    """
    if not n:
        yield
        return
    import fcntl
    d = os.path.join(work_root(), f'.slots_{name}')
    os.makedirs(d, exist_ok=True)
    while True:
        for i in range(int(n)):
            fh = open(os.path.join(d, str(i)), 'w')
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            try:
                yield
            finally:
                fh.close()
            return
        time.sleep(poll)


def _die_with_parent():
    """preexec_fn: SIGKILL the child when its parent dies (Linux), so a killed run leaves no orphan code."""
    import ctypes
    try:
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except OSError:
        pass


def run_code(cmd, shell=True, env=None, stack_bytes=None):
    """subprocess.call in its own process group; the group is killed after $TPS_CODE_TIMEOUT s (unset: no limit).

    stack_bytes raises the soft stack limit of the child (up to its hard limit). Raises TimeoutError on a timeout.
    """
    timeout = float(os.environ.get('TPS_CODE_TIMEOUT') or 0) or None

    def preexec():
        _die_with_parent()
        if stack_bytes:
            import resource
            hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
            resource.setrlimit(resource.RLIMIT_STACK,
                               (stack_bytes if hard == resource.RLIM_INFINITY else min(stack_bytes, hard), hard))
    p = subprocess.Popen(cmd, shell=shell, start_new_session=True, preexec_fn=preexec, env=env)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        raise TimeoutError(f'{cmd!r} exceeded TPS_CODE_TIMEOUT={timeout:.0f} s') from None
    except BaseException:
        os.killpg(p.pid, signal.SIGKILL)
        raise


def tps_home():
    """TPS repository root, from $TPSHOME (set by tearing_physics_suite_env.sh)."""
    try:
        return os.environ['TPSHOME']
    except KeyError:
        raise KeyError('TPSHOME is not set; source tearing_physics_suite_env.sh first.') from None


def gpec_home():
    """The GPEC executables' root (<root>/<code>/<code>): $TPS_GPEC_DIR (e.g. a portable runtime's copies), else
    TPSHOME's GPEC build."""
    return os.environ.get('TPS_GPEC_DIR') or os.path.join(tps_home(), 'submodules/GPEC')
