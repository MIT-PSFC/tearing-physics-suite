"""
Compiler detection and configuration utilities for tearing-physics-suite builds.

Provides centralized compiler detection and flag selection to avoid duplication
across multiple build scripts.
"""

import os
import shutil
import subprocess
from pathlib import Path


def _is_mpi_wrapper(compiler):
    """True for MPI compiler wrappers such as mpif90, mpifort or mpicc."""
    return bool(compiler) and Path(compiler).name.startswith("mpi")


def _which_first(*names):
    """Full path of the first of *names* found on PATH, or None."""
    return next((p for p in map(shutil.which, names) if p), None)


def detect_compilers(mpi=True):
    """
    Detect available Fortran and C compilers using environment variables
    or system PATH searching. Returns a dict with compiler info and flags.

    Parameters
    ----------
    mpi : bool
        If True (default), fall back to the MPI wrappers (mpif90, mpicc) when
        FC/CC are unset. If False, MPI wrappers are ignored, also when set in
        FC/CC/F77, and the plain GNU compilers are used instead (for codes
        without MPI, e.g. GPEC, whose makefile rejects wrapper names).

    Returns
    -------
    dict
        {
            'fc': path/name of Fortran compiler,
            'cc': path/name of C compiler,
            'f77': path/name of F77 compiler (usually same as fc),
            'compiler_type': 'gfortran' | 'ifort' | 'ifx' | 'pgfortran' | 'flang' | 'unknown',
            'fflags_base': base compiler flags for the detected Fortran compiler
        }
    """
    # Check for explicit compiler environment variables first
    fc = os.environ.get('FC')
    cc = os.environ.get('CC')
    f77 = os.environ.get('F77', fc)  # Fall back to FC if F77 not set
    if not mpi:
        fc, cc, f77 = (None if _is_mpi_wrapper(c) else c for c in (fc, cc, f77))

    # If not set, try to find MPI wrappers (if mpi), then individual compilers
    if not fc:
        fc = _which_first(*(["mpif90", "mpifort"] if mpi else []), "gfortran")
    if not cc:
        cc = _which_first(*(["mpicc"] if mpi else []), "gcc")
    if not f77:
        f77 = fc

    # Determine compiler type by examining the binary name or path
    compiler_type = 'unknown'
    fflags_base = "-O3"  # Default optimization level

    # Extract base name from full path or wrapped command
    fc_basename = Path(fc).name if fc else ''

    # Detect compiler type from executable name
    if 'ifort' in fc_basename or 'ifx' in fc_basename:
        compiler_type = 'ifort' if 'ifort' in fc_basename else 'ifx'
        # ifort doesn't need -fallow-argument-mismatch
        # Instead, it may need -assume byterecl for some legacy code
        fflags_base = "-O3 -assume byterecl"
    elif 'gfortran' in fc_basename or 'mpif90' in fc_basename:
        compiler_type = 'gfortran'
        # gfortran 10+ needs this flag for legacy Fortran code
        fflags_base = "-O3 -fallow-argument-mismatch"
    elif 'pgfortran' in fc_basename or 'pgf95' in fc_basename:
        compiler_type = 'pgfortran'
        fflags_base = "-O3 -Mrecursive"
    elif 'flang' in fc_basename:
        compiler_type = 'flang'
        fflags_base = "-O3 -fallow-argument-mismatch"  # flang is GNU-compatible
    elif 'armflang' in fc_basename:
        compiler_type = 'armflang'
        fflags_base = "-O3"

    # Fallback if we couldn't determine type but have a compiler
    if fc and compiler_type == 'unknown':
        try:
            result = subprocess.run(
                [fc, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_output = result.stdout + result.stderr
            if 'gfortran' in version_output.lower():
                compiler_type = 'gfortran'
                fflags_base = "-O3 -fallow-argument-mismatch"
            elif 'ifort' in version_output.lower():
                compiler_type = 'ifort'
                fflags_base = "-O3 -assume byterecl"
            elif 'pgfortran' in version_output.lower() or 'pgi' in version_output.lower():
                compiler_type = 'pgfortran'
                fflags_base = "-O3 -Mrecursive"
        except Exception:
            pass  # Continue with defaults

    return {
        'fc': fc,
        'cc': cc,
        'f77': f77,
        'compiler_type': compiler_type,
        'fflags_base': fflags_base
    }


def get_cmake_fortran_flags(compiler_type='gfortran'):
    """
    Get appropriate CMAKE_Fortran_FLAGS for different compilers.

    Parameters
    ----------
    compiler_type : str
        Type of Fortran compiler ('gfortran', 'ifort', 'ifx', 'pgfortran', etc.)

    Returns
    -------
    str
        Compiler flags appropriate for the detected compiler
    """
    flags = {
        'gfortran': '-O3 -fallow-argument-mismatch',
        'ifort': '-O3 -assume byterecl',
        'ifx': '-O3 -assume byterecl',
        'pgfortran': '-O3 -Mrecursive',
        'flang': '-O3 -fallow-argument-mismatch',
        'armflang': '-O3',
    }
    return flags.get(compiler_type, '-O3')


def gnu_major_version(compiler):
    """Major version of a GNU compiler (or a wrapper around one), else None."""
    def query(flag):
        return subprocess.run([compiler, flag], capture_output=True,
                              text=True, timeout=30).stdout
    try:
        if "Free Software Foundation" not in query("--version"):
            return None
        return int(query("-dumpversion").split(".")[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def get_cmake_c_flags(cc):
    """
    Get CMAKE_C_FLAGS for the C compiler *cc*. GCC >= 14 turns implicit
    function declarations into errors; -fpermissive makes them warnings again,
    as legacy C sources (PEST3 portlib) rely on them.
    """
    major = gnu_major_version(cc)
    return "-fpermissive" if major and major >= 14 else ""
