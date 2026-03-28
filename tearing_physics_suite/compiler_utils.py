"""
Compiler detection and configuration utilities for tearing-physics-suite builds.

Provides centralized compiler detection and flag selection to avoid duplication
across multiple build scripts.
"""

import os
import subprocess
from pathlib import Path


def detect_compilers():
    """
    Detect available Fortran and C compilers using environment variables
    or system PATH searching. Returns a dict with compiler info and flags.
    
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
    import shutil as _shutil
    
    # Check for explicit compiler environment variables first
    fc = os.environ.get('FC')
    cc = os.environ.get('CC')
    f77 = os.environ.get('F77', fc)  # Fall back to FC if F77 not set
    
    # If not set, try to find MPI wrappers, then individual compilers
    if not fc:
        fc = _shutil.which("mpif90") or _shutil.which("mpifort") or _shutil.which("gfortran")
    if not cc:
        cc = _shutil.which("mpicc") or _shutil.which("gcc")
    if not f77:
        f77 = fc
    
    # Determine compiler type by examining the binary name or path
    compiler_type = 'unknown'
    fflags_base = "-O3"  # Default optimization level
    
    # Extract base name from full path or wrapped command
    fc_basename = Path(fc).name if fc else ''
    cc_basename = Path(cc).name if cc else ''
    
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
