# Master build script for tearing-physics-suite.
"""
Master build script for tearing-physics-suite.

Orchestrates the full build by calling:
  1. build_netcdf_lapack.build_libraries()  – downloads & builds LAPACK, BLAS,
     HDF5, NetCDF-C and NetCDF-Fortran from source.
  2. build_GPEC_PEST3.build_PEST3()         – clones & builds PEST3 against
     the freshly compiled libraries.
  3. build_GPEC_PEST3.build_GPEC()          – clones & builds GPEC against
     the same libraries.

Usage (from the repository root):
    uv run tearing_physics_suite/build_tearing_physics_suite.py [options]

See README.md for prerequisite system packages (gcc, openmpi, cmake, make).
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure the package is importable when running as a script.
# When Python (or uv) runs this file directly it inserts the script's directory
# (.../tearing_physics_suite/) at sys.path[0].  That directory contains
# tearing_physics_suite.py which shadows the actual tearing_physics_suite
# *package*, so we must remove the script dir and promote the repo root.
_script_dir = Path(__file__).resolve().parent
_repo_dir = _script_dir.parent

# Strip the script directory from sys.path (it will shadow the package)
while str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))

# Ensure the repo root is at the front so the package is found first
if str(_repo_dir) in sys.path:
    sys.path.remove(str(_repo_dir))
sys.path.insert(0, str(_repo_dir))

# Set TPSHOME to the repo root for this build process (overriding any existing value)
if 'TPSHOME' in os.environ and os.environ['TPSHOME'] != str(_repo_dir):
    print(f"WARNING: TPSHOME environment variable ({os.environ['TPSHOME']}) "
          f"does not match repository root ({_repo_dir}). "
          "Overriding TPSHOME to repository root for this build.")
os.environ['TPSHOME'] = str(_repo_dir)

from tearing_physics_suite.build_netcdf_lapack import build_libraries
from tearing_physics_suite.build_GPEC_PEST3 import build_PEST3, build_GPEC


def write_env_file(lib_paths, repo_root, out_path=None):
    """
    Write a bash environment setup file that configures the shell to run
    GPEC (rdcon, stride, dcon) and PEST3 using the installed libraries.
    Automatically detects the Fortran and C compilers and sets appropriate flags.

    Parameters
    ----------
    lib_paths : dict
        Library paths dict as returned by build_libraries().
    repo_root : Path
        Root of the tearing-physics-suite repository.
    out_path : str or Path, optional
        Output path for the bash file.  Defaults to
        ``<repo_root>/tearing_physics_suite_env.sh``.

    Returns
    -------
    Path
        Path to the written file.
    """
    if out_path is None:
        out_path = Path(repo_root) / "tearing_physics_suite_env.sh"
    out_path = Path(out_path)

    # Detect compilers and get appropriate flags
    from tearing_physics_suite.compiler_utils import detect_compilers
    compiler_info = detect_compilers()

    # Collect unique LD_LIBRARY_PATH entries in order
    ld_dirs = []
    for key in ("lapack", "blas", "hdf5", "netcdf", "netcdf-fortran"):
        d = lib_paths.get(key, {}).get("lib")
        if d and str(d) not in ld_dirs:
            ld_dirs.append(str(d))

    utils_prefix = lib_paths.get("lapack", {}).get("prefix", "")
    netcdf_prefix = (
        lib_paths.get("netcdf-fortran", {}).get("prefix", "")
        or lib_paths.get("netcdf", {}).get("prefix", "")
    )
    netcdf_inc = (
        lib_paths.get("netcdf-fortran", {}).get("include", "")
        or str(Path(netcdf_prefix) / "include")
    )

    gpec_bin  = str(Path(repo_root) / "submodules" / "GPEC" / "install" / "bin")
    pest3_bin = str(Path(repo_root) / "submodules" / "PEST3" / "cmake_build" / "pest3")

    lines = [
        "#!/usr/bin/env bash",
        "# tearing_physics_suite_env.sh",
        "# Source this file to set up the shell environment for running GPEC and PEST3.",
        "# Generated automatically by build_tearing_physics_suite.py.",
        "#",
        "# Prerequisites (load before sourcing this file on Slurm/ORCD):",
        "#   module load gcc/12.2.0 openmpi/4.1.4",
        "",
        "# ── Tearing Physics Suite repository root ────────────────────────────────",
        f"export TPSHOME={repo_root}",
        "",
        "# ── Compiler settings (auto-detected, override via FC/CC env vars) ─────────",
        f"export FC={compiler_info['fc']}",
        f"export CC={compiler_info['cc']}",
        f'export FFLAGS="{compiler_info["fflags_base"]}"',
        f"# Compiler type: {compiler_info['compiler_type']}",
        "",
        "# ── LAPACK / BLAS ─────────────────────────────────────────────────────────",
        f"export LAPACKHOME={utils_prefix}",
        "",
        "# ── NetCDF ────────────────────────────────────────────────────────────────",
        f"export NETCDFHOME={netcdf_prefix}",
        f"export NETCDFINC={netcdf_inc}",
        "",
        "# ── Shared library search path ────────────────────────────────────────────",
        "export LD_LIBRARY_PATH=" + ":".join(ld_dirs) + ":${LD_LIBRARY_PATH:-}",
        "",
        "# ── Executables ───────────────────────────────────────────────────────────",
        f"export PATH={gpec_bin}:{pest3_bin}:$PATH",
        "",
        "echo \"tearing_physics_suite environment loaded.\"",
        f"echo \"  TPSHOME : {repo_root}\"",
        f"echo \"  Compiler: {compiler_info['fc']} ({compiler_info.get('compiler_type', 'unknown')})\"",
        f"echo \"  FFLAGS : {compiler_info['fflags_base']}\"",
        f"echo \"  GPEC   : {gpec_bin}\"",
        f"echo \"  PEST3  : {pest3_bin}\"",
        f"echo \"  LAPACKHOME : {utils_prefix}\"",
        f"echo \"  NETCDFHOME : {netcdf_prefix}\"",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    out_path.chmod(0o755)
    return out_path

def build_all(
    install_dir=None,
    build_dir=None,
    skip_libs=False,
    skip_pest3=False,
    skip_gpec=False,
    rebuild_libs=False,
    rebuild_pest3=False,
    rebuild_gpec=False,
    remake_gpec=False,
    debug=False,
    verbose=False,
    gpec_single_threaded=False,
):
    """
    Run the full tearing-physics-suite build pipeline.

    Parameters
    ----------
    install_dir : str, optional
        Installation prefix for the dependency libraries (LAPACK, HDF5,
        NetCDF, NetCDF-Fortran).  Defaults to ``<repo>/submodules/utils``.
    build_dir : str, optional
        Directory in which library sources are downloaded and built.
        Defaults to ``<repo>/build``.
    skip_libs : bool
        Skip building the dependency libraries entirely (assumes they are
        already present in *install_dir*).
    skip_pest3 : bool
        Skip building PEST3.
    skip_gpec : bool
        Skip building GPEC.
    rebuild_libs : bool
        Force rebuild of all dependency libraries even if they appear to be
        already built.
    rebuild_pest3 : bool
        Force a clean rebuild of PEST3.
    rebuild_gpec : bool
        Force a clean rebuild of GPEC (removes and re-clones source).
    remake_gpec : bool
        Re-run ``make clean`` + ``make`` on the existing GPEC source without
        re-downloading it.
    debug : bool
        Run post-build tests where available.
    verbose : bool
        Enable verbose output during library builds.
    gpec_single_threaded : bool
        If True, build GPEC with ``OMPFLAG=`` (empty), disabling OpenMP so
        the compiled executables run single-threaded.  Default is False
        (OpenMP enabled via ``OMPFLAG=-fopenmp``).

    Returns
    -------
    bool
        True if every requested build step succeeded; False otherwise.
    """
    repo_root = _repo_dir

    # Default directories
    if install_dir is None:
        install_dir = str(repo_root / "submodules" / "utils")
    if build_dir is None:
        build_dir = str(repo_root / "build")

    print("=" * 60)
    print("  tearing-physics-suite – master build")
    print("=" * 60)
    print(f"  Repository root : {repo_root}")
    print(f"  Install prefix  : {install_dir}")
    print(f"  Build directory : {build_dir}")
    print()

    all_ok = True
    missing = set()   # populated in Step 1 when skip_libs=False
    lib_paths = {}    # populated in Step 1

    # ------------------------------------------------------------------
    # Step 1: Build dependency libraries (LAPACK, HDF5, NetCDF, NetCDF-F)
    # ------------------------------------------------------------------
    if not skip_libs:
        print("\n" + "=" * 60)
        print("  Step 1 / 3 : Building dependency libraries")
        print("=" * 60 + "\n")

        force_rebuild = (
            ["lapack", "hdf5", "netcdf", "netcdf-fortran"] if rebuild_libs else None
        )

        lib_paths = build_libraries(
            install_dir=install_dir,
            build_dir=build_dir,
            force_rebuild=force_rebuild,
            verbose=verbose,
        )

        required = {"lapack", "blas", "hdf5", "netcdf", "netcdf-fortran"}
        missing = required - set(lib_paths.keys())
        if missing:
            print(f"\nERROR: The following libraries failed to build: {missing}")
            all_ok = False
        else:
            print("\nAll dependency libraries built successfully.")
    else:
        print("\nSkipping dependency library builds (--skip-libs).")
        # Construct lib_paths from existing install_dir so PEST3/GPEC can
        # still find the previously built libraries.
        from tearing_physics_suite.build_netcdf_lapack import (
            LAPACKBuilder,
            HDF5Builder,
            NetCDFBuilder,
            NetCDFFortranBuilder,
        )

        common = {"install_dir": install_dir, "build_dir": build_dir, "verbose": verbose}
        lib_paths = {}
        for Cls, names in [
            (LAPACKBuilder, ["lapack", "blas"]),
            (HDF5Builder, ["hdf5"]),
            (NetCDFBuilder, ["netcdf"]),
            (NetCDFFortranBuilder, ["netcdf-fortran"]),
        ]:
            builder = Cls(**common)
            paths = builder.get_library_paths()
            for name in names:
                lib_paths[name] = paths

    # ------------------------------------------------------------------
    # Step 2: Build PEST3
    # ------------------------------------------------------------------
    pest3_ok = True  # default: not run
    if not skip_pest3:
        print("\n" + "=" * 60)
        print("  Step 2 / 3 : Building PEST3")
        print("=" * 60 + "\n")

        pest3_ok = build_PEST3(
            lib_paths,
            rebuild=rebuild_pest3,
            debug=debug,
        )
        if not pest3_ok:
            print("\nERROR: PEST3 build failed.")
            all_ok = False
    else:
        print("\nSkipping PEST3 build (--skip-pest3).")

    # ------------------------------------------------------------------
    # Step 3: Build GPEC
    # ------------------------------------------------------------------
    gpec_ok = True  # default: not run
    if not skip_gpec:
        print("\n" + "=" * 60)
        print("  Step 3 / 3 : Building GPEC")
        print("=" * 60 + "\n")

        gpec_ok = build_GPEC(
            lib_paths,
            rebuild=rebuild_gpec,
            remake=remake_gpec,
            debug=debug,
            disable_openmp=gpec_single_threaded,
        )
        if not gpec_ok:
            print("\nERROR: GPEC build failed.")
            all_ok = False
    else:
        print("\nSkipping GPEC build (--skip-gpec).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    libs_str  = "SKIPPED" if skip_libs  else ("OK" if all_ok and not missing else "FAILED")
    p3_str    = "SKIPPED" if skip_pest3 else ("OK" if pest3_ok else "FAILED")
    gpec_str  = "SKIPPED" if skip_gpec  else ("OK" if gpec_ok  else "FAILED")
    print("\n" + "=" * 60)
    print("  Build summary")
    print("=" * 60)
    print(f"  Libraries : {libs_str}")
    print(f"  PEST3     : {p3_str}")
    print(f"  GPEC      : {gpec_str}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Write environment setup bash file
    # ------------------------------------------------------------------
    if lib_paths:
        env_file = write_env_file(lib_paths, repo_root)
        print(f"\n  Environment file : {env_file}")
        print("  Source it with:  source tearing_physics_suite_env.sh")
    else:
        print("\n  Skipping env file generation (no lib_paths available).")

    return all_ok


def main():
    """Parse CLI arguments and run the build pipeline."""
    parser = argparse.ArgumentParser(
        description="Build tearing-physics-suite and all its dependencies",
    )

    parser.add_argument(
        "--install-dir",
        type=str,
        default=None,
        help="Installation prefix for dependency libraries (default: submodules/utils)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Directory for downloading / building library sources (default: build/)",
    )

    # Skip flags
    parser.add_argument("--skip-libs", action="store_true", help="Skip building dependency libraries")
    parser.add_argument("--skip-pest3", action="store_true", help="Skip building PEST3")
    parser.add_argument("--skip-gpec", action="store_true", help="Skip building GPEC")

    # Rebuild flags
    parser.add_argument("--rebuild-libs", action="store_true", help="Force rebuild of dependency libraries")
    parser.add_argument("--rebuild-pest3", action="store_true", help="Force clean rebuild of PEST3")
    parser.add_argument("--rebuild-gpec", action="store_true", help="Force clean rebuild of GPEC (removes and re-clones source)")
    parser.add_argument("--remake-gpec", action="store_true",
                        help="Re-run make clean + make on existing GPEC source without re-downloading")

    # Misc
    parser.add_argument("--debug", action="store_true", help="Run post-build verification tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--gpec-single-threaded", action="store_true",
                        help="Build GPEC with OMPFLAG= (no OpenMP) so executables run single-threaded")

    args = parser.parse_args()

    success = build_all(
        install_dir=args.install_dir,
        build_dir=args.build_dir,
        skip_libs=args.skip_libs,
        skip_pest3=args.skip_pest3,
        skip_gpec=args.skip_gpec,
        rebuild_libs=args.rebuild_libs,
        rebuild_pest3=args.rebuild_pest3,
        rebuild_gpec=args.rebuild_gpec,
        remake_gpec=args.remake_gpec,
        debug=args.debug,
        verbose=args.verbose,
        gpec_single_threaded=args.gpec_single_threaded,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

