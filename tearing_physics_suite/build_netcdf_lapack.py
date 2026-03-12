#!/usr/bin/env python3
"""
Script to download and build netcdf, netcdf-fortran, and lapack libraries.
Inspired by OpenFUSIONToolkit build system patterns.
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
import shutil
import json
from pathlib import Path
import argparse


class LibraryBuilder:
    """Base class for building libraries from source."""
    
    def __init__(self, name, version, url, install_dir=None, build_dir=None, 
                 extra_config_args=None, verbose=False):
        """
        Initialize builder.
        
        Parameters:
        -----------
        name : str
            Name of the library
        version : str
            Version of the library
        url : str
            URL to download the source
        install_dir : str
            Installation directory (default: current directory/install)
        build_dir : str
            Build directory (default: current directory/build)
        extra_config_args : list
            Extra arguments to pass to configure
        verbose : bool
            Print verbose output
        """
        self.name = name
        self.version = version
        self.url = url
        self.verbose = verbose
        
        current_dir = Path.cwd()
        self.install_dir = Path(install_dir) if install_dir else current_dir / 'install'
        self.build_dir = Path(build_dir) if build_dir else current_dir / 'build'
        self.source_dir = self.build_dir / f"{name}-{version}"
        
        self.extra_config_args = extra_config_args or []
        
        # Create directories if they don't exist
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.install_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, message):
        """Print verbose log message."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def run_command(self, cmd, cwd=None, env=None):
        """
        Run a shell command.
        
        Parameters:
        -----------
        cmd : list or str
            Command to run
        cwd : str
            Working directory
        env : dict
            Environment variables
        
        Returns:
        --------
        int
            Return code
        """
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        working_dir = cwd or os.getcwd()
        self.log(f"Running: {' '.join(cmd)}")
        self.log(f"In directory: {working_dir}")
        
        try:
            result = subprocess.run(cmd, cwd=working_dir, env=env, check=False)
            return result.returncode
        except Exception as e:
            print(f"Error running command: {e}")
            return 1
    
    def download(self, force=False):
        """
        Download the library source code.
        
        Parameters:
        -----------
        force : bool
            Force re-download even if file exists
        """
        # Extract filename from URL
        filename = self.url.split('/')[-1]
        filepath = self.build_dir / filename
        
        if filepath.exists() and not force:
            self.log(f"Source already downloaded: {filepath}")
            return True
        
        print(f"Downloading {self.name} v{self.version}...")
        try:
            urllib.request.urlretrieve(self.url, filepath)
            print(f"Downloaded to {filepath}")
            return True
        except Exception as e:
            print(f"Error downloading {self.name}: {e}")
            return False
    
    def extract(self):
        """Extract the downloaded source code."""
        filename = self.url.split('/')[-1]
        filepath = self.build_dir / filename
        
        if not filepath.exists():
            print(f"Source file not found: {filepath}")
            return False
        
        # Check if already extracted
        if self.source_dir.exists():
            self.log(f"Source already extracted: {self.source_dir}")
            return True
        
        print(f"Extracting {self.name}...")
        try:
            with tarfile.open(filepath) as tar:
                tar.extractall(path=self.build_dir)
            print(f"Extracted to {self.source_dir}")
            return True
        except Exception as e:
            print(f"Error extracting {self.name}: {e}")
            return False
    
    def configure(self):
        """Configure the library (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement configure()")
    
    def build(self):
        """Build the library (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement build()")
    
    def get_library_paths(self):
        """
        Get the lib and include directories for this library.
        
        Returns:
        --------
        dict
            Dictionary with 'lib' and 'include' paths
        """
        # Check for lib and lib64 - prefer the one with more .so files
        lib_dir = None
        lib64_dir = None
        
        lib_candidate = self.install_dir / 'lib'
        lib64_candidate = self.install_dir / 'lib64'
        
        lib_so_count = 0
        lib64_so_count = 0
        
        if lib_candidate.exists():
            lib_so_count = len(list(lib_candidate.glob("*.so*")))
        
        if lib64_candidate.exists():
            lib64_so_count = len(list(lib64_candidate.glob("*.so*")))
        
        # Choose the directory with shared object files
        if lib64_so_count > 0 and lib64_so_count >= lib_so_count:
            lib_dir = lib64_candidate
        elif lib_so_count > 0:
            lib_dir = lib_candidate
        elif lib_candidate.exists():
            lib_dir = lib_candidate
        elif lib64_candidate.exists():
            lib_dir = lib64_candidate
        
        inc_dir = self.install_dir / "include"
        
        return {
            'lib': str(lib_dir) if lib_dir else None,
            'include': str(inc_dir) if inc_dir.exists() else None,
            'prefix': str(self.install_dir)
        }
    
    def full_build(self, force_download=False):
        """Execute full build process: download, extract, configure, build."""
        print(f"\n{'='*60}")
        print(f"Building {self.name} v{self.version}")
        print(f"{'='*60}")
        
        try:
            if not self.download(force=force_download):
                return False
            
            if not self.extract():
                return False
            
            if not self.configure():
                return False
            
            if not self.build():
                return False
            
            print(f"Successfully built {self.name}!")
            return True
        except Exception as e:
            print(f"Error during build of {self.name}: {e}")
            return False


class LAPACKBuilder(LibraryBuilder):
    """Builder for LAPACK library."""
    
    def __init__(self, version="3.12.0", **kwargs):
        url = f"https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v{version}.tar.gz"
        super().__init__("lapack", version, url, **kwargs)
    
    def configure(self):
        """Configure LAPACK using CMake."""
        print(f"Configuring {self.name}...")
        
        build_subdir = self.source_dir / "cmake_build"
        # Clean stale CMake cache to ensure fresh configuration
        if build_subdir.exists():
            shutil.rmtree(build_subdir)
        build_subdir.mkdir(parents=True, exist_ok=True)
        
        cmake_cmd = [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_TESTING=OFF",
            ".."
        ]
        
        if self.run_command(cmake_cmd, cwd=build_subdir) != 0:
            print(f"Error configuring {self.name}")
            return False
        
        self.cmake_build_dir = build_subdir
        return True
    
    def build(self):
        """Build LAPACK."""
        print(f"Building {self.name}...")
        
        if not hasattr(self, 'cmake_build_dir'):
            print(f"cmake_build_dir not set. Run configure() first.")
            return False
        
        build_cmd = ["cmake", "--build", ".", "-j", "4"]
        
        if self.run_command(build_cmd, cwd=self.cmake_build_dir) != 0:
            print(f"Error building {self.name}")
            return False
        
        # Install
        print(f"Installing {self.name}...")
        install_cmd = ["cmake", "--install", "."]
        
        if self.run_command(install_cmd, cwd=self.cmake_build_dir) != 0:
            print(f"Error installing {self.name}")
            return False
        
        return True


class HDF5Builder(LibraryBuilder):
    """Builder for HDF5 library."""
    
    def __init__(self, version="1.14.6", **kwargs):
        # HDF5 releases are available from the GitHub releases page
        # Tag format: hdf5_X.Y.Z (with dots, not underscores)
        url = f"https://github.com/HDFGroup/hdf5/releases/download/hdf5_{version}/hdf5-{version}.tar.gz"
        super().__init__("hdf5", version, url, **kwargs)
        # Update source_dir name to match extracted directory
        self.source_dir = self.build_dir / f"hdf5-{version}"
    
    def configure(self):
        """Configure HDF5 using CMake."""
        print(f"Configuring {self.name}...")
        
        build_subdir = self.source_dir / "cmake_build"
        # Clean stale CMake cache to ensure fresh configuration
        if build_subdir.exists():
            shutil.rmtree(build_subdir)
        build_subdir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 is a pure C library and should NOT link against any MPI.
        # Explicitly set CC to the plain C compiler (not the MPI wrapper)
        # to avoid accidentally pulling in MPI dependencies.
        import shutil as _shutil
        gcc_path = _shutil.which("gcc")
        
        cmake_cmd = [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DHDF5_BUILD_TOOLS=OFF",
            "-DHDF5_BUILD_EXAMPLES=OFF",
            "-DHDF5_BUILD_TESTS=OFF",
            "-DHDF5_ENABLE_PARALLEL=OFF",
        ]
        
        if gcc_path:
            cmake_cmd.append(f"-DCMAKE_C_COMPILER={gcc_path}")
        
        cmake_cmd.append("..")
        
        if self.run_command(cmake_cmd, cwd=build_subdir) != 0:
            print(f"Error configuring {self.name}")
            return False
        
        self.cmake_build_dir = build_subdir
        return True
    
    def build(self):
        """Build HDF5."""
        print(f"Building {self.name}...")
        
        if not hasattr(self, 'cmake_build_dir'):
            print(f"cmake_build_dir not set. Run configure() first.")
            return False
        
        build_cmd = ["cmake", "--build", ".", "-j", "4"]
        
        if self.run_command(build_cmd, cwd=self.cmake_build_dir) != 0:
            print(f"Error building {self.name}")
            return False
        
        # Install
        print(f"Installing {self.name}...")
        install_cmd = ["cmake", "--install", "."]
        
        if self.run_command(install_cmd, cwd=self.cmake_build_dir) != 0:
            print(f"Error installing {self.name}")
            return False
        
        return True


class NetCDFBuilder(LibraryBuilder):
    """Builder for NetCDF library."""
    
    def __init__(self, version="4.9.2", hdf5_dir=None, lapack_dir=None, **kwargs):
        # NetCDF-C releases are on GitHub releases
        url = f"https://github.com/Unidata/netcdf-c/archive/refs/tags/v{version}.tar.gz"
        super().__init__("netcdf", version, url, **kwargs)
        self.hdf5_dir = hdf5_dir
        self.lapack_dir = lapack_dir
        # Update source_dir name to match extracted directory
        self.source_dir = self.build_dir / f"netcdf-c-{version}"
    
    def clean_source(self):
        """Clean previous autotools build artifacts so configure runs fresh."""
        config_status = self.source_dir / "config.status"
        if config_status.exists():
            self.log("Cleaning previous build (make distclean)...")
            self.run_command(["make", "distclean"], cwd=self.source_dir)
    
    def configure(self):
        """Configure NetCDF."""
        print(f"Configuring {self.name}...")
        
        # Clean stale autotools state from any prior build
        self.clean_source()
        
        env = os.environ.copy()
        
        # Explicitly set compilers so autotools doesn't pick up a wrong MPI
        import shutil as _shutil
        cc = _shutil.which("mpicc") or _shutil.which("gcc")
        if cc:
            env['CC'] = cc
        
        # Helper function to find lib directory
        def find_lib_dir(base_dir):
            for libname in ['lib', 'lib64']:
                candidate = base_dir / libname
                if candidate.exists():
                    return candidate
            return base_dir / 'lib'
        
        # Set paths for HDF5 and LAPACK
        if self.hdf5_dir:
            hdf5_lib = find_lib_dir(self.hdf5_dir)
            hdf5_inc = self.hdf5_dir / "include"
            env['LD_LIBRARY_PATH'] = f"{hdf5_lib}:{env.get('LD_LIBRARY_PATH', '')}"
            env['LDFLAGS'] = f"-L{hdf5_lib} {env.get('LDFLAGS', '')}"
            env['CPPFLAGS'] = f"-I{hdf5_inc} {env.get('CPPFLAGS', '')}"
            env['HDF5_DIR'] = str(self.hdf5_dir)
        
        if self.lapack_dir:
            lapack_lib = find_lib_dir(self.lapack_dir)
            env['LD_LIBRARY_PATH'] = f"{lapack_lib}:{env.get('LD_LIBRARY_PATH', '')}"
            env['LDFLAGS'] = f"{env.get('LDFLAGS', '')} -L{lapack_lib}"
            env['CPPFLAGS'] = f"{env.get('CPPFLAGS', '')} -I{self.lapack_dir}/include"
        
        config_cmd = [
            str(self.source_dir / "configure"),
            f"--prefix={self.install_dir}",
            "--enable-shared",
            "--disable-static",
            "--disable-dap",  # Disable DAP to avoid extra dependencies
            "--disable-libxml2",  # Use bundled xml2 parser to avoid external dependency
        ]
        
        # Add HDF5 specific configuration if available
        if self.hdf5_dir:
            hdf5_prefix = str(self.hdf5_dir)
            config_cmd.extend([
                f"--with-hdf5={hdf5_prefix}",
            ])
        
        config_cmd.extend(self.extra_config_args)
        
        if self.run_command(config_cmd, cwd=self.source_dir, env=env) != 0:
            print(f"Error configuring {self.name}")
            return False
        
        return True
    
    def build(self):
        """Build and install NetCDF."""
        print(f"Building {self.name}...")
        
        make_cmd = ["make", "-j", "4"]
        
        if self.run_command(make_cmd, cwd=self.source_dir) != 0:
            print(f"Error building {self.name}")
            return False
        
        # Install
        print(f"Installing {self.name}...")
        install_cmd = ["make", "install"]
        
        if self.run_command(install_cmd, cwd=self.source_dir) != 0:
            print(f"Error installing {self.name}")
            return False
        
        return True


class NetCDFFortranBuilder(LibraryBuilder):
    """Builder for NetCDF-Fortran library."""
    
    def __init__(self, version="4.6.1", netcdf_dir=None, hdf5_dir=None, lapack_dir=None, **kwargs):
        # NetCDF-Fortran releases are on GitHub releases
        url = f"https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v{version}.tar.gz"
        super().__init__("netcdf-fortran", version, url, **kwargs)
        self.netcdf_dir = netcdf_dir
        self.hdf5_dir = hdf5_dir
        self.lapack_dir = lapack_dir
        # Update source_dir name to match extracted directory
        self.source_dir = self.build_dir / f"netcdf-fortran-{version}"
    
    def clean_source(self):
        """Clean previous autotools build artifacts so configure runs fresh."""
        config_status = self.source_dir / "config.status"
        if config_status.exists():
            self.log("Cleaning previous build (make distclean)...")
            self.run_command(["make", "distclean"], cwd=self.source_dir)
    
    def configure(self):
        """Configure NetCDF-Fortran."""
        print(f"Configuring {self.name}...")
        
        # Clean stale autotools state from any prior build
        self.clean_source()
        
        env = os.environ.copy()
        
        # Explicitly set compilers so autotools uses the correct MPI wrappers.
        # This is critical — without it, configure may pick up a stale/wrong
        # MPI (e.g. Intel oneAPI MPI linked against mvapich2) and the resulting
        # libnetcdff.so will depend on the wrong libmpifort.so.
        import shutil as _shutil
        fc = _shutil.which("mpif90") or _shutil.which("gfortran")
        cc = _shutil.which("mpicc") or _shutil.which("gcc")
        if fc:
            env['FC'] = fc
            env['F77'] = fc
        if cc:
            env['CC'] = cc
        
        # Helper function to find lib directory
        def find_lib_dir(base_dir):
            for libname in ['lib', 'lib64']:
                candidate = base_dir / libname
                if candidate.exists():
                    return candidate
            return base_dir / 'lib'
        
        # Set paths for NetCDF and LAPACK
        if self.netcdf_dir:
            netcdf_lib = find_lib_dir(self.netcdf_dir)
            netcdf_inc = self.netcdf_dir / "include"
            env['LD_LIBRARY_PATH'] = f"{netcdf_lib}:{env.get('LD_LIBRARY_PATH', '')}"
            env['LDFLAGS'] = f"-L{netcdf_lib}"
            env['CPPFLAGS'] = f"-I{netcdf_inc}"
        
        if self.hdf5_dir:
            hdf5_lib = find_lib_dir(self.hdf5_dir)
            hdf5_inc = self.hdf5_dir / "include"
            env['LD_LIBRARY_PATH'] = f"{hdf5_lib}:{env.get('LD_LIBRARY_PATH', '')}"
            env['LDFLAGS'] = f"{env.get('LDFLAGS', '')} -L{hdf5_lib}"
            env['CPPFLAGS'] = f"{env.get('CPPFLAGS', '')} -I{hdf5_inc}"
        
        if self.lapack_dir:
            lapack_lib = find_lib_dir(self.lapack_dir)
            env['LD_LIBRARY_PATH'] = f"{lapack_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        
        config_cmd = [
            str(self.source_dir / "configure"),
            f"--prefix={self.install_dir}",
            "--enable-shared",
            "--disable-static",
        ] + self.extra_config_args
        
        if self.run_command(config_cmd, cwd=self.source_dir, env=env) != 0:
            print(f"Error configuring {self.name}")
            return False
        
        return True
    
    def build(self):
        """Build and install NetCDF-Fortran."""
        print(f"Building {self.name}...")
        
        make_cmd = ["make", "-j", "4"]
        
        if self.run_command(make_cmd, cwd=self.source_dir) != 0:
            print(f"Error building {self.name}")
            return False
        
        # Install
        print(f"Installing {self.name}...")
        install_cmd = ["make", "install"]
        
        if self.run_command(install_cmd, cwd=self.source_dir) != 0:
            print(f"Error installing {self.name}")
            return False
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and build netcdf, netcdf-fortran, and lapack libraries"
    )
    parser.add_argument(
        "--install-dir",
        type=str,
        default=None,
        help="Installation directory (default: ./install)"
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Build directory (default: ./build)"
    )
    parser.add_argument(
        "--lapack-version",
        type=str,
        default="3.12.0",
        help="LAPACK version (default: 3.12.0)"
    )
    parser.add_argument(
        "--hdf5-version",
        type=str,
        default="1.14.6",
        help="HDF5 version (default: 1.14.6)"
    )
    parser.add_argument(
        "--netcdf-version",
        type=str,
        default="4.9.2",
        help="NetCDF version (default: 4.9.2)"
    )
    parser.add_argument(
        "--netcdf-fortran-version",
        type=str,
        default="4.6.1",
        help="NetCDF-Fortran version (default: 4.6.1)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of source files"
    )
    parser.add_argument(
        "--skip-lapack",
        action="store_true",
        help="Skip building LAPACK"
    )
    parser.add_argument(
        "--skip-hdf5",
        action="store_true",
        help="Skip building HDF5"
    )
    parser.add_argument(
        "--skip-netcdf",
        action="store_true",
        help="Skip building NetCDF"
    )
    parser.add_argument(
        "--skip-netcdf-fortran",
        action="store_true",
        help="Skip building NetCDF-Fortran"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create common kwargs for all builders
    common_kwargs = {
        "install_dir": args.install_dir,
        "build_dir": args.build_dir,
        "verbose": args.verbose,
    }
    
    # Determine directories
    build_dir = Path(args.build_dir) if args.build_dir else Path.cwd() / "build"
    install_dir = Path(args.install_dir) if args.install_dir else Path.cwd() / "install"
    
    print(f"Build directory: {build_dir}")
    print(f"Install directory: {install_dir}")
    
    results = {}
    builders = {}  # Store builder objects to retrieve paths later
    
    # Build LAPACK
    if not args.skip_lapack:
        lapack = LAPACKBuilder(version=args.lapack_version, **common_kwargs)
        builders['lapack'] = lapack
        results['lapack'] = lapack.full_build(force_download=args.force_download)
    else:
        print("Skipping LAPACK build")
        results['lapack'] = None
    
    # Build HDF5
    if not args.skip_hdf5:
        hdf5 = HDF5Builder(version=args.hdf5_version, **common_kwargs)
        builders['hdf5'] = hdf5
        results['hdf5'] = hdf5.full_build(force_download=args.force_download)
    else:
        print("Skipping HDF5 build")
        results['hdf5'] = None
    
    # Build NetCDF
    if not args.skip_netcdf:
        netcdf = NetCDFBuilder(version=args.netcdf_version, 
                              hdf5_dir=install_dir if not args.skip_hdf5 else None,
                              lapack_dir=install_dir if not args.skip_lapack else None,
                              **common_kwargs)
        builders['netcdf'] = netcdf
        results['netcdf'] = netcdf.full_build(force_download=args.force_download)
    else:
        print("Skipping NetCDF build")
        results['netcdf'] = None
    
    # Build NetCDF-Fortran
    if not args.skip_netcdf_fortran:
        netcdf_fortran = NetCDFFortranBuilder(version=args.netcdf_fortran_version,
                                            netcdf_dir=install_dir if not args.skip_netcdf else None,
                                            hdf5_dir=install_dir if not args.skip_hdf5 else None,
                                            lapack_dir=install_dir if not args.skip_lapack else None,
                                            **common_kwargs)
        builders['netcdf-fortran'] = netcdf_fortran
        results['netcdf-fortran'] = netcdf_fortran.full_build(force_download=args.force_download)
    else:
        print("Skipping NetCDF-Fortran build")
        results['netcdf-fortran'] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("Build Summary")
    print(f"{'='*60}")
    for lib, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "SUCCESS"
        else:
            status = "FAILED"
        print(f"{lib:20s}: {status}")
    
    # Print install paths
    print(f"\n{'='*60}")
    print("Installation paths:")
    print(f"{'='*60}")
    if install_dir.exists():
        if (install_dir / "lib").exists():
            print(f"Libraries: {install_dir / 'lib'}")
        if (install_dir / "include").exists():
            print(f"Headers:   {install_dir / 'include'}")
    
    # Collect library paths
    library_paths = {}
    for lib_name, builder in builders.items():
        if results.get(lib_name, False):
            library_paths[lib_name] = builder.get_library_paths()
            print(f"\n{lib_name.upper()}:")
            print(f"  Lib:     {library_paths[lib_name]['lib']}")
            print(f"  Include: {library_paths[lib_name]['include']}")
    
    # Check for failures
    failures = [lib for lib, success in results.items() if success is False]
    if failures:
        print(f"\nFailed to build: {', '.join(failures)}")
        return 1, library_paths
    
    return 0, library_paths


def is_library_built(install_dir, lib_names=None, verbose=False):
    """
    Check if a library is already built and has necessary files.
    
    Parameters:
    -----------
    install_dir : str or Path
        Installation directory to check
    lib_names : list of str, optional
        List of library names to look for (e.g., ['lapack', 'blas'])
        If None, checks for any .so files
    verbose : bool
        Print verbose output
    
    Returns:
    --------
    bool
        True if library appears to be built with necessary files, False otherwise
    """
    install_dir = Path(install_dir)
    
    if not install_dir.exists():
        if verbose:
            print(f"Installation directory does not exist: {install_dir}")
        return False
    
    # Check for lib or lib64 directory
    lib_dir = None
    for libname in ['lib', 'lib64']:
        candidate = install_dir / libname
        if candidate.exists():
            lib_dir = candidate
            break
    
    if not lib_dir:
        if verbose:
            print(f"No lib or lib64 directory found in {install_dir}")
        return False
    
    if verbose:
        print(f"Found library directory: {lib_dir}")
    
    # Check for library files
    if lib_names:
        for lib_name in lib_names:
            # Look for .so files matching the pattern
            so_files = list(lib_dir.glob(f"lib{lib_name}*.so*"))
            if not so_files:
                if verbose:
                    print(f"No library files found for {lib_name} in {lib_dir}")
                return False
            if verbose:
                print(f"Found library files for {lib_name}: {so_files}")
    else:
        # Just check if there are any .so files
        so_files = list(lib_dir.glob("*.so*"))
        if not so_files:
            if verbose:
                print(f"No library files (.so) found in {lib_dir}")
            return False
        if verbose:
            print(f"Found {len(so_files)} library files in {lib_dir}")
    
    # Check for include directory
    include_dir = install_dir / "include"
    if not include_dir.exists():
        if verbose:
            print(f"Include directory does not exist: {include_dir}")
        return False
    
    if verbose:
        print(f"✓ Library appears to be built correctly in {install_dir}")
    
    return True


def build_libraries(install_dir=None, build_dir=None, lapack_version="3.12.0", hdf5_version="1.14.6", 
                    netcdf_version="4.9.2", netcdf_fortran_version="4.6.1",
                    skip_lapack=False, skip_hdf5=False, skip_netcdf=False, skip_netcdf_fortran=False,
                    force_download=False, force_rebuild=None, verbose=False):
    """
    Build libraries programmatically and return library paths.
    
    This function can be imported and called from other modules to build the libraries
    and get their installation paths.
    
    Parameters:
    -----------
    install_dir : str, optional
        Installation directory (default: ./install)
    build_dir : str, optional
        Build directory (default: ./build)
    lapack_version : str
        LAPACK version (default: 3.12.0)
    hdf5_version : str
        HDF5 version (default: 1.14.6)
    netcdf_version : str
        NetCDF version (default: 4.9.2)
    netcdf_fortran_version : str
        NetCDF-Fortran version (default: 4.6.1)
    skip_lapack : bool
        Skip building LAPACK
    skip_hdf5 : bool
        Skip building HDF5
    skip_netcdf : bool
        Skip building NetCDF
    skip_netcdf_fortran : bool
        Skip building NetCDF-Fortran
    force_download : bool
        Force re-download of source files
    force_rebuild : list of str, optional
        List of library names to force rebuild even if already built
        Valid names: 'lapack', 'hdf5', 'netcdf', 'netcdf-fortran'
        Example: force_rebuild=['lapack', 'netcdf']
    verbose : bool
        Verbose output
    
    Returns:
    --------
    dict
        Dictionary mapping library names to their paths:
        {
            'lapack': {'lib': path, 'include': path, 'prefix': path},
            'hdf5': {'lib': path, 'include': path, 'prefix': path},
            'netcdf': {'lib': path, 'include': path, 'prefix': path},
            'netcdf-fortran': {'lib': path, 'include': path, 'prefix': path}
        }
    """
    # Normalize force_rebuild to a set for faster lookups
    if force_rebuild is None:
        force_rebuild = set()
    else:
        force_rebuild = set(str(lib).lower() for lib in force_rebuild)
    
    # Create common kwargs for all builders
    common_kwargs = {
        "install_dir": install_dir,
        "build_dir": build_dir,
        "verbose": verbose,
    }
    
    # Determine directories
    build_dir_path = Path(build_dir) if build_dir else Path.cwd() / "build"
    install_dir_path = Path(install_dir) if install_dir else Path.cwd() / "install"
    
    library_paths = {}
    builders = {}
    
    # Build LAPACK
    if not skip_lapack:
        # Check if LAPACK should be force rebuilt or needs building
        should_rebuild = 'lapack' in force_rebuild
        
        if not should_rebuild and is_library_built(install_dir_path, lib_names=['lapack', 'blas'], verbose=verbose):
            print(f"LAPACK already built in {install_dir_path}, skipping rebuild")
            lapack = LAPACKBuilder(version=lapack_version, **common_kwargs)
            library_paths['lapack'] = lapack.get_library_paths()
            library_paths['blas'] = lapack.get_library_paths()  # LAPACK build includes BLAS
        else:
            if should_rebuild:
                print(f"Force rebuilding LAPACK...")
            lapack = LAPACKBuilder(version=lapack_version, **common_kwargs)
            builders['lapack'] = lapack
            if lapack.full_build(force_download=force_download):
                library_paths['lapack'] = lapack.get_library_paths()
                library_paths['blas'] = lapack.get_library_paths()  # LAPACK build includes BLAS
    
    # Build HDF5
    if not skip_hdf5:
        # Check if HDF5 should be force rebuilt or needs building
        should_rebuild = 'hdf5' in force_rebuild
        
        if not should_rebuild and is_library_built(install_dir_path, lib_names=['hdf5'], verbose=verbose):
            print(f"HDF5 already built in {install_dir_path}, skipping rebuild")
            hdf5 = HDF5Builder(version=hdf5_version, **common_kwargs)
            library_paths['hdf5'] = hdf5.get_library_paths()
        else:
            if should_rebuild:
                print(f"Force rebuilding HDF5...")
            hdf5 = HDF5Builder(version=hdf5_version, **common_kwargs)
            builders['hdf5'] = hdf5
            if hdf5.full_build(force_download=force_download):
                library_paths['hdf5'] = hdf5.get_library_paths()
    
    # Build NetCDF
    if not skip_netcdf:
        # Check if NetCDF should be force rebuilt or needs building
        should_rebuild = 'netcdf' in force_rebuild
        
        if not should_rebuild and is_library_built(install_dir_path, lib_names=['netcdf'], verbose=verbose):
            print(f"NetCDF already built in {install_dir_path}, skipping rebuild")
            netcdf = NetCDFBuilder(version=netcdf_version,
                                  hdf5_dir=install_dir_path if not skip_hdf5 else None,
                                  lapack_dir=install_dir_path if not skip_lapack else None,
                                  **common_kwargs)
            library_paths['netcdf'] = netcdf.get_library_paths()
        else:
            if should_rebuild:
                print(f"Force rebuilding NetCDF...")
            netcdf = NetCDFBuilder(version=netcdf_version,
                                  hdf5_dir=install_dir_path if not skip_hdf5 else None,
                                  lapack_dir=install_dir_path if not skip_lapack else None,
                                  **common_kwargs)
            builders['netcdf'] = netcdf
            if netcdf.full_build(force_download=force_download):
                library_paths['netcdf'] = netcdf.get_library_paths()
    
    # Build NetCDF-Fortran
    if not skip_netcdf_fortran:
        # Check if NetCDF-Fortran should be force rebuilt or needs building
        should_rebuild = 'netcdf-fortran' in force_rebuild
        
        if not should_rebuild and is_library_built(install_dir_path, lib_names=['netcdff'], verbose=verbose):
            print(f"NetCDF-Fortran already built in {install_dir_path}, skipping rebuild")
            netcdf_fortran = NetCDFFortranBuilder(version=netcdf_fortran_version,
                                                netcdf_dir=install_dir_path if not skip_netcdf else None,
                                                hdf5_dir=install_dir_path if not skip_hdf5 else None,
                                                lapack_dir=install_dir_path if not skip_lapack else None,
                                                **common_kwargs)
            library_paths['netcdf-fortran'] = netcdf_fortran.get_library_paths()
        else:
            if should_rebuild:
                print(f"Force rebuilding NetCDF-Fortran...")
            netcdf_fortran = NetCDFFortranBuilder(version=netcdf_fortran_version,
                                                netcdf_dir=install_dir_path if not skip_netcdf else None,
                                                hdf5_dir=install_dir_path if not skip_hdf5 else None,
                                                lapack_dir=install_dir_path if not skip_lapack else None,
                                                **common_kwargs)
            builders['netcdf-fortran'] = netcdf_fortran
            if netcdf_fortran.full_build(force_download=force_download):
                library_paths['netcdf-fortran'] = netcdf_fortran.get_library_paths()
    
    return library_paths


if __name__ == "__main__":
    exit_code, lib_paths = main()
    
    # Print library paths as JSON for easy parsing
    if lib_paths:
        print(f"\n{'='*60}")
        print("Library paths (JSON format):")
        print(f"{'='*60}")
        print(json.dumps(lib_paths, indent=2, default=str))
    
    sys.exit(exit_code)
