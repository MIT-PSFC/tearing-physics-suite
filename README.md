# tearing-physics-suite

All development to follow Vincent Driessen's GitFlow <https://nvie.com/posts/a-successful-git-branching-model/> to safely add features.

# Installation 

Short version: (requires uv, gcc, openmpi, cmake and make)  
    cd path/to/tearing-physics-suite  
    uv sync  
    uv run tearing_phyics_suite/build_tearing_physics_suite.py  
    uv run tests/unit_test_suite.py  
    source tearing_physics_suite_env.sh (optional)  

Long version:

1. Set up python environment:
    Please use uv python software (simple installation guide here - https://github.com/astral-sh/uv).
    Once you have uv, initialise the uv python environment by entering the tearing-physics-suite master and entering terminal command 'uv sync'.
    You can now open python in the terminal with command 'uv run python' or scripts using 'uv run python_script.py'.

2. Set up fortran environment:
    You need to install the software 'gcc', 'openmpi', 'cmake' and 'make' on your system. I used versions gcc/12.2.0 and openmpi/4.1.4, make/4.2.1 and cmake greater than 3.5 but you can try other versions at your own risk.
    Installation may be trivial on a cluster with commands such as 'module load gcc/<version>', 'module load openmpi/<version>'. 
    If you have a linux system with sudo privilege, you can run the terminal command 'apt install gcc openmpi cmake make'. On macOS you can download Homebrew and run in the terminal 'brew install gcc openmpi cmake make'.

    Once you have gcc, openmpi, make and cmake, you can run python script build_tearing_physics_suite.py using the command 'uv run path/to/tearing-physics-suite/tearing_phyics_suite/build_tearing_physics_suite.py'.
    Then, slowly and surely, this AI-written gem of a script will download the following codes from the following links:
        lapack - https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
        hdf5   - https://github.com/HDFGroup/hdf5/releases/download/hdf5_{version}/hdf5-1.14.6.tar.g
        netcdf - https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.2.tar.gz
        netcdf-fortran - https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.6.1.tar.gz
        scimake - https://github.com/Tech-XCorp/scimake.git
        PEST3   - https://github.com/MIT-PSFC/PEST3 (we have a local copy of the publically available scripts from https://svn.code.sf.net/p/pest3code/code/)
        GPEC    - https://github.com/PrincetonUniversity/GPEC
    After downloading these packages, build_tearing_physics_suite.py will build them using a combination of make and cmake software. It will link them and they should work. I recommend debugging this script with an AI agent if something
    goes wrong, but I'll make sure it works on the clusters OMEGA, SPC-LAC and Engaging, as well as macOS.

3. Run tests:
    'uv run path/to/tearing-physics-suite/tests/unit_test_suite.py' will go through and tell you if the package is behaving correctly. 
    Again I'll make sure it works on the clusters OMEGA, SPC-LAC and Engaging, as well as macOS.

Optional: 
    To run the build PEST3 and GPEC packages from the terminal, without using the python wrappers, you can load all the necessary paths and environmental variables using the command
    source path/to/tearing-physics-suite/tearing_physics_suite_env.sh
    This requires having first ran build_tearing_physics_suite.py, and loaded gcc and openmpi. For example on engaging, the terminal commands are:
        module load gcc/12.2.0 openmpi/4.1.4
        source tearing_physics_suite_env.sh

# Examples

See tests/tearing_physics_suite_tests.py for example calculations of linear and nonlinear plasma tearing stability (without rotation terms).

See tests/rotation_tests.py for an example calculation of the nonlinear tearing stability, with dimensionless rotation-decorrelation timescale ratios included.

To run parallel calculation examples, call 'uv run tests/parallelisation_tests.py' in a multi-CPU slurm environment.

To run example numerical sensitivity scans, and input variable scans, see tests/input_test_runner_test.py 