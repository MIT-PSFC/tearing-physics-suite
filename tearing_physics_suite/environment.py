# Global environment variables for the tearing physics suite
# Update these based on your install location

engaging = True
poetry_py = True

if engaging:
    home_dir = "/home/stubenj9/tearing-physics-suite"
    '''
    cd /home/stubenj9/tearing-physics-suite
    '''

    if poetry_py: # SET UP POETRY ENVIRONMENT
        ''' #Put this in your bash script for job, or type into terminal
        source /home/stubenj9/.cache/pypoetry/virtualenvs/export_poetry-virtualenvs-in-project=1-VSsruy4k-py3.11/bin/activate
        cd /home/stubenj9/DeltaPrimeValidationScans
        poetry env use /home/stubenj9/.cache/pypoetry/virtualenvs/export_poetry-virtualenvs-in-project=1-VSsruy4k-py3.11/bin/python 
        cd /home/stubenj9/tearing-physics-suite

        '''

########################################################################################################
    #PEST3 BUILD OPTIONS:
########################################################################################################
    ''' #PEST3 BUILD MODULES 
    module load hdf5/1.14.3 
    module load netcdf-c/4.9.2
    module load netcdf-fortran/4.6.1
    module load netlib-lapack/3.10.1
    module load cmake/3.27.9
    module load intel/2024.2.1 
    

    #To build PEST3, load the above, then make directory build, then in that directory run 'cmake ..'
    # Then run 'make', then 
    cd /home/stubenj9/tearing-physics-suite/submodules/PEST3/build
    cmake .. 
    make DESTDIR=/home/stubenj9/tearing-physics-suite/submodules/PEST3/build install
    # If you do all of this, you can still run pest3x using the module set used in #GPEC module initialization. I ran the three test cases with #PEST3 BUILD modules and GPEC modules and they gave the same answer.
    '''

########################################################################################################
    #GPEC BUILD OPTIONS:
########################################################################################################
    # BRANCH CHOICE:
    # Pull: 

    ''' #MODULES FOR BUILDING GPEC + RUNNING GPEC & PEST3  
    module use /orcd/nese/psfc/001/software/spack/2023-07-01-physics-rpp/spack/share/spack/modules-test/linux-rocky8-x86_64 
    module load hdf5/1.12.2-gcc-12.2.0-module-v654jvm
    module load netlib-lapack/3.10.1-gcc-12.2.0-module-pj6b3k7 
    module load netcdf-c/4.9.0-gcc-12.2.0-module-qvvpxf6
    module load netcdf-fortran/4.6.0-gcc-12.2.0-module-7dvzbuu
    module load intel-oneapi-mkl/2023.1.0-intel-oneapi-mpi-2021.9.0-gcc-12.2.0-module-seow5nc
    module load cmake/3.24.3-gcc-12.2.0-module-7kayese
    
    '''

    ''' #GPEC options: add the following to install/DEFAULTS.inc line 457
    NETCDF_EXTRA_LIBS += -lm -lzstd -Wl,-rpath,/orcd/nese/psfc/001/jinchen/petsc/petsc20230612/mit-intel-impi-st/lib -L/orcd/nese/psfc/001/jinchen/petsc/petsc20230612/mit-intel-impi-st/lib -lhdf5hl_fortran -lhdf5_fortran -lhdf5_hl -lhdf5 -Wl,-rpath,/orcd/nese/psfc/001/jinchen/petsc/petsc20230612/mit-intel-impi-st/lib -L/orcd/nese/psfc/001/jinchen/petsc/petsc20230612/mit-intel-impi-st/lib -lz -lquadmath
    Also swap lines 490, 491 to 
    OMPFLAG ?= -fopenmp
    RECURSFLAG ?= -frecursive
    '''