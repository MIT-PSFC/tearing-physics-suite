# Python functions to write input files for RDCON and STRIDE

import os
import shutil
import xarray as xr

# Called from within write_rdcon_stride_inputs
def write_equil_in(working_dir,eq_filename,write_equil_filename='/equil.in',
        #EQUIL_CONTROL
        eq_type="""'efit'""",     #Type of the input 2D equilibrium file. Accepts efit, chease, fluxgrid, transp, jsolver, lar, sol, etc.
        jac_type="""'hamada'""",  # Working coordinate system for all DCON and GPEC calculations. Overrides individual powers. Accepts hamada, pest, boozer, equal_arc
        power_bp=0,         #del.B ~ B_p**power_bp * B**power_b / R**power_r
        power_b=0,          #del.B ~ B_p**power_bp * B**power_b / R**power_r
        power_r=0,          #del.B ~ B_p**power_bp * B**power_b / R**power_r
        grid_type="""'ldp'""",    #Radial grid packing of equilibrium quantities. Accepts rho, ldp, pow1, pow2, or original. ldp packs points near the core and edge. pow* packs near the edge.
        psilow=1e-4,        #Minimum value of psi, normalized from 0 to 1
        psihigh=0.999,      #Maximum value of psi, normalized from 0 to 1
        mpsi=257,           #Number of radial grid intervals for equilibrium quantities. Large values (~800) can introduce numerical instabilities.
        mtheta=257,         #Number of equally spaced poloidal grid intervals for all splines. Large values (~800) can introduce numerical instabilities.
        nstepd=None,        #Deprecated, leave as none.
        etol=1e-10,
        newq0=0,            #Grad-Shafranov solution invariant adjustment of the q profile to give the specified value of q at the axis. Default 0 uses input file value.
        use_classic_splines='f', # Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions
        input_only='f',      #Generate information about the input and then quit with no further calculation
        #EQUIL_OUTPUT
        gse_flag='t',       #Produces diagnostic output for accuracy of solution to Grad-Shafranov equation
        out_eq_1d='f',      #Ascii output of 1D equilibrium file data
        bin_eq_1d='f',      #Binary output of 1D equilibrium file data
        out_eq_2d='f',      #Ascii output of 2D equilibrium file data
        bin_eq_2d='f',      #Binary output of 2D equilibrium file data (set true for GPEC)
        out_2d='f',         #Ascii output of processed 2D data
        bin_2d='f',         #Binary output of processed 2D data
        dump_flag='f',      #Binary dump of basic equilibrium data and 2D rzphi spline
        a_wall=21           #Controls ideal conformal shell distance. See vac.in description below.
        ):

    f = open(working_dir+write_equil_filename, 'w')

    f.write('&EQUIL_CONTROL'+'\n')
    f.write('    eq_type='+eq_type +'\n') #Type of the input 2D equilibrium file. Accepts efit, chease, fluxgrid, transp, jsolver, lar, sol, etc.
    f.write('    eq_filename='+"""'"""+eq_filename+"""'"""'\n') #Path to input file

    f.write('    jac_type='+jac_type +'\n') # Working coordinate system for all DCON and GPEC calculations. Overrides individual powers. Accepts hamada, pest, boozer, equal_arc
    f.write('    power_bp='+str(power_bp)+'\n') #del.B ~ B_p**power_bp * B**power_b / R**power_r
    f.write('    power_b='+str(power_b) +'\n') #del.B ~ B_p**power_bp * B**power_b / R**power_r
    f.write('    power_r='+str(power_r) +'\n') #del.B ~ B_p**power_bp * B**power_b / R**power_r

    f.write('    grid_type='+grid_type +'\n') #Radial grid packing of equilibrium quantities. Accepts rho, ldp, pow1, pow2, or original. ldp packs points near the core and edge. pow* packs near the edge.
    f.write('    psilow='+str(psilow) +'\n') #Minimum value of psi, normalized from 0 to 1
    f.write('    psihigh='+str(psihigh)   +'\n') #Maximum value of psi, normalized from 0 to 1
    f.write('    mpsi='+str(mpsi)  +'\n') #Number of radial grid intervals for equilibrium quantities
    f.write('    mtheta='+str(mtheta)+'\n') #Number of equally spaced poloidal grid intervals for all splines
    if not (nstepd is None):
        f.write('    nstepd='+str(nstepd)+'\n')
    f.write('    etol='+str(etol)+'\n')
    f.write('    newq0='+str(newq0)   +'\n') #Grad-Shafranov solution invariant adjustment of the q profile to give the specified value of q at the axis. Default 0 uses input file value.
    f.write('    use_classic_splines ='+use_classic_splines +'\n') # Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions

    f.write('    input_only='+input_only+'\n') #Generate information about the input and then quit with no further calculation
    f.write('/'+'\n')
    f.write('&EQUIL_OUTPUT'+'\n')
    f.write('    gse_flag='+gse_flag+'\n') #Produces diagnostic output for accuracy of solution to Grad-Shafranov equation
    f.write('    out_eq_1d='+out_eq_1d +'\n') #Ascii output of 1D equilibrium file data
    f.write('    bin_eq_1d='+bin_eq_1d +'\n') #Binary output of 1D equilibrium file data
    f.write('    out_eq_2d='+out_eq_2d +'\n') #Ascii output of 2D equilibrium file data
    f.write('    bin_eq_2d='+bin_eq_2d +'\n') #Binary output of 2D equilibrium file data (set true for GPEC)
    f.write('    out_2d='+out_2d  +'\n') #Ascii output of processed 2D data
    f.write('    bin_2d='+bin_2d  +'\n') #Binary output of processed 2D data
    f.write('    dump_flag='+dump_flag +'\n') #Binary dump of basic equilibrium data and 2D rzphi spline
    f.write('/'+'\n')

    f.close()

    #writing vac.in: Glasser didn't suggest modifying this in the DCON readme, outside of a (here a_wall), which is the distance of the conformal ideal wall
    #from the plasma in units of minor radius. a>20 is infinite vacuum. a=0 is close fitting ideal wall. small a>0 is ill-advised as the wall can clip the plasma.
    f = open(working_dir+'/vac.in', 'w')
    f.write('&MODES\n')
    f.write('   mth = 480\n')
    f.write('   xiin(1:9) = 0 0 0 0 0 0 0 1 0\n')
    f.write('   lsymz = .TRUE.\n')
    f.write('   leqarcw = 1\n')
    f.write('   lzio = 0\n')
    f.write('   lgato = 0\n')
    f.write('   lrgato = 0\n')
    f.write('/\n')
    f.write('&DEBUGS\n')
    f.write('   checkd = .FALSE.\n')
    f.write('   check1 = .FALSE.\n')
    f.write('   check2 = .FALSE.\n')
    f.write('   checke = .FALSE.\n')
    f.write('   checks = .FALSE.\n')
    f.write('   wall = .FALSE.\n')
    f.write('   lkplt = 0\n')
    f.write('   verbose_timer_output = f\n')
    f.write('/\n')
    f.write('&VACDAT\n')
    f.write('   ishape = 6\n')
    f.write('   aw = 0.05\n')
    f.write('   bw = 1.5\n')
    f.write('   cw = 0\n')
    f.write('   dw = 0.5\n')
    f.write('   tw = 0.05\n')
    f.write('   nsing = 500\n')
    f.write('   epsq = 1e-05\n')
    f.write('   noutv = 37\n')
    f.write('   idgt = 6\n')
    f.write('   idot = 0\n')
    f.write('   idsk = 0\n')
    f.write('   delg = 15.01\n')
    f.write('   delfac = 0.001\n')
    f.write('   cn0 = 1\n')
    f.write('/\n')
    f.write('&SHAPE\n')
    f.write('   ipshp = 0\n')
    f.write('   xpl = 100\n')
    f.write('   apl = 1\n')
    f.write('   a = '+str(a_wall)+'\n')
    f.write('   b = 170\n')
    f.write('   bpl = 1\n')
    f.write('   dpl = 0\n')
    f.write('   r = 1\n')
    f.write('   abulg = 0.932\n')
    f.write('   bbulg = 17.0\n')
    f.write('   tbulg = 0.02\n')
    f.write('   qain = 2.5\n')
    f.write('/\n')
    f.write('&DIAGNS\n')
    f.write('   lkdis = .FALSE.\n')
    f.write('   ieig = 0\n')
    f.write('   iloop = 0\n')
    f.write('   lpsub = 1\n')
    f.write('   nloop = 128\n')
    f.write('   nloopr = 0\n')
    f.write('   nphil = 3\n')
    f.write('   nphse = 1\n')
    f.write('   xofsl = 0\n')
    f.write('   ntloop = 32\n')
    f.write('   aloop = 0.01\n')
    f.write('   bloop = 1.6\n')
    f.write('   dloop = 0.5\n')
    f.write('   rloop = 1.0\n')
    f.write('   deloop = 0.001\n')
    f.write('   mx = 21\n')
    f.write('   mz = 21\n')
    f.write('   nph = 0\n')
    f.write('   nxlpin = 6\n')
    f.write('   nzlpin = 11\n')
    f.write('   epslp = 0.02\n')
    f.write('   xlpmin = 0.7\n')
    f.write('   xlpmax = 2.7\n')
    f.write('   zlpmin = -1.5\n')
    f.write('   zlpmax = 1.5\n')
    f.write('   linterior = 2\n')
    f.write('/\n')
    f.write('&SPRK\n')
    f.write('   nminus = 0\n')
    f.write('   nplus = 0\n')
    f.write('   mphi = 16\n')
    f.write('   lwrt11 = 0\n')
    f.write('   civ = 0.0\n')
    f.write('   sp2sgn1 = 1\n')
    f.write('   sp2sgn2 = 1\n')
    f.write('   sp2sgn3 = 1\n')
    f.write('   sp2sgn4 = 1\n')
    f.write('   sp2sgn5 = 1\n')
    f.write('   sp3sgn1 = -1\n')
    f.write('   sp3sgn2 = -1\n')
    f.write('   sp3sgn3 = -1\n')
    f.write('   sp3sgn4 = 1\n')
    f.write('   sp3sgn5 = 1\n')
    f.write('   lff = 0\n')
    f.write('   ff = 1.6\n')
    f.write('   fv = 1.6 1.6 1.6 1.6 1.6 1.0 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6\n')
    f.write('/\n')
    f.close()

    #Turn all inputs into a dictionary to return
    return_dict = {
        'eq_filename': os.path.basename(eq_filename),
        'eq_type': eq_type,
        'jac_type': jac_type,
        'power_bp': power_bp,
        'power_b': power_b,
        'power_r': power_r,
        'grid_type': grid_type,
        'psilow': psilow,
        'psihigh': psihigh,
        'mpsi': mpsi,
        'mtheta': mtheta,
        'nstepd': nstepd,
        'etol': etol,
        'newq0': newq0,
        'use_classic_splines': use_classic_splines,
        'input_only': input_only,
        'gse_flag': gse_flag,
        'out_eq_1d': out_eq_1d,
        'bin_eq_1d': bin_eq_1d,
        'out_eq_2d': out_eq_2d,
        'bin_eq_2d': bin_eq_2d,
        'out_2d': out_2d,
        'bin_2d': bin_2d,
        'dump_flag': dump_flag,
        'a_wall': a_wall
    }

    return return_dict

def write_rdcon_stride_inputs(working_dir,eq_filename,write_equil_filename='/equil.in',write_rdcon_filename='/rdcon.in',write_stride_filename='/stride.in',run_stride=True, run_rdcon=True, fresh_start=True,
            debug=True,
            ##GAL_INPUT
            nx=256,                 # The number of elements in each interval between two singular surfaces. Must be even!
            pfac=0.001,             # Packing ratio near the singular surface
            gal_tol=1e-10,          # Tolerance of lsode integration
            dx1dx2_flag='t',        # Flag to include the special dx1 and dx2 treatments for resonant and extension element
            dx0=5e-4,               # The distance to the singular surface to truncate the lsode integration in resonant element
            dx1=1.e-3,              # The size of resonant element
            dx2=1.e-3,              # The size of extension element
            cutoff=10,              # The number of elements include the large solution as the driving term
            solver="""'LU'""",      # LU factorization of solving Galerkinn matrix (cholesky broken for small nx, run RDCON_solver_scan to confirm this)
            nq=6,                   # The number of Gaussian points in each Galerkin element
            ##GAL_OUTPUT
            interp_np=3,            # The number of interpration points for outputting Galerkin solution
            restore_uh='t',         # Include the Hermite solution in Galerkin soluitn
            restore_us='t',         # Include the small solution in Galerkin solution
            restore_ul='t',         # Include the larger solution in Galerkin solution
            bin_delmatch='f',       # Output solution for rmatch
            out_galsol='f',         # Output Galerkin solution in ascii files
            bin_galsol='f',         # Output Galerkin solution in binary files
            b_flag='f',             # Output the perturbation of radial b field
            rpec_flag='f',          # Resistive perturbed equilibrium computation
            bin_coilsol='f',        # Output binary files for every unit-m driven solution
            ##RDCON_CONTROL
            bal_flag='f',           # Ideal MHD ballooning criterion for short wavelengths
            mat_flag='f',           # Construct coefficient matrices for diagnostic purposes
            ode_flag='t',           # RDCON control only (forced true for STRIDE) - Integrate ODEs for determining stability of internal long-wavelength mode.
            vac_flag='t',           # Compute plasma, vacuum, and total energies for free-boundary modes
            gal_flag='t',           # Compute outer regime using resonant Galerkin method
            dump_MRE_data=False,    # Dump MRE data, exits before gal_flag can run

            sas_flag='t',           # Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
            reform_eq_with_psilim='t', # Reform the equilibrium with the specified psi limits
            dmlim=0.2,              # See sas_flag
            sing_start=0,           # Start integration at the sing_start'th rational from the axis (psilow)
            qlow=0.0,               # Lower bound of q for the Galerkin method
            qhigh=1e3,              # Upper bound of q for the Galerkin method

            nn=1,                   # Toroidal mode number
            delta_mlow=8,           # Expands lower bound of Fourier harmonics
            delta_mhigh=8,          # Expands upper bound of  Fourier harmonics
            delta_mband=0,          # Integration keeps only this wide a band of solutions along the diagonal in m,m'
            mthvac=960,            # Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
            thmax0=1,               # Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

            tol_nr=1e-10,           # Relative tolerance of dynamic integration steps away from rationals
            tol_r=1e-10,            # Relative tolerance of dynamic integration steps near rationals
            crossover=1e-2,         # Fractional distance from rational q at which tolerance is switched to tol_r
            singfac_min=1e-4,       # Fractional distance from rational q at which ideal jump condition is enforced
            ucrit=1e3,              # Maximum fraction of solutions allowed before re-normalized

            cyl_flag='f',           # Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).

            sing1_flag='t',         # Special power series treatment
            gal_xmin_flag='f',      # Special power series treatment for Galerkin method
            sing_order=8,          # The highest order of power series to be retained
            sing_order_ceiling='f', # Auto detect the minium order to be retained in power series...

            regrid_flag='f',        # Redo the grid generation for galerkin method
            Zeff=1.52,              # Plasma Z effective

            #RDCON_OUTPUT
            crit_break='t',         # Color of the crit curve changes when crossing a singular surface

            ahb_flag='f',           # Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
            msol_ahb=1,             # Number of eigenfunctions output by ahb_flag=t ?
            mthsurf0=1,             # Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag=t

            bin_euler='f',          # Output M psi-by-M euler-lagrange solutions to binary file euler.bin
            euler_stride=1,         # Output only every euler_stride'th psi step to binary file

            out_bal1='f',           # Ascii output for bal_flag poloidal functions
            bin_bal1='f',           # Binary output for bal_flag poloidal functions
            out_bal2='f',           # Ascii output for bal_flag functions
            bin_bal2='f',           # Binary output for bal_flag functions
            out_ahg2msc='f',        # If true, uses old vacuum.io print-to-file logic
            MRE_flag='t',           # If true, outputs modified rutherford equation data 
            geom_flag='t',          # If true, outputs surface integral information for the equilibrium

            #UA_DIAGNOSE_LIST
            flag='f',
            phase='t',
            eq_type="""'efit'""", #Type of the input 2D equilibrium file. Accepts efit, chease, fluxgrid, transp, jsolver, lar, sol, etc.
            
            #STRIDE_CONTROL
            use_classic_splines='f', # Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions
            use_notaknot_splines='f', # Use not-a-knot boundary conditions for splines with extrapolation boundary conditions

            #STRIDE_PARAMS
            nThreads=32,                         # Number of threads used to calculate intervals in parallel
            fourfit_metric_parallel='f',         # Compute equilibrium metric tensor components in parallel
            vac_parallel='t',                    # Doubles the number of main level threads, creating more threads than processors

            nIntervalsTot=33,                    # Number of radial intervals calculated in separately and recombined at end
            grid_packing="""'singularities'""",  # Choose from "singularities" and "naive"
            axis_mid_pt_skew=12.0,               # Skews the distribution of intervals between psilow and psihigh
            asymp_at_sing='t',                   # Use asymptotic expansions at singular surfaces
            kill_big_soln_for_ideal_dW='f',      # Explicitly remove the big solution

            calc_delta_prime='t',                # When false, sing_order=2 is better for speed, (with less accuracy).
            calc_dp_with_vac='t',                # When vac_flag='t', this will use vacuum edge conditions to calculate delta prime.
            big_soln_err_tol=1e-7,               # Threshold error for correct solution to the delta prime bvp.

            integrate_riccati='f',               # Use Riccati integration for dW. Cannot calculate delta_prime.
            riccati_bounce='t',                  # Toggle integration mode to control growth of matrix
            riccati_match_hamiltonian_evals='f', #
            verbose_riccati_output='t',          # Creates peig and Psize output files
            ric_dt=1e-8,                         # Initial step size of Riccati integration
            ric_tol=1e-6,                        # Relative tolerance of Riccati integration

            verbose_performance_output='t',      # Print detailed timing information to terminal
            sing_start_str=0,                    # Start integration at the sing_start'th rational from the axis (psilow). Different from rdcon sing_start since stride finds q_low searching from outside in

            #Extra 
            a_wall=21,                           #Controls ideal conformal shell distance. See vac.in description below.
            verbose = False,                     # Print verbose output to terminal

            #Scan logic: Default values will not affect above inputs.
            set_delta_mlow_to_delta_mhigh=False, # If true, delta_mlow is set to delta_mhigh.
            set_dx1dx2_dx0_mult = 0.0,           # If set, dx1, dx2 = dx0 multiplied by this value.
            set_singfac_min_to_dx = False,       # If true, set singfac_min to dx0. This is useful for scans where you want to see the effect of changing asymptotic matching point on Delta Prime.
            set_int_tolerances_equal = False,    # If true, set tol_nr, gal_tol equal to tol_r.
            **kwargs):

    if verbose: print('printing eq_type=',eq_type)
    if not run_rdcon and not run_stride:
        print("Warning: Neither run_rdcon nor run_stride is set to True. Only equil.in file will be written.")
    if fresh_start:
        # If fresh_start is True, remove existing files in the working directory
        if os.path.exists(working_dir+write_equil_filename):
            os.remove(working_dir+write_equil_filename)
        if os.path.exists(working_dir+write_rdcon_filename):
            os.remove(working_dir+write_rdcon_filename)
        if os.path.exists(working_dir+write_stride_filename):
            os.remove(working_dir+write_stride_filename)
        if os.path.exists(working_dir+'/vac.in'):
            os.remove(working_dir+'/vac.in')

    if a_wall==0:
        vac_flag='f' 
    if vac_flag=='f':
        calc_dp_with_vac = 'f' # If vac_flag is false, then calc_dp_with_vac must be false
        a_wall = 0.0 # If vac_flag is false, then a_wall must be zero. This'll automatically set a_wall_pest to 0 too.
    if nx % 2 != 0: 
        if verbose: print("Warning: nx is not even, increasing by 1 to make it even.")
        nx += 1

    if set_delta_mlow_to_delta_mhigh:
        delta_mlow = delta_mhigh
    if set_dx1dx2_dx0_mult > 0.0:
        dx1 = dx0 * set_dx1dx2_dx0_mult
        dx2 = dx0 * set_dx1dx2_dx0_mult
    if set_singfac_min_to_dx:
        singfac_min = dx0
    if set_int_tolerances_equal:
        tol_nr = tol_r
        gal_tol = tol_r

    #Turn all inputs to write_rdcon_stride_inputs into a dictionary to return
    return_dict = {
        'run_stride': run_stride,
        'run_rdcon': run_rdcon,
        'nx': nx,
        'pfac': pfac,
        'gal_tol': gal_tol,
        'dx1dx2_flag': dx1dx2_flag,
        'dx0': dx0,
        'dx1': dx1,
        'dx2': dx2,
        'cutoff': cutoff,
        'solver': solver,
        'nq': nq,
        'interp_np': interp_np,
        'restore_uh': restore_uh,
        'restore_us': restore_us,
        'restore_ul': restore_ul,
        'bin_delmatch': bin_delmatch,
        'out_galsol': out_galsol,
        'bin_galsol': bin_galsol,
        'b_flag': b_flag,
        'rpec_flag': rpec_flag,
        'bin_coilsol': bin_coilsol,
        'bal_flag': bal_flag,
        'mat_flag': mat_flag,
        'ode_flag': ode_flag,
        'vac_flag': vac_flag,
        'gal_flag': gal_flag,
        'dump_MRE_data': dump_MRE_data,
        'sas_flag': sas_flag,
        'reform_eq_with_psilim': reform_eq_with_psilim,
        'dmlim': dmlim,
        'sing_start': sing_start,
        'qlow': qlow,
        'qhigh': qhigh,
        'nn': nn,
        'delta_mlow': delta_mlow,
        'delta_mhigh': delta_mhigh,
        'delta_mband': delta_mband,
        'mthvac': mthvac,
        'thmax0': thmax0,
        'tol_nr': tol_nr,
        'tol_r': tol_r,
        'crossover': crossover,
        'singfac_min': singfac_min,
        'ucrit': ucrit,
        'cyl_flag': cyl_flag,
        'sing1_flag': sing1_flag,
        'gal_xmin_flag': gal_xmin_flag,
        'sing_order': sing_order,
        'sing_order_ceiling': sing_order_ceiling,
        'regrid_flag': regrid_flag,
        'Zeff': Zeff,
        'crit_break': crit_break,
        'ahb_flag': ahb_flag,
        'msol_ahb': msol_ahb,
        'mthsurf0': mthsurf0,
        'bin_euler': bin_euler,
        'euler_stride': euler_stride,
        'out_bal1': out_bal1,
        'bin_bal1': bin_bal1,
        'out_bal2': out_bal2,
        'bin_bal2': bin_bal2,
        'out_ahg2msc': out_ahg2msc,
        'MRE_flag': MRE_flag,
        'geom_flag': geom_flag,
        'flag': flag,
        'phase': phase,
        'eq_type': eq_type,
        'use_classic_splines': use_classic_splines,
        'use_notaknot_splines': use_notaknot_splines,
        'nThreads': nThreads,
        'fourfit_metric_parallel': fourfit_metric_parallel,
        'vac_parallel': vac_parallel,
        'nIntervalsTot': nIntervalsTot,
        'grid_packing': grid_packing,
        'axis_mid_pt_skew': axis_mid_pt_skew,
        'asymp_at_sing': asymp_at_sing,
        'kill_big_soln_for_ideal_dW': kill_big_soln_for_ideal_dW,
        'calc_delta_prime': calc_delta_prime,
        'calc_dp_with_vac': calc_dp_with_vac,
        'big_soln_err_tol': big_soln_err_tol,
        'integrate_riccati': integrate_riccati,
        'riccati_bounce': riccati_bounce,
        'riccati_match_hamiltonian_evals': riccati_match_hamiltonian_evals,
        'verbose_riccati_output': verbose_riccati_output,
        'ric_dt': ric_dt,
        'ric_tol': ric_tol,
        'verbose_performance_output': verbose_performance_output,
        'sing_start_str': sing_start_str,
        'verbose': verbose
    }
    
    equil_dict = write_equil_in(working_dir,eq_filename,write_equil_filename=write_equil_filename,eq_type=eq_type,a_wall=a_wall,**kwargs)
    #combine the dictionaries
    return_dict.update(equil_dict)
    
    if run_rdcon:
        f = open(working_dir+write_rdcon_filename, 'w')

        f.write('&GAL_INPUT'+'\n')
        f.write('    nx='+str(nx)+'\n')  #The number of elements in each interval between two singular surfaces
        f.write('    pfac='+str(pfac)+'\n') #Packing ratio near the singular surface
        f.write('    gal_tol='+str(gal_tol) +'\n') #Tolerance of lsode integration
        f.write('    dx1dx2_flag='+dx1dx2_flag+'\n')  #Flag to include the special dx1 and dx2 treatments for resonant and extension element
        f.write('    dx0='+str(dx0)+'\n') #The distance to the singular surface to truncate the lsode integration in resonant element
        f.write('    dx1='+str(dx1) +'\n') #The size of resonant element
        f.write('    dx2='+str(dx2) +'\n') #The size of extension element
        f.write('    cutoff='+str(cutoff)+'\n')  #The number of elements include the large solution as the driving term
        f.write('    solver='+solver+'\n') #LU factorization of solving Galerkinn matrix
        f.write('    nq='+str(nq)+'\n')  #The number of Gaussian points in each Galerkin element
        f.write('    gal_xmin_flag='+gal_xmin_flag   +'\n') #Flag for automatically setting width of resonant element based on power series convergence. Convergence is not well behaved in my experience, this can commonly error out. 

        f.write('/'+'\n')
        f.write('&GAL_OUTPUT'+'\n')
        f.write('    interp_np='+str(interp_np)+'\n') #The number of interpration points for outputting Galerkin solution
        f.write('    restore_uh='+restore_uh   +'\n') #Include the Hermite solution in Galerkin soluitn
        f.write('    restore_us='+restore_us   +'\n') #Include the small solution in Galerkin solution
        f.write('    restore_ul='+restore_ul   +'\n') #Include the larger solution in Galerkin solution
        f.write('    bin_delmatch='+bin_delmatch +'\n') #Output solution for rmatch
        f.write('    out_galsol='+out_galsol   +'\n') #Output Galerkin solution in ascii files
        f.write('    bin_galsol='+bin_galsol   +'\n') #Output Galerkin solution in binary files
        f.write('    b_flag='+b_flag   +'\n') #Output the perturbation of radial b field
        f.write('    bin_coilsol='+bin_coilsol +'\n') #Output binary files for every unit-m driven solution
        f.write('/'+'\n')
        f.write('&RDCON_CONTROL'+'\n')
        f.write('    bal_flag='+bal_flag +'\n') #Ideal MHD ballooning criterion for short wavelengths
        f.write('    mat_flag='+mat_flag +'\n') #Construct coefficient matrices for diagnostic purposes
        f.write('    ode_flag='+ode_flag +'\n') #Integrate ODEs for determining stability of internal long-wavelength mode (must be true for GPEC)
        f.write('    vac_flag='+vac_flag +'\n') #Compute plasma, vacuum, and total energies for free-boundary modes
        f.write('    gal_flag='+gal_flag +'\n') #Compute outer regime using resonant Galerkin method
        if not dump_MRE_data==False:
            f.write('    dump_MRE_data='+dump_MRE_data +'\n') #Dump MRE data, exits before gal_flag can run

        f.write('    sas_flag='+sas_flag +'\n') #Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
        f.write('    reform_eq_with_psilim='+reform_eq_with_psilim +'\n') #Reform the equilibrium with the specified psi limits
        f.write('    qlow='+str(qlow) +'\n') #Lower bound of q for the equilibrium
        f.write('    qhigh='+str(qhigh) +'\n') #Upper bound of q for the equilibrium
        f.write('    dmlim='+str(dmlim)  +'\n') #See sas_flag
        f.write('    sing_start='+str(sing_start)   +'\n') #Start integration at the sing_start'th rational from the axis (psilow)

        f.write('    nn='+str(nn)   +'\n') #Toroidal mode number
        f.write('    delta_mlow='+str(delta_mlow)   +'\n') #Expands lower bound of Fourier harmonics
        f.write('    delta_mhigh='+str(delta_mhigh)  +'\n') #Expands upper bound of  Fourier harmonics
        f.write('    delta_mband='+str(delta_mband)  +'\n') #Integration keeps only this wide a band of solutions along the diagonal in m,m'
        f.write('    mthvac='+str(mthvac) +'\n') #Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
        f.write('    thmax0='+str(thmax0)   +'\n') #Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

        f.write('    tol_nr='+str(tol_nr)+'\n') #Relative tolerance of dynamic integration steps away from rationals
        f.write('    tol_r='+str(tol_r) +'\n') #Relative tolerance of dynamic integration steps near rationals
        f.write('    crossover='+str(crossover) +'\n') #Fractional distance from rational q at which tolerance is switched to tol_r
        f.write('    singfac_min='+str(singfac_min)  +'\n') # Fractional distance from rational q at which ideal jump condition is enforced
        f.write('    ucrit='+str(ucrit)  +'\n') #Maximum fraction of solutions allowed before re-normalized

        f.write('    cyl_flag='+cyl_flag +'\n') #Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).

        if gal_xmin_flag=='t':
            sing1_flag='t' #This is necessary to use gal_xmin_flag
        f.write('    sing1_flag='+sing1_flag   +'\n') #Special power series treatment
        f.write('    sing_order='+str(sing_order)   +'\n') #The highest order of power series to be retained
        f.write('    sing_order_ceiling='+sing_order_ceiling +'\n') # Auto detect the minium order to be retained in power series

        f.write('    regrid_flag='+regrid_flag  +'\n') #Redo the grid generation for galerkin method
        f.write('    Zeff='+str(Zeff) +'\n') #Plasma Z effective
        f.write('/'+'\n')

        f.write('&RDCON_OUTPUT'+'\n')
        f.write('    crit_break='+crit_break  +'\n') #Color of the crit curve changes when crossing a singular surface

        f.write('    ahb_flag='+ahb_flag  +'\n') #Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
        f.write('    msol_ahb='+str(msol_ahb)   +'\n') #Number of eigenfunctions output by ahb_flag='t ?
        f.write('    mthsurf0='+str(mthsurf0)  +'\n') #Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag='t

        f.write('    bin_euler='+bin_euler  +'\n') #Output M psi-by-M euler-lagrange solutions to binary file euler.bin
        f.write('    euler_stride='+str(euler_stride)  +'\n') #Output only every euler_stride'th psi step to binary file

        f.write('    out_bal1='+out_bal1  +'\n') #Ascii output for bal_flag poloidal functions
        f.write('    bin_bal1='+bin_bal1  +'\n') #Binary output for bal_flag poloidal functions
        f.write('    out_bal2='+out_bal2  +'\n') #Ascii output for bal_flag functions
        f.write('    bin_bal2='+bin_bal2  +'\n') #Binary output for bal_flag functions
        if not (out_ahg2msc is None):
            f.write('    out_ahg2msc='+out_ahg2msc  +'\n')
        f.write('    MRE_flag='+MRE_flag  +'\n') #If true, outputs modified rutherford equation data
        f.write('    geom_flag='+geom_flag  +'\n') #If true, outputs surface integral information for the equilibrium
        f.write('/'+'\n')

        f.write('&UA_DIAGNOSE_LIST'+'\n')
        f.write('    uad%flag='+flag  +'\n')
        f.write('    uad%phase='+phase  +'\n')
        f.write('/'+'\n')

        f.close()

    if run_stride:
        if sing1_flag=='t':
            #print a warning if sing1_flag is set, since this is not supported in STRIDE
            print("Warning: sing1_flag is set to true, but STRIDE does not support this option. This may introduce discrepancies of O(1e-3) in Delta_prime (run sing1_flag_scan to see effect).")
        if sing_order_ceiling=='t':
            print("Warning: sing_order_ceiling is set to true, but STRIDE does not support this option. Consider manually setting sing_order to a value that works for your case.")
        if gal_xmin_flag=='t':
            print("Warning: gal_xmin_flag is set to true, so RDCON will automatically set its asymptotic matching distances, while STRIDE does not support this option.")   

        f = open(working_dir+write_stride_filename, 'w')

        f.write('&stride_control'+'\n')
        f.write('    bal_flag='+bal_flag +'\n') #Ideal MHD ballooning criterion for short wavelengths
        f.write('    mat_flag='+mat_flag +'\n') #Construct coefficient matrices for diagnostic purposes
        f.write('    ode_flag=t' +'\n') #Integrate ODEs for determining stability of internal long-wavelength mode (must be true for GPEC)
        f.write('    vac_flag='+vac_flag +'\n') #Compute plasma, vacuum, and total energies for free-boundary modes
        f.write('    mer_flag='+gal_flag +'\n') #Evaluate the Mercier criterian

        f.write('    sas_flag='+sas_flag +'\n') #Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
        f.write('    dmlim='+str(dmlim)  +'\n') #See sas_flag
        f.write('    qlow='+str(qlow)  +'\n') #Integration initiated at q determined by minimum of qlow and q0 from equil
        f.write('    qhigh='+str(qhigh) +'\n') #Integration terminated at q limit determined by minimum of qhigh and qa from equil
        f.write('    sing_start='+str(sing_start_str)   +'\n') #Start integration at the sing_start'th rational from the axis (psilow)

        f.write('    nn='+str(nn)   +'\n') #Toroidal mode number
        f.write('    delta_mlow='+str(delta_mlow)   +'\n') #Expands lower bound of Fourier harmonics
        f.write('    delta_mhigh='+str(delta_mhigh)  +'\n') #Expands upper bound of  Fourier harmonics
        f.write('    delta_mband='+str(delta_mband)  +'\n') #Integration keeps only this wide a band of solutions along the diagonal in m,m'
        f.write('    mthvac='+str(mthvac) +'\n') #Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
        f.write('    thmax0='+str(thmax0)   +'\n') #Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

        f.write('    tol_nr='+str(tol_nr)+'\n') #Relative tolerance of dynamic integration steps away from rationals
        f.write('    tol_r='+str(tol_r) +'\n') #Relative tolerance of dynamic integration steps near rationals
        f.write('    crossover='+str(crossover) +'\n') #Fractional distance from rational q at which tolerance is switched to tol_r
        f.write('    singfac_min='+str(singfac_min)  +'\n') # Fractional distance from rational q at which ideal jump condition is enforced
        f.write('    ucrit='+str(ucrit)  +'\n') #Maximum fraction of solutions allowed before re-normalized
        f.write('    sing_order='+str(sing_order)   +'\n') #The highest order of power series to be retained

        f.write('    cyl_flag='+cyl_flag +'\n') #Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).

        f.write('    use_classic_splines='+use_classic_splines +'\n') # Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions
        f.write('    use_notaknot_splines='+use_notaknot_splines +'\n') # Use not-a-knot boundary instead of either "extrap" spline extrapolation
        f.write('/'+'\n')

        f.write('&stride_output'+'\n')
        f.write('    crit_break='+crit_break  +'\n') #Color of the crit curve changes when crossing a singular surface

        f.write('    ahb_flag='+ahb_flag  +'\n') #Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
        f.write('    msol_ahb='+str(msol_ahb)   +'\n') #Number of eigenfunctions output by ahb_flag='t ?
        f.write('    mthsurf0='+str(mthsurf0)  +'\n') #Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag='t

        f.write('    bin_euler='+bin_euler  +'\n') #Output M psi-by-M euler-lagrange solutions to binary file euler.bin
        f.write('    euler_stride='+str(euler_stride)  +'\n') #Output only every euler_stride'th psi step to binary file

        f.write('    out_bal1='+out_bal1  +'\n') #Ascii output for bal_flag poloidal functions
        f.write('    bin_bal1='+bin_bal1  +'\n') #Binary output for bal_flag poloidal functions
        f.write('    out_bal2='+out_bal2  +'\n') #Ascii output for bal_flag functions
        f.write('    bin_bal2='+bin_bal2  +'\n') #Binary output for bal_flag functions

        f.write("""    netcdf_out=t""" +'\n') #Replicate ascii stride.out and delta_prime.out information in a netcdf file
        if not (out_ahg2msc is None):
            f.write('    out_ahg2msc='+out_ahg2msc  +'\n') 
        f.write('/'+'\n')

        f.write('&stride_params'+'\n')
        f.write('    nThreads='+str(nThreads) +'\n')
        f.write('    fourfit_metric_parallel='+fourfit_metric_parallel +'\n')
        f.write('    vac_parallel='+vac_parallel +'\n')
        f.write('    nIntervalsTot='+str(nIntervalsTot) +'\n')
        f.write('    grid_packing='+grid_packing +'\n')
        f.write('    axis_mid_pt_skew='+str(axis_mid_pt_skew) +'\n')
        f.write('    asymp_at_sing='+asymp_at_sing +'\n')
        f.write('    kill_big_soln_for_ideal_dW='+kill_big_soln_for_ideal_dW +'\n')
        f.write('    calc_delta_prime='+calc_delta_prime +'\n')
        f.write('    calc_dp_with_vac='+calc_dp_with_vac +'\n')
        f.write('    big_soln_err_tol='+str(big_soln_err_tol) +'\n')
        f.write('    integrate_riccati='+integrate_riccati +'\n')
        f.write('    riccati_bounce='+riccati_bounce +'\n')
        f.write('    riccati_match_hamiltonian_evals='+riccati_match_hamiltonian_evals +'\n')
        f.write('    verbose_riccati_output='+verbose_riccati_output +'\n')
        f.write('    ric_dt='+str(ric_dt) +'\n')
        f.write('    ric_tol='+str(ric_tol) +'\n')

        f.write('    verbose_performance_output='+verbose_performance_output +'\n')
        f.write('/'+'\n')


        f.close()


    # {a_wall, delta_mlow, delta_mhigh, delta_mband, psilow, psihigh, mtheta, mpsi, nx} are useful to optionally pass to PEST3
    #if debug: print(return_dict)
    return return_dict