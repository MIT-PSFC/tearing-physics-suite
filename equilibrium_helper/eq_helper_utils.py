# Requires you to already have a TokaMaker package loaded

def get_all_profiles(mygs, psi_pad=1.E-3, npsi=100):
    r'''! Get all equilibrium profiles from TokaMaker object

    @param mygs TokaMaker object
    @param psi_pad Padding to add to psi array (to avoid issues at psi=0 and psi=1)
    @param npsi Number of psi points to sample
    '''
    psi,f,fp,p,pp = mygs.get_profiles(npsi=npsi,psi_pad=psi_pad)
    psi_q,qvals,ravgs,_,_,_ = mygs.get_q(npsi=npsi,psi_pad=psi_pad) # get flux averaged R and 1/R from equilibrium solution
    assert np.allclose(psi,psi_q), "Error: psi arrays from get_profiles and get_q do not match!"
    R_avg = ravgs[0] # R
    one_over_R_avg = ravgs[1] # 1/R
    mu0 = np.pi*4.E-7
    jtor = R_avg * pp + one_over_R_avg*(f*fp) / mu0 # Jtor = R*P' + FF' / (mu0*R)
    ffprime = f*fp
    return psi,f,fp,p,pp,R_avg,one_over_R_avg,jtor,ffprime,ravgs,qvals

def ffprime_from_jtor_pprime(jtor, pprime, R_avg, one_over_R_avg):
    r'''! Convert from J_toroidal to FF' using Grad-Shafranov equation

    @param jtor Toroidal current profile
    @param R_avg Flux averaged R, calculated by TokaMaker
    @param one_over_R_avg Flux averaged 1/R, calculated by TokaMaker
    @param pprime dP/dPsi profile
    '''
    ffprime = 2.0*(jtor -  R_avg * (-pprime)) * (mu0 / one_over_R_avg)
    return ffprime

def jtor_from_GS(ffprime, pprime, R_avg, one_over_R_avg):
    r'''! Convert from J_toroidal to FF' using Grad-Shafranov equation
    @param jtor Toroidal current profile
    @param R_avg Flux averaged R, calculated by TokaMaker
    @param one_over_R_avg Flux averaged 1/R, calculated by TokaMaker
    @param pprime dP/dPsi profile
    '''
    mu0 = np.pi*4.E-7
    jtor = ffprime * (one_over_R_avg / mu0) + R_avg * pprime
    return jtor

def extract_normalised_pp(mygs,numpts=257):
    # Extract pprofile once:
    psi,f,fp,p,pp = mygs.get_profiles(npsi=numpts)
    pax = p[0]
    pp_prof = {'type': 'linterp', 'x': psi, 'y': pp}
    pp_prof['y'] /= np.abs(pp_prof['y']).max() # Normalize profile (not required but convienient)
    if pp_prof['y'][0]<0.0:
        pp_prof['y'] *= -1.0 # Make sure pp is positive at axis
    return pp_prof

from scipy.stats import skewnorm
def gaussian(x, amp, center, width, offset, sk):
    skew = skewnorm.pdf(x, sk, center, width)
    skew = skew/max(skew)
    skew *= amp
    skew += offset
    return skew

from scipy.interpolate import CubicSpline
def get_psis_from_qs(qfinds,psi_q,qvals,debug=True):
    psis = []
    for q in qfinds:
        if q<qvals.min() or q>qvals.max():
            print("Warning: q="+str(q)+" is outside range of q values, skipping...")
            psis.append(np.nan)
        else:
            roots = CubicSpline(psi_q,qvals - q,extrapolate=False).roots()
            print(roots)
            psis.append(roots[-1])
            #func = lambda x: q_spline(x)-q
            #sol = root_scalar(func,bracket=[0.0,1.0],method='bisect')
            #psis.append(sol.root)
    return np.array(psis)

def easy_plot(mygs,filename,starting_dir,npsi=257,psi_pad=0.005,profiles_xr=None,**kwargs):
    if not (profiles_xr is None):
        easy_plot_kin(mygs,filename,starting_dir,profiles_xr,npsi=npsi,psi_pad=psi_pad,**kwargs)
        return
    # Plot it: 
    plt.close('all')
    #print("Final equilibrium for shape from: "+shape_eqdsk_name)
    #print("Final equilibrium for profiles from: "+profile_eqdsk_name)
    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]',coil_symmap=True)
    mygs.plot_psi(fig,ax,plasma_nlevels=5,vacuum_nlevels=5)
    mygs.plot_constraints(fig, ax)
    plt.savefig(str(starting_dir+"/MCHN_"+filename+".pdf"),dpi=300)
    psi,f,fp,p,pp,R_avg,one_over_R_avg,J_tor,ffprime,ravgs,qvals = get_all_profiles(mygs,psi_pad=psi_pad,npsi=npsi)
    plt.close('all')
    fig, ax = plt.subplots(4,1,sharex=True)
    # Plot F*F'
    ax[0].plot(psi,f*fp)#,label='TokaMaker')
    #ax[0].set_ylim(bottom=-10)
    ax[0].set_ylabel("FF'")
    ax[0].legend()
    # Plot P'
    ax[1].plot(psi,p/1e6)
    #ax[1].set_ylim(top=2.E6)
    ax[1].set_ylabel("P [MPa]")
    # Plot q
    ax[2].plot(psi,qvals)
    ax[2].set_ylabel("q")
    ax[2].axhline(1.0,ls=':',color='k',alpha=0.5)
    # Plot < J_phi >
    ax[3].plot(psi,J_tor/1e6)
    ax[3].set_ylabel(r"< $J_{\phi}$ > [MA/m$^2$]")
    # Add dotted line at J_tor = 0 
    ax[3].axhline(0.0,ls=':',color='k',alpha=0.5)
    _ = ax[-1].set_xlabel(r"$\psi_N$")
    plt.savefig(str(starting_dir+"/PROF_"+filename+".pdf"),dpi=300)


def res_func(profiles_xr):
    ### Keep up to date with res_func in tearing-physics-suite 
    ###
    if 'eta_fac' in profiles_xr:
        eta_fac = profiles_xr.eta_fac.values
    else:
        eta_fac = 1.0
    if 'Coulomb_logarithm' in profiles_xr:
        Coulomb_logarithm = profiles_xr.Coulomb_logarithm.values
    else:
        Coulomb_logarithm = None
    ###
    ###
    if Coulomb_logarithm is None:
        # Coulomb Logarithm using Wesson Tokamaks page 727:
        profiles_xr = profiles_xr.assign(
            lnLamb_ei = 15.2-0.5*np.log(profiles_xr['ne']/1e20)+np.log(profiles_xr['te'])   # Dimless, requires te in keV and ne in m^-3
        ) 
    else:
        profiles_xr = profiles_xr.assign(
            lnLamb_ei = Coulomb_logarithm+0.0*profiles_xr['psi_n_1d']                 # Dimless
        )
    # Resistivity in Ohm m from Wesson Tokamaks
    profiles_xr = profiles_xr.assign(
        eta_spitz = eta_fac*1.65*1e-9*profiles_xr['lnLamb_ei']*(profiles_xr['te']**(-3/2))  # Ohm m
    ) 
    return profiles_xr.eta_spitz.values


def easy_plot_kin(mygs,filename,starting_dir,profiles_xr,npsi=257,psi_pad=0.005,plot_input_jind=False):
    # Plot it: 
    plt.close('all')
    #print("Final equilibrium for shape from: "+shape_eqdsk_name)
    #print("Final equilibrium for profiles from: "+profile_eqdsk_name)
    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]',coil_symmap=True)
    mygs.plot_psi(fig,ax,plasma_nlevels=5,vacuum_nlevels=5)
    mygs.plot_constraints(fig, ax)
    plt.savefig(str(starting_dir+"/MCHN_"+filename+".pdf"),dpi=300)
    psi,f,fp,p,pp,R_avg,one_over_R_avg,J_tor,ffprime,ravgs,qvals = get_all_profiles(mygs,psi_pad=psi_pad,npsi=npsi)
    plt.close('all')
    fig, ax = plt.subplots(4,1,sharex=True)
    # Plot F*F'
    ax[0].plot(psi,f*fp)#,label='TokaMaker')
    #ax[0].set_ylim(bottom=-10)
    ax[0].set_ylabel("FF'")
    ax[0].legend()
    # Plot P'
    ax[1].plot(psi,p/1e6)
    #ax[1].set_ylim(top=2.E6)
    ax[1].set_ylabel("P [MPa]")
    # Plot q
    ax[2].plot(psi,qvals)
    ax[2].set_ylabel("q")
    ax[2].axhline(1.0,ls=':',color='k',alpha=0.5)
    # Plot < J_phi >
    if ('j_BS' in profiles_xr) or ('j_IND' in profiles_xr) or (plot_input_jind and ('input_jtor' in profiles_xr)):
        ax[3].plot(psi,J_tor/1e6,label='total')
        if 'j_BS' in profiles_xr:
            ax[3].plot(profiles_xr.psi_n_1d.values,profiles_xr.j_BS.values/1e6,label='bootstrap')
        if 'j_IND' in profiles_xr:
            ax[3].plot(profiles_xr.psi_n_1d.values,profiles_xr.j_IND.values/1e6,label='inductive')
        if plot_input_jind and 'input_jtor' in profiles_xr:
            ax[3].plot(profiles_xr.psi_n_1d.values,profiles_xr.input_jtor.values/1e6,label='input j_ind')
        # Make legend if any labels were added:
        ax[3].legend(fontsize=8)
    else:
        ax[3].plot(psi,J_tor/1e6)
    ax[3].set_ylabel(r"< $J_{\phi}$ > [MA/m$^2$]")
    # Add dotted line at J_tor = 0 
    ax[3].axhline(0.0,ls=':',color='k',alpha=0.5)
    _ = ax[-1].set_xlabel(r"$\psi_N$")
    plt.savefig(str(starting_dir+"/PROF_"+filename+".pdf"),dpi=300)
    plt.close('all')
    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(profiles_xr.psi_n_1d.values,profiles_xr.te.values)
    ax[0].set_ylabel("Te [keV]")
    ax[1].plot(profiles_xr.psi_n_1d.values,profiles_xr.ne.values)
    ax[1].set_ylabel(r"ne [m$^{-3}$]")
    # Label ion mass to 2 s.f.
    ion_mass = profiles_xr.ss_ion_mass.values
    ion_charge = profiles_xr.ss_ion_charge.values
    ax[2].plot(profiles_xr.psi_n_1d.values,profiles_xr.ss_ion_density.values)
    ax[2].set_ylabel(r"ni [m$^{-3}$]")
    # Annontate ax[2] with str(r'ion mass [amu] ='+"{:.2f}".format(ion_mass))
    ax[2].annotate(str(r'ion mass    = '+"{:.2f}".format(ion_mass)+" [amu]"),xy=(0.55,0.85),xycoords='axes fraction')
    ax[2].annotate(str(r'ion charge = '+"{:.2f}".format(ion_charge)+" [e]"),xy=(0.55,0.65),xycoords='axes fraction')
    #ax[2].legend()
    eta_spitz = res_func(profiles_xr)
    ax[3].plot(profiles_xr.psi_n_1d.values,eta_spitz)
    ax[3].set_ylabel(r"$\eta$ [Ohm m]")
    # Make ax[3] y-axis log scale:
    ax[3].set_yscale('log')
    _ = ax[-1].set_xlabel(r"$\psi_N$")
    plt.savefig(str(starting_dir+"/KPROF_"+filename+".pdf"),dpi=300)
    plt.close('all')


def simple_current_prof(alpha,gamma,
            num_points=100,
            fade_dist=0.05,
            pos_shift=0.02,
            use_tanh=False,
            tanh_offset=0.5):
    """
    Create a simple current profile function, using create_power_flux_fun.
    Positive shift and logarithmic fade to zero at edge combine to form a simple pedestal.
        Apply jtor_prof using mygs.set_profiles(ffp_prof=jtor_prof, pp_prof=pp_prof)
    """
    #Start
    jtor_prof = create_power_flux_fun(num_points,alpha,gamma)
    jtor_prof['y'] = jtor_prof['y']/jtor_prof['y'][0]
    #Positive shift
    if pos_shift != 0.0:
        jtor_prof['y'] += pos_shift
        jtor_prof['y'] = jtor_prof['y']/jtor_prof['y'][0]
        if not use_tanh:
            #Fade to zero at edge
            scaler_length = int(fade_dist*num_points)
            scalers = 1-np.logspace(0,-2,scaler_length)
            for i, s in enumerate(scalers):
                jtor_prof['y'][-i-1] *= s
        else:
            # Use tanh to get a smooth transition to zero at edge
            x = np.linspace(0,1,num_points)
            tanh_prof = 0.5*(1.0 - np.tanh((x - (1.0 - tanh_offset*fade_dist)) # Set center of tanh to be 0.5*fade_dist from edge
                                /(fade_dist/np.pi))) # Convert np.pi to width fade_dist
            jtor_prof['y'] *= tanh_prof
    jtor_prof['type'] = 'jphi-linterp'
    return jtor_prof

# Daniel Burgess Sep 15th
def solve_with_bootstrap_new(self,
                       ne,
                       Te,
                       ni,
                       Ti,
                       Zeff,
                       Ip_target,
                       inductive_jtor=None,
                       initial_ffp=None,
                       Zis=[1.],
                       R0=None,
                       Z0=None,
                       a0=None,
                       scale_jBS=None,
                       heightp=None,
                       inexp=None,
                       outexp=None,
                       max_iterations=6,
                       initialize_eq=True):
    '''! Self-consistently compute bootstrap contribution from H-mode profiles. 
    If l_i is set, match to a user specified internal inductance by scanning a 
    parametrized FF' profile, and iterate solution until all functions of Psi converge. 
    If inductive_jtor is set, solve for FF' using Grad-Shafranov equation and 
    iterate solution until all functions of Psi converge. 

    @note if using nis and Zis, dnis_dpsi must be specified in sauter_bootstrap() 
    as a list of impurity gradients over Psi. See 
    https://omfit.io/_modules/omfit_classes/utils_fusion.html for more 
    detailed documentation

    @note if initialize_eq=True, cubic polynomials will be fit to the core of all 
    kinetic profiles in order to flatten the pedestal. This will initialize the G-S 
    solution at an estimated L-mode pressure profile and using a generic current profile. 

    @param ne Electron density profile, sampled over psi_norm
    @param Te Electron temperature profile [eV], sampled over psi_norm
    @param ni Ion density profile, sampled over psi_norm
    @param Ti Ion temperature profile [eV], sampled over psi_norm
    @param Zeff Effective Z profile, sampled over psi_norm
    @param Zis List of impurity profile atomic numbers; currently set to 1. 
    @param nis NOT USED: list of impurity density profiles, currently ni = nis
    @param inductive_jtor Inductive toroidal current, sampled over psi_norm
    @param scale_jBS Factor by which to scale bootstrap current fraction
    @param heightp Height of parametrized FF' spike, dimensionless. Overrides Sauter 
    matching procedure
    @param inexp Inner core exponent of parametrized FF' profile, dimensionless. Overrides
    l_i matching procedure
    @param outexp Outer core exponent of parametrized FF' profile, dimensionless. Overrides
    l_i matching procedure
    @param max_iterations Maximum number of H-mode mygs.solve() iterations
    @param initialize_eq Initialize equilibrium solve with flattened pedestal. 
    '''
    from scipy.signal import find_peaks, peak_widths
    from scipy.optimize import curve_fit
    from scipy.optimize import root_scalar
    from scipy.stats import skewnorm

    try:
        from omfit_classes.utils_fusion import sauter_bootstrap
    except:
        raise ImportError('omfit_classes.utils_fusion not installed')

    # Turn off solver iteration printout
    # self.settings.pm = False
    # self.update_settings()
    
    def jtor_from_GS(ffprime, pprime, R_avg, one_over_R_avg):
        r'''! Convert from J_toroidal to FF' using Grad-Shafranov equation

        @param jtor Toroidal current profile
        @param R_avg Flux averaged R, calculated by TokaMaker
        @param one_over_R_avg Flux averaged 1/R, calculated by TokaMaker
        @param pprime dP/dPsi profile
        '''
        mu0 = np.pi*4.E-7
        jtor = ffprime * (one_over_R_avg / mu0) + R_avg * pprime
        return jtor
    def ffprime_from_jtor_pprime(jtor, pprime, R_avg, one_over_R_avg):
        r'''! Convert from J_toroidal to FF' using Grad-Shafranov equation

        @param jtor Toroidal current profile
        @param R_avg Flux averaged R, calculated by TokaMaker
        @param one_over_R_avg Flux averaged 1/R, calculated by TokaMaker
        @param pprime dP/dPsi profile
        '''
        ffprime = 2.0*(jtor -  R_avg * (-pprime)) * (mu0 / one_over_R_avg)
        return ffprime
    
    #### SET PROFILES AND SOLVE
    def iterate_ffp(self,ffprime_temp,pp_prof,psi_norm):
        ffp_prof = {
            'type': 'linterp',
            'x': psi_norm,
            'y': ffprime_temp / ffprime_temp[0]
        }
    #    pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
    #    ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge

        pp_prof['y'] = np.nan_to_num(pp_prof['y']) # Check for any nan's
        ffp_prof['y'] = np.nan_to_num(ffp_prof['y']) # Check for any nan's

        self.set_targets(Ip=Ip_target,pax=pressure[0])
        self.set_profiles(ffp_prof=ffp_prof,pp_prof=pp_prof)
        flag = self.solve()

        return flag

    def analyze_bootstrap_edge_spike(psi_N, j_bootstrap):
        r'''! Analyze bootstrap edge spike location, width, and height

        @param psi_N Normalized psi profile
        @param j_bootstrap Bootstrap current profile
        Returns:
        dict: Dictionary containing pedestal properties and spike model
        '''
        # Focus on the edge region (psi_N > 0.8)
        edge_mask = psi_N >= 0.7
        psi_edge = psi_N[edge_mask]
        j_edge = j_bootstrap[edge_mask]
        
     #   plt.plot(psi_edge,j_edge)
     #   plt.show()

        # Find peak in the edge region
        peaks, properties = find_peaks(j_edge, height=0.)#0.5*np.max(j_edge))
        
        if len(peaks) == 0:
            print("No clear peak found in the edge region")
            return None

        # Choose peak closest to psi_N = 1 if multiple peaks exist
        peak_idx = peaks[np.argmax(psi_edge[peaks])]
        peak_psi = psi_edge[peak_idx]
        peak_height = j_edge[peak_idx]

        # Calculate initial FWHM (full width at half maximum)
        widths = peak_widths(j_edge, [peak_idx], rel_height=0.5)
        left_idx, right_idx = int(widths[2][0]), int(widths[3][0])

        # Convert to psi_N coordinates
        fwhm = psi_edge[right_idx] - psi_edge[left_idx]

        # Fit a Gaussian to isolate the edge spike
        # Define a Gaussian function with offset (for background)
  #      def gaussian(x, amp, center, width, offset):
  #          return amp * np.exp(-(x - center)**2 / (2 * width**2)) + offset

        def gaussian(x, amp, center, width, offset, sk):
            skew = skewnorm.pdf(x, sk, center, width)
            skew = skew/max(skew)
            skew *= amp
            skew += offset
            return skew

        # Get good range for fitting - wider than the spike to capture baseline
        fit_range = max(0.15, 3*fwhm)  # At least 0.15 in psi_N or 3x FWHM
        fit_mask = (psi_N >= peak_psi - fit_range) & (psi_N <= min(1.0, peak_psi + fit_range))

        # Initial parameter guess
        # Estimate the background level from points away from the peak
        background_mask = (psi_N >= 0.7) & (psi_N <= 0.75)
        if np.sum(background_mask) > 5:
            background_level = np.median(j_bootstrap[background_mask])
        else:
            background_level = np.min(j_edge)

        sigma_init = fwhm/2.355  # Convert FWHM to sigma
        p0 = [peak_height - background_level, peak_psi, sigma_init, background_level, 1.0]

        # Perform the fit
        popt, pcov = curve_fit(gaussian, psi_N[fit_mask], j_bootstrap[fit_mask], p0=p0, 
                              maxfev=10000)

        amp, center, width, offset, sk = popt
        perr = np.sqrt(np.diag(pcov))

        def edge_spike_model(psi, scale_factor=1.0):
            """Returns the edge spike component scaled by a factor"""
            return scale_factor * amp * np.exp(-(psi - center)**2 / (2 * width**2))

        spike_only = gaussian(psi_N, amp, center, width, offset, sk)

        results = {
            'sigma': width,                 # Gaussian width (sigma)
            'background': offset,           # background level
            'gaussian_params': popt,        # Raw parameters [amp, center, width, offset]
            'spike_profile': spike_only,    # Array of spike component values
        }

        return results



    def objective_function(alpha, jtor_prof, spike_profile, my_psi_N, my_Ip_target):
        r'''! Compute difference between integrated a*j_tor+j_spike profile and Ip_target

        @param alpha Scaling factor to solve for
        @param jtor_prof Input j_inductive profile
        @param spike_profile Isolated j_bootstrap spike (a Gaussian), 0.0 everywhere else
        @param my_psi_N Local psi_N grid
        @param my_Ip_target Ip target
        '''
        Ip_computed = self.flux_integral(my_psi_N, (alpha*jtor_prof)+spike_profile)
        return Ip_computed - my_Ip_target
    
    def profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,scale_jBS,Zis,include_jBS=True):

        ### Get final remaining quantities for Sauter from TokaMaker
        psi_tmp,f,fp,_,_ = self.get_profiles(npsi=n_psi,psi_pad=1e-4)
        _,fc,r_avgs,_ = self.sauter_fc(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
        ft = 1 - fc # Trapped particle fraction on each flux surface
        eps = r_avgs[2] / r_avgs[0] # Inverse aspect ratio
        _,qvals,ravgs,_,_,_ = self.get_q(psi=psi_tmp,npsi=n_psi)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]
        
        pprime = np.gradient(pressure) / (np.gradient(psi_tmp) * (self.psi_bounds[1]-self.psi_bounds[0]))
        
        if include_jBS:
            ### Calculate flux derivatives for Sauter
            dn_e_dpsi = np.gradient(ne) / (np.gradient(psi_tmp) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dT_e_dpsi = np.gradient(Te) / (np.gradient(psi_tmp) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dn_i_dpsi = np.gradient(ni) / (np.gradient(psi_tmp) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dT_i_dpsi = np.gradient(Ti) / (np.gradient(psi_tmp) * (self.psi_bounds[1]-self.psi_bounds[0]))

            ### Solve for bootstrap current profile. See https://omfit.io/_modules/omfit_classes/utils_fusion.html for more detailed documentation 
            j_BS_neo = sauter_bootstrap(
                                    psi_N=psi_tmp,
                                    Te=Te,
                                    Ti=Ti,
                                    ne=ne,
                                    p=pressure,
                                    nis=[ni,],
                                    Zis=Zis,
                                    Zeff=Zeff,
                                    gEQDSKs=[None],
                                    R0=0., # not used
                                    device=None,
                                    psi_N_efit=None,
                                    psiraw=psi*(self.psi_bounds[1]-self.psi_bounds[0]) + self.psi_bounds[0],
                                    R=R_avg,
                                    eps=eps, 
                                    q=qvals,
                                    fT=ft,
                                    I_psi=f,
                                    nt=1,
                                    version='neo_2021',
                                    debug_plots=False,
                                    return_units=True,
                                    return_package=False,
                                    charge_number_to_use_in_ion_collisionality='Koh',
                                    charge_number_to_use_in_ion_lnLambda='Zavg',
                                    dT_e_dpsi=dT_e_dpsi,
                                    dT_i_dpsi=dT_i_dpsi,
                                    dn_e_dpsi=dn_e_dpsi,
                                    dnis_dpsi=[dn_i_dpsi,],
                                    )[0]
                
            j_BS = j_BS_neo*(R_avg / f) ### Convert into [A/m^2]

            if (inductive_jtor is not None):

                inductive_jtor[-1] = 0. # Enforce 0.0 at edge

                
                j_BS = np.nan_to_num(j_BS,nan=0.0)
                my_psi_N = psi_tmp
                my_Ip_target = Ip_target

                if scale_jBS is None:
                    spike_profile = j_BS
                    spike_profile[-1] = 0. # Enforce 0.0 at edge
                else:
                    jBS_results = analyze_bootstrap_edge_spike(psi_norm, j_BS)
                    spike_profile = jBS_results['spike_profile']*scale_jBS
                    spike_profile[-1] = 0. # Enforce 0.0 at edge

                # Find scalar "alpha" that solves integral(alpha*inductive_jtor + spike_profile) = Ip_target
                result = root_scalar(objective_function,
                                     args=(inductive_jtor, spike_profile, my_psi_N, my_Ip_target),
                                     bracket=[0.0001*my_Ip_target, 2*my_Ip_target],
                                     method='brentq',
                                     rtol=1e-6)

                # Extract the solution
                a_optimal = result.root

                matched_jtor_prof = (a_optimal * inductive_jtor)+spike_profile
                
                # Verify the solution
                Ip_final = self.flux_integral(my_psi_N, matched_jtor_prof)
                Ip_error = abs(Ip_final - Ip_target)
                
                print('a_optimal:')
                print(a_optimal)
                
                if (Ip_error < 10.):
                    print('Integrated j_tor match found')
                else:
                    print('WARNING: j_tor integration match failed~!')

                ffprime = ffprime_from_jtor_pprime(matched_jtor_prof, pprime, R_avg, one_over_R_avg)
                if scale_jBS is not None:
                    plt.plot(psi_tmp,scale_jBS*spike_profile,label='ped_spike')
                plt.plot(psi_tmp,matched_jtor_prof,label='matched_jtor_prof')
                plt.plot(psi_tmp,j_BS,ls='--',label='j_BS')
                plt.legend()
                plt.title('Iteration input J_tor')
                plt.show()
            else:
                print('Error: must specify j_tor profile function')
            
        else:
            j_BS = np.zeros_like(pressure)
            inductive_jtor[-1] = 0. # Enforce 0.0 at edge

            my_psi_N = psi_tmp
            my_Ip_target = Ip_target

            # Find scalar "alpha" that solves integral(alpha*inductive_jtor) = Ip_target
            result = root_scalar(objective_function,
                                 args=(inductive_jtor, j_BS, my_psi_N, my_Ip_target),
                                 bracket=[0.0001*my_Ip_target, 2*my_Ip_target],  # Provide a reasonable initial bracket 
                                 method='brentq',    # Robust bracketing method
                                 rtol=1e-6)          # Relative tolerance

            a_optimal = result.root

            matched_jtor_prof = a_optimal * inductive_jtor
            
            # Verify the solution
            Ip_final = self.flux_integral(my_psi_N, matched_jtor_prof)
            Ip_error = abs(Ip_final - Ip_target)

            ffprime = ffprime_from_jtor_pprime(matched_jtor_prof, pprime, R_avg, one_over_R_avg)

            plt.plot(psi_tmp,matched_jtor_prof,label='matched_jtor_prof')
            plt.plot(psi_tmp,j_BS,ls='--',label='j_BS')
            plt.legend()
            plt.title('L-mode Jtor')
            plt.show()
        
        ffp_prof = {
            'type': 'linterp',
            'x': psi_tmp,
            'y': ffprime / ffprime[0]
        }

        pp_prof = {
            'type': 'linterp',
            'x': psi_tmp,
            'y': pprime / pprime[0]
        }

        return pp_prof, ffp_prof, j_BS, matched_jtor_prof, psi_tmp, a_optimal

    def flatten_pedestals(self,inductive_jtor,psi_norm,ne,Te,ni,Ti,kBoltz):

        x_trimmed = psi_norm.tolist().copy()
        ne_trimmed = ne.tolist().copy()
        Te_trimmed = Te.tolist().copy()
        ni_trimmed = ni.tolist().copy()
        Ti_trimmed = Ti.tolist().copy()

        ### Remove profile values from psi_norm ~0.5 to ~0.99, leaving single value at the edge
        mid_index = int(len(x_trimmed)/2)
        end_index = len(x_trimmed)-1
        del x_trimmed[mid_index:end_index]
        del ne_trimmed[mid_index:end_index]
        del Te_trimmed[mid_index:end_index]
        del ni_trimmed[mid_index:end_index]
        del Ti_trimmed[mid_index:end_index]

        ### Fit cubic polynomials through all core and one edge value
        ne_model = np.poly1d(np.polyfit(x_trimmed, ne_trimmed, 3))
        Te_model = np.poly1d(np.polyfit(x_trimmed, Te_trimmed, 3))
        ni_model = np.poly1d(np.polyfit(x_trimmed, ni_trimmed, 3))
        Ti_model = np.poly1d(np.polyfit(x_trimmed, Ti_trimmed, 3))

        init_ne = ne_model(psi_norm)
        init_Te = Te_model(psi_norm)
        init_ni = ni_model(psi_norm)
        init_Ti = Ti_model(psi_norm)
        

        init_pressure = (kBoltz * init_ne * init_Te) + (kBoltz * init_ni * init_Ti)
        
        plt.plot(init_pressure)
        plt.title('init_pressure')
        plt.show()
        
        return init_pressure,init_ne,init_ni,init_Te,init_Ti

    if R0 and Z0 and a0:
        kappa = 1.0
        delta = 0.0
        print('>>> Initializing psi')
        err_flag = self.init_psi(R0, Z0, a0, kappa, delta)
        err_flag = self.solve()
        
    mu0 = np.pi*4.E-7
    kBoltz = 1.602e-19
    pressure = (kBoltz * ne * Te) + (kBoltz * ni * Ti) # 1.602e-19 * [m^-3] * [eV] = [Pa]

    ### Reconstruct psi_norm and n_psi from pressure
    psi_norm = np.linspace(0.,1.,len(pressure))
    n_psi = len(pressure)
        
    ### Set new pax target
    self.set_targets(Ip=Ip_target,pax=pressure[0])

    ### Initialize equilibirum on L-mode-like P' and inductive j_tor profiles
    if initialize_eq:
        print('>>> Initializing equilibrium with pedestal removed')
        init_pressure,init_ne,init_ni,init_Te,init_Ti = flatten_pedestals(self,inductive_jtor,psi_norm,ne,Te,ni,Ti,kBoltz)
        init_pp_prof,init_ffp_prof,j_BS,_,psi_tmp,a_optimal = profile_iteration(self,init_pressure,init_ne,init_ni,init_Te,init_Ti,psi_norm,n_psi,Zeff,inductive_jtor,scale_jBS,Zis,include_jBS=False)

        init_pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
        init_ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge

        init_pp_prof['y'] = np.nan_to_num(init_pp_prof['y'])
        init_ffp_prof['y'] = np.nan_to_num(init_ffp_prof['y'])

        self.set_targets(Ip=Ip_target,pax=pressure[0])
        self.set_profiles(ffp_prof=init_ffp_prof,pp_prof=init_pp_prof)
        
        flag = self.solve()
        psi_init = self.get_psi(False)
        
        psi,f,fp,p,pp = self.get_profiles(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
        _,qvals,ravgs,_,_,_ = self.get_q(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]
        l_mode_ffprime = f*fp
        lmode_jtor = jtor_from_GS(l_mode_ffprime, pp, R_avg, one_over_R_avg)
        print('calculated L-mode jtor')
    else:
        psi_init = self.get_psi(False)
        psi,f,fp,p,pp = self.get_profiles(npsi=n_psi,psi_pad=1e-4)
        _,qvals,ravgs,_,_,_ = self.get_q(npsi=n_psi,psi_pad=1e-4)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]
        l_mode_ffprime = f*fp
        lmode_jtor = jtor_from_GS(l_mode_ffprime, pp, R_avg, one_over_R_avg)
        pass
        # USE EXISTING EQUILIBRIUM IN MYGS?
        
    ### This block solves for the complete current profile using a user-specified inductive j_tor profile
    if inductive_jtor is not None:
        ### Specify original H-mode profiles, iterate on bootstrap contribution until reasonably converged
        n = 0
        print('>>> Iterating on H-mode equilibrium solution')
        jtor_last = lmode_jtor
        while n < max_iterations:
            print('> Iteration '+str(n)+':')

            # Need L-mode jtor
            pp_prof, ffp_prof, j_BS, input_jtor, psi_tmp, a_optimal = profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,scale_jBS,Zis)

            pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
            ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge

            pp_prof['y'] = np.nan_to_num(pp_prof['y']) # Check for any nan's
            ffp_prof['y'] = np.nan_to_num(ffp_prof['y']) # Check for any nan's

            plt.plot(pp_prof['x'],pp_prof['y'])
            plt.title('Iteration input pprime')
            plt.show()
            
            plt.plot(ffp_prof['x'],ffp_prof['y'])
            plt.title('Iteration input ffprime')
            plt.show()

            ffp_prof['type'] = 'jphi-linterp'
            ffp_prof['y'] = input_jtor #(input_jtor + jtor_last)/2.0
            # jtor_last = ffp_prof['y']
        
            self.set_targets(Ip=Ip_target,pax=pressure[0])
            self.set_profiles(ffp_prof=ffp_prof,pp_prof=pp_prof)

            flag = self.solve()

            n += 1
            if (n > 4):
                break
                
            psi,f,fp,p,pp = self.get_profiles(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
            _,qvals,ravgs,_,_,_ = self.get_q(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
            R_avg = ravgs[0]
            one_over_R_avg = ravgs[1]
            tmp_ffprime = f*fp
            tmp_jtor = jtor_from_GS(tmp_ffprime, pp, R_avg, one_over_R_avg)

            Ip_tmp = self.flux_integral(psi, tmp_jtor)
            Ip_error_tmp = abs(Ip_tmp - Ip_target)

            print('Ip target: '+str(Ip_target/1e+6)+' [MA]')
            print('Ip error: '+str(Ip_error_tmp/1e+6)+' [MA]')

            plt.plot(psi,tmp_jtor)
            plt.plot(psi,input_jtor,linestyle='--')
            plt.title('Iteration output j_tor, Ip error: '+str(round(Ip_error_tmp/1e+6,5))+' [MA]')
            plt.show()
            
            # Need to interface with Chris about how to assess solve success
            
         #   elif n >= max_iterations:
         #       raise TypeError('H-mode equilibrium solve did not converge')
        
        psi,f,fp,p,pp = self.get_profiles(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
        _,qvals,ravgs,_,_,_ = self.get_q(psi=psi_tmp,npsi=n_psi,psi_pad=1e-4)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]
        ffprime = f*fp
        final_jtor = jtor_from_GS(ffprime, pp, R_avg, one_over_R_avg)
        #def jtor_from_GS(ffprime, pprime, R_avg, one_over_R_avg):
    return self, flag, final_jtor, j_BS, input_jtor, a_optimal

# Import fit functions
import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep
from scipy.optimize import minimize

def guess(x, y, k, s, w=None):
    """Do an ordinary spline fit to provide knots"""
    return splrep(x, y, w, k=k, s=s)

def err(c, x, y, t, k, w=None):
    """The error function to minimize"""
    diff = y - splev(x, (t, c, k))
    if w is None:
        diff = np.einsum('...i,...i', diff, diff)
    else:
        diff = np.dot(diff*diff, w)
    return np.abs(diff)

def prior(c, x, t, k, p):
    return splev(x, (t, c, k), der=2).sum()/p

def err_prior(c, x, y, t, k, p, w=None):
    """The error function to minimize"""
    diff = y - splev(x, (t, c, k))
    if w is None:
        diff = np.einsum('...i,...i', diff, diff)
    else:
        diff = np.dot(diff*diff, w)
    return np.abs(diff)+np.abs(prior(c, x, t, k, p))

def spline(x, y, p=1, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    opt = minimize(err_prior, c0, (x, y, t, k, p, w))
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_negative_derivative_curvature(x, y, p=1, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    con = {'type': 'ineq',
           'fun': lambda c: -splev(x, (t, c, k), der=1),
           }
    opt = minimize(err_prior, c0, (x, y, t, k, p, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_negative_derivative_null_center(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[0]
    con = [{'type': 'ineq',
        'fun': lambda c: -splev(x, (t, c, k), der=1),
           },\
          {'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1),
           }]
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_null_center(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[0]
    con = [{'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1),
           },]
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_negative_derivative(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    con = {'type': 'ineq', 'fun': lambda c: -splev(x, (t, c, k), der=1)}
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_positive_derivative(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    con = {'type': 'ineq', 'fun': lambda c: splev(x, (t, c, k), der=1)}
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_neumann_left(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[0] # point at which zero slope is required
    con = {'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1),
           #'jac': lambda c: splev(x0, (t, c, k), der=2) # doesn't help, dunno why
           }
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def spline_neumann_right(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[-1] # point at which zero slope is required
    con = {'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1),
           #'jac': lambda c: splev(x0, (t, c, k), der=2) # doesn't help, dunno why
           }
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))
