#exec(open("/home/stubenj9/tearing-physics-suite/equilibrium_helper/re_solver_functions.py").read())
import json
import numpy as np

def SPARC_bounds(mygs_col_sets):
    coil_bounds = {}
    curr_ratio = 4.8E4/4.2E4
    for key in mygs_col_sets:
        if key.startswith('CS1') or key.startswith('CS2'):
            coil_bounds[key] = [-5.00E+04,5.00E+04]
        elif key.startswith('CS3'):
            coil_bounds[key] = [0.0,5.00E+04]
        elif key.startswith('PF1') or key.startswith('PF2'):
            coil_bounds[key] = [-2.50E+04,5.00E+04]
        elif key.startswith('PF3'):
            coil_bounds[key] = [-5.00E+04,2.50E+04]
        elif key.startswith('PF4'):
            coil_bounds[key] = [-5.00E+04,0.0]
        elif key.startswith('DV'):
            coil_bounds[key] = [-3.20E+04,3.20E+04]
        elif key.startswith('VS'):
            coil_bounds[key] = [-3.00E+04,3.00E+04]
    return coil_bounds

def SPARC_coil_reg(mygs):
    # Set regularization weights for coils
    regularization_terms = []
    for name, coil in mygs.coil_sets.items():
        # Set zero target current and different small weights to help conditioning of fit
        if name.startswith('CS'):
            if name.startswith('CS1'):
                regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=2.E-2))
            else:
                regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
        elif name.startswith('PF'):
            regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
        elif name.startswith('VS'):
            regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
    # Disable VSC virtual coil
    regularization_terms.append(mygs.coil_reg_term({'#VSC': 1.0},target=0.0,weight=1.E4))
    mygs.set_coil_reg(reg_terms=regularization_terms)
    return mygs

def re_solve_equilibrium(mygs, mesh_filename, eqdsk_filename, device,
    eqdsk_filename_for_profiles=None,json_for_profiles=None, gacode_filename_for_profiles=None,
    maxits=200, nl_tol=1e-6, order=2, ip_scale=1.0, p_scale=1.0,len_psi_sample=128, use_cubic_splines=False,**kwargs):

    if gacode_filename_for_profiles is not None and json_for_profiles is not None:
        raise ValueError("Cannot specify both gacode_filename_for_profiles and json_for_profiles")
    if gacode_filename_for_profiles is not None and eqdsk_filename_for_profiles is not None:
        raise ValueError("Cannot specify both gacode_filename_for_profiles and eqdsk_filename_for_profiles")
    if json_for_profiles is not None and eqdsk_filename_for_profiles is not None:
        raise ValueError("Cannot specify both json_for_profiles and eqdsk_filename_for_profiles")

    # Clear mygs 
    mygs.reset()

    sparc_dict = {
        'F0': 1.85*12.5,
        'coil_vsc': {'VS1U': 1.0},
        'coil_bounds': [-50.E6, 50.E6],
        'lim_zmax': 1.15,
        'coil_bound_func': SPARC_bounds,
        'coil_reg_func': SPARC_coil_reg,
        'x_points': np.array([[1.52, 1.11],[1.52, -1.11]]),
        'x_weights': np.array([10.0, 10.0]),
        'R0': 1.85,
        'Z0': 0.0,
        'a': 0.56,
        'kappa': 1.7,
        'delta': 0.2,
        'up_down_symmetric': True
    }
    sparc_Ldict = {
        'F0': 1.85*12.5,
        'coil_vsc': {'VS1U': 1.0},
        'coil_bounds': [-50.E6, 50.E6],
        'lim_zmax': 1.15,
        'coil_bound_func': SPARC_bounds,
        'coil_reg_func': SPARC_coil_reg,
        'x_points': np.array([[1.52, -1.12]]),
        'x_weights': np.array([10.0]),
        'R0': 1.85,
        'Z0': 0.0,
        'a': 0.56,
        'kappa': 1.7,
        'delta': 0.2,
        'up_down_symmetric': False
    }

    if device == 'SPARC_Hmode':
        device_dict = sparc_dict
    elif device == 'SPARC_Lmode':
        device_dict = sparc_Ldict
    else:
        raise ValueError('Device not recognised.')

    if json_for_profiles is not None:
        with open(json_for_profiles, 'r') as f:
            omas_data = json.load(f)
        # Global equilibrium quantities:
        B0 = omas_data['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['b_field_tor']
        R0 = omas_data['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['r']
        F0 = B0*R0
        device_dict['F0'] = F0
        device_dict['R0'] = R0
        if abs(omas_data['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['z']) > 5e-3:
            raise ValueError("Equilibrium magnetic axis is not in midplane, need to adjust m3dc1 settings to handle this")
    elif gacode_filename_for_profiles is not None:
        gacode_data = read_gacode(gacode_filename_for_profiles)
        B0 = gacode_data['bcentr']
        assert gacode_data['bcentr_units'] == 'T', "Add non-Tesla units handling"
        R0 = gacode_data['rcentr']
        assert gacode_data['rcentr_units'] == 'm', "Add non-metre units handling"
        F0 = B0*R0
        device_dict['F0'] = F0
        device_dict['R0'] = R0
        

    mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh(mesh_filename)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
    mygs.settings.maxits=maxits
    mygs.settings.nl_tol=nl_tol

    if 'lim_zmax' in device_dict:
        mygs.lim_zmax = device_dict['lim_zmax']

    mygs.setup(order=order, F0=device_dict['F0'])

    mygs.set_coil_vsc(device_dict['coil_vsc'])

    coil_bounds = device_dict['coil_bound_func'](mygs.coil_sets)
    mygs.set_coil_bounds(coil_bounds)

    #read in target equilibrium
    eqdsk = read_eqdsk(eqdsk_filename)

    if device_dict['up_down_symmetric']:
        x_ind = abs(eqdsk['rzout'][:,1]).argmax()
        isoflux_pts = eqdsk['rzout']
        isoflux_pts = np.vstack((isoflux_pts,[eqdsk['rzout'][x_ind,:],]))
        isoflux_pts_mirror = isoflux_pts.copy()
        isoflux_pts_mirror[:,1] *= -1.0
        isoflux_pts = np.vstack((isoflux_pts,isoflux_pts_mirror))
    else:
        isoflux_pts = eqdsk['rzout']

    isoflux_weights = np.ones(len(isoflux_pts[:,0]))
    mygs.set_isoflux(isoflux_pts, isoflux_weights)

    mygs.set_saddles(device_dict['x_points'], device_dict['x_weights'])
    
    mygs = device_dict['coil_reg_func'](mygs)

    if json_for_profiles is None and gacode_filename_for_profiles is None: 
        if eqdsk_filename_for_profiles is None:
            eqdsk_w_profiles = eqdsk
        else:
            eqdsk_w_profiles = read_eqdsk(eqdsk_filename_for_profiles)
        ffp_prof,pp_prof = get_profs(eqdsk_w_profiles,len_psi_sample=len_psi_sample,use_cubic_splines=use_cubic_splines)
        Ip_target = abs(eqdsk_w_profiles['ip'])*ip_scale
        pax_target = eqdsk_w_profiles['pres'][0]*p_scale# 2600000 #
    elif json_for_profiles is not None: # Assumes json_for_profiles is not None
        with open(json_for_profiles, 'r') as f:
            omas_data = json.load(f)
        ffp_prof, pp_prof = get_profs_json(omas_data,len_psi_sample=len_psi_sample,use_cubic_splines=use_cubic_splines)
        Ip_target = omas_data['equilibrium']['time_slice'][0]['global_quantities']['ip']*ip_scale
        pax_target = omas_data['equilibrium']['time_slice'][0]['profiles_1d']['pressure'][0]*p_scale
    else:
        gacode_data = read_gacode(gacode_filename_for_profiles) 
        ffp_prof, pp_prof = get_profs_gacode(gacode_data,len_psi_sample=len_psi_sample,use_cubic_splines=use_cubic_splines,**kwargs)
        Ip_target = gacode_data['current']*1e6*ip_scale # Convert to Amps
        assert gacode_data['current_units'] == 'MA', "Add non-MA units handling"
        pax_target = gacode_data['ptot'][0]*p_scale # in Pa
        assert gacode_data['ptot_units'] == 'Pa', "Add non-Pa units handling"


    mygs.set_targets(Ip=Ip_target,pax=pax_target)
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

    device_dict['Ip_target'] = Ip_target
    device_dict['pax_target'] = pax_target
    device_dict['ffp_prof'] = ffp_prof
    device_dict['pp_prof'] = pp_prof

    # Initialize psi and run free boundary
    err_flag = mygs.init_psi(device_dict['R0'], device_dict['Z0'], device_dict['a'], device_dict['kappa'], device_dict['delta'])
    err_flag = mygs.solve()
    #psi0 = mygs.get_psi(normalized=False)

    return mygs, device_dict

def get_profs_gacode(gacode_data=None, gacode_filename_for_profiles=None, normalise=True,set_edge_zero=True,len_psi_sample=257,use_cubic_splines=False, assume_jbstor_is_total_jtor=True,
                SPARC_CGYRO_PRD=False, use_thermal_pressure=True):
    #########################################################################################################
    # Read in data: 
    #########################################################################################################
    if gacode_filename_for_profiles is None and gacode_data is None:
        raise ValueError("Either omas_data or json_for_profiles must be provided")
    if gacode_data is None:
        gacode_data = read_gacode(gacode_filename_for_profiles)
    pp_nonnormalised = np.array(gacode_data['ptot'])
    if use_thermal_pressure:
        psi_norm_therm, pp_nonnormalised = get_kinetic_profs_gacode(gacode_data=gacode_data,return_thermal_pressure=True,debug=False)
        assert np.all(psi_norm_therm == np.array(np.array(gacode_data['polflux']/gacode_data['polflux'][-1]))), "Something wrong with psi_normal"
    psi_gacode = np.array(gacode_data['polflux']/gacode_data['polflux'][-1])
    assert gacode_data.get('ptot_units', None) == 'Pa', "Add non-Pa units handling"
    if assume_jbstor_is_total_jtor:
        Jtot_unnormalised = np.array(gacode_data['jbstor'])
        assert gacode_data.get('jbstor_units', None) == 'MA/m^2', "Add non-'MA/m^2' units handling"
        # Look for big jumps in jbstor and remove them:
        psi_gradient = np.diff(Jtot_unnormalised)/np.diff(np.array(gacode_data['polflux']/gacode_data['polflux'][-1]))
        #bad_inds = np.where(np.abs(psi_gradient/np.median(psi_gradient)) > 20) 
        if max(psi_gradient) > 15 and not SPARC_CGYRO_PRD:
            raise ValueError("jbstor profile has large gradients, please check the input data")
        elif SPARC_CGYRO_PRD:
            Jtot_unnormalised = Jtot_unnormalised[2:107]
            psi_normalised_jtot = np.array(gacode_data['polflux']/gacode_data['polflux'][-1])[2:107]
    else:
        raise ValueError("Update this to handle other current profiles from gacode output")
    # Check minimum psi spacing:
    psi_min_diff = np.min(np.diff(psi_gacode))
    psi_sample = np.linspace(0.0,1.0,len_psi_sample)
    # Check if psi_sample spacing is less than half the minimum psi spacing in json:
    if np.min(np.diff(psi_sample)) < 0.5*psi_min_diff:
        # Reformulate psi_sample:
        len_psi_sample = int(1.0/(0.5*psi_min_diff))+1
        psi_sample = np.linspace(0.0,1.0,len_psi_sample)
        print("Warning: psi spacing in json is quite coarse, reformulating psi_sample to have spacing of half the minimum psi spacing in json, new len_psi_sample = "+str(len_psi_sample))
    psi_prof = np.copy(psi_sample)
    # Make psi_prof_jtor equal to all psi_prof points that in the range of psi_normalised_jtot
    #print('max and min of psi_normalised_jtot:', max(psi_normalised_jtot), min(psi_normalised_jtot))
    psi_prof_jtor = [i for i in psi_prof if (i < max(psi_normalised_jtot) and i > min(psi_normalised_jtot))]
    #print("Using "+str(len(psi_prof_jtor))+" points for jtor profile out of "+str(len(psi_prof))+" total points")
    if not use_cubic_splines:
        pp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_gacode,pp_nonnormalised)))).copy()
        jtor_prof = np.transpose(np.vstack((psi_prof_jtor,np.interp(psi_prof_jtor,psi_normalised_jtot,Jtot_unnormalised)))).copy()
        #print(jtor_prof)
    else:
        from scipy.interpolate import CubicSpline
        pp_spline = CubicSpline(psi_gacode,pp_nonnormalised,extrapolate=True)
        pp_prof = np.transpose(np.vstack((psi_prof,pp_spline(psi_sample)))).copy()
        jtor_spline = CubicSpline(psi_normalised_jtot,Jtot_unnormalised,extrapolate=True)
        jtor_prof = np.transpose(np.vstack((psi_prof_jtor,jtor_spline(psi_prof_jtor)))).copy()
    jtor_prof={'type': 'jphi-linterp', 'y': jtor_prof[:,1], 'x': psi_prof_jtor}
    pp_prof={'type': 'linterp', 'y': pp_prof[:,1], 'x': psi_sample}
    if normalise:
        pp_prof['y'] /= np.abs(pp_prof['y']).max() # Normalize profile (not required but convienient)
        jtor_prof['y'] /= np.abs(jtor_prof['y']).max() # Normalize profile (not required but convienient)
        if pp_prof['y'][0]<0.0:
            pp_prof['y'] *= -1.0 # Make sure pp is positive at axis
        if jtor_prof['y'][0]<0.0:
            jtor_prof['y'] *= -1.0 # Make sure jtor is positive at axis
    if set_edge_zero:
        pp_prof['y'][-1] = 0.0 # Set edge to zero
        jtor_prof['y'][-1] = 0.0 # Set edge to zero
    return jtor_prof, pp_prof

def get_profs_json(omas_data,normalise=True,set_edge_zero=True,len_psi_sample=257,use_cubic_splines=False):
    if len(omas_data['equilibrium']['time_slice']) > 1: # Fix this to allow time slices
        raise ValueError("Multiple time slices found in json, please provide a json with a single time slice")
    ffp_nonnormalised = np.array(omas_data['equilibrium']['time_slice'][0]['profiles_1d']['f_df_dpsi'])
    pp_nonnormalised = np.array(omas_data['equilibrium']['time_slice'][0]['profiles_1d']['dpressure_dpsi'])
    psi_json = np.array(omas_data['equilibrium']['time_slice'][0]['profiles_1d']['psi_norm'])
    # Check minimum psi spacing:
    psi_min_diff = np.min(np.diff(psi_json))
    psi_sample = np.linspace(0.0,1.0,len_psi_sample)
    # Check if psi_sample spacing is less than half the minimum psi spacing in json:
    if np.min(np.diff(psi_sample)) < 0.5*psi_min_diff:
        # Reformulate psi_sample:
        len_psi_sample = int(1.0/(0.5*psi_min_diff))+1
        psi_sample = np.linspace(0.0,1.0,len_psi_sample)
        print("Warning: psi spacing in json is quite coarse, reformulating psi_sample to have spacing of half the minimum psi spacing in json, new len_psi_sample = "+str(len_psi_sample))
    psi_prof = np.copy(psi_sample)
    if not use_cubic_splines:
        ffp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_json,ffp_nonnormalised)))).copy()
        pp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_json,pp_nonnormalised)))).copy()
    else:
        from scipy.interpolate import CubicSpline
        ffp_spline = CubicSpline(psi_json,ffp_nonnormalised,extrapolate=True)
        pp_spline = CubicSpline(psi_json,pp_nonnormalised,extrapolate=True)
        ffp_prof = np.transpose(np.vstack((psi_prof,ffp_spline(psi_sample)))).copy()
        pp_prof = np.transpose(np.vstack((psi_prof,pp_spline(psi_sample)))).copy()
    ffp_prof={'type': 'linterp', 'y': ffp_prof[:,1], 'x': psi_sample}
    pp_prof={'type': 'linterp', 'y': pp_prof[:,1], 'x': psi_sample}
    if normalise:
        ffp_prof['y'] /= np.abs(ffp_prof['y']).max() # Normalize profile (not required but convienient)
        pp_prof['y'] /= np.abs(pp_prof['y']).max() # Normalize profile (not required but convienient)
        if ffp_prof['y'][0]<0.0:
            ffp_prof['y'] *= -1.0 # Make sure ffp is positive at axis
        if pp_prof['y'][0]<0.0:
            pp_prof['y'] *= -1.0 # Make sure pp is positive at axis
    if set_edge_zero:
        ffp_prof['y'][-1] = 0.0 # Set edge to zero
        pp_prof['y'][-1] = 0.0 # Set edge to zero
    return ffp_prof, pp_prof

def get_kinetic_profs_json(omas_data=None,json_for_profiles=None,mygs=None, single_species_simplification=True, eta_fac=None, Coulomb_logarithm=17, exclude_species=[], debug=True):
    """ Extract kinetic profiles from omas_data or json_for_profiles and return as xarray Dataset.
        If mygs is provided, computes the equivalent single-ion-species model for the combined mass and 
        charge of the multiple ion species. """
    import xarray as xr
    kBoltz = 1.602e-19
    #########################################################################################################
    # Read in omas_data if not provided:
    #########################################################################################################
    if json_for_profiles is not None:
        with open(json_for_profiles, 'r') as f:
            omas_data = json.load(f)
    elif omas_data is None:
        raise ValueError("Either omas_data or json_for_profiles must be provided")
    chargedict={'e': -1.0, 'H': 1.0, 'D': 1.0, 'T': 1.0, 'He': 2.0, 'He3': 2.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'Ne': 10.0, 'Ar': 18.0,'Fe': 26.0,'W': 74.0}
    massdict_amu={'H': 1.00784, 'D': 2.01410, 'T': 3.01605, 'He': 4.002602, 'He3': 3.016029, 'C': 12.011, 'N': 14.00, 'O': 16.00, 'Ne': 20.1797, 'Ar': 39.9,'Fe': 55.845,'W': 183.84}
    
    #########################################################################################################
    # Check omas_data has single time slice (add multiple timeslice support later):
    #########################################################################################################
    if len(omas_data['core_profiles']['time']) > 1: # Fix this to allow time slices
        raise ValueError("Multiple time slices found in json, please provide a json with a single time slice")

    ### Double check that the thermal pressure matches the equilibrium pressure in omas_data:
    pax_eqdsk = omas_data['equilibrium']['time_slice'][0]['profiles_1d']['pressure'][0] # in Pa
    pax_profiles_1d = omas_data['core_profiles']['profiles_1d'][0]['pressure_thermal'][0] # in Pa
    if np.abs(pax_eqdsk - pax_profiles_1d)/pax_eqdsk > 0.001:
        print("Warning: pax from equilibrium profiles_1d = "+str(pax_eqdsk)+" Pa does not match pax from core_profiles_1d = "+str(pax_profiles_1d)+" Pa, relative difference = "+str(np.abs(pax_eqdsk - pax_profiles_1d)/pax_eqdsk))
    
    #########################################################################################################
    ### Extract Electron info with unit checks:
    #########################################################################################################
    psi_normalised = np.array(omas_data['core_profiles']['profiles_1d'][0]['grid']['psi'])
    ne_prof = np.array(omas_data['core_profiles']['profiles_1d'][0]['electrons']['density']) # units are 10^20 m^-3 or m^-3
    ### We make sure it's m^-3:
    if np.max(ne_prof) < 1.E6:
        ne_prof *= 1.E20

    te_prof = np.array(omas_data['core_profiles']['profiles_1d'][0]['electrons']['temperature']) # units are keV or eV
    ###
    ### We make sure it's keV:
    if np.max(te_prof) > 500:
        te_prof /= 1.E3

    # Make into xarray DataArrays:
    te_da = xr.DataArray(te_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='te', attrs={'units': 'keV'})
    ne_da = xr.DataArray(ne_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ne', attrs={'units': 'm^-3'})
    profiles_xr = xr.Dataset({'te': te_da, 'ne': ne_da})

    #########################################################################################################
    ### Extract Ion info with unit checks:
    #########################################################################################################
    ion_labels=[]
    ni_profs=[]
    ti_profs=[]
    ion_datasets = []
    ###
    ### Cycle through ion species:
    for i in range(len(omas_data['core_profiles']['profiles_1d'][0]['ion'])):
        # Check if any component of density is nonzero:
        if np.all(np.array(omas_data['core_profiles']['profiles_1d'][0]['ion'][i]['density'])==0.0):
            continue
        if omas_data['core_profiles']['profiles_1d'][0]['ion'][i]['label'] in exclude_species:
            continue
        ion_labels.append(omas_data['core_profiles']['profiles_1d'][0]['ion'][i]['label'])
        # Unit checks and conversions:
        ni_profs.append(np.array(omas_data['core_profiles']['profiles_1d'][0]['ion'][i]['density'])) # units are 10^20 m^-3 or m^-3
        if np.max(ni_profs[-1]) < 1.E6:
            ni_profs[-1] *= 1.E20
        ti_profs.append(np.array(omas_data['core_profiles']['profiles_1d'][0]['ion'][i]['temperature'])) # units are keV or eV
        if np.max(ti_profs[-1]) > 100:
            ti_profs[-1] /= 1.E3
        # Convert to xarray DataArrays:
        ni_da = xr.DataArray(ni_profs[-1], coords=[psi_normalised], dims=['psi_n_1d'], name='ni', attrs={'units': 'm^-3'})
        ti_da = xr.DataArray(ti_profs[-1], coords=[psi_normalised], dims=['psi_n_1d'], name='ti', attrs={'units': 'keV'})
        # Add a species coordinate to these DataArrays:
        ni_da = ni_da.expand_dims({'species': [ion_labels[-1]]})
        ti_da = ti_da.expand_dims({'species': [ion_labels[-1]]})
        # Create a dataset for this species:
        ion_ds = xr.Dataset({'ni': ni_da, 'ti': ti_da, 'charge': chargedict[ion_labels[-1]], 'mass_amu': massdict_amu[ion_labels[-1]]})
        ion_datasets.append(ion_ds)
    # Concatenate all ion datasets along species dimension:
    ion_ds_all = xr.concat(ion_datasets, dim='species')
    # Merge with profiles_xr:
    profiles_xr = xr.merge([profiles_xr, ion_ds_all])
    
    #########################################################################################################
    ### Check all profiles have same length:
    #########################################################################################################
    len_prof = len(psi_normalised)
    if len(ne_prof) != len_prof:
        raise ValueError("ne profile length does not match psi length")
    if len(te_prof) != len_prof:
        raise ValueError("te profile length does not match psi length")
    for i in range(len(ion_labels)):
        if len(ni_profs[i]) != len_prof:
            raise ValueError("ni profile length for ion "+ion_labels[i]+" does not match psi length")
        if len(ti_profs[i]) != len_prof:
            raise ValueError("ti profile length for ion "+ion_labels[i]+" does not match psi length")

    #########################################################################################################
    ### Create total ion mass profile and ion number density profiles. Also check quasineutrality (full ionization, not accurate for high-Z impurities):
    #########################################################################################################
    ne_check = np.zeros(len_prof)
    total_ion_mass_prof = np.zeros(len_prof)
    total_ion_density_prof = np.zeros(len_prof)
    for i in range(len(ion_labels)):
        if ion_labels[i] not in chargedict:
            raise ValueError("Ion label "+ion_labels[i]+" not recognised, please add to chargedict")
        ne_check += ni_profs[i]*chargedict[ion_labels[i]]
        total_ion_mass_prof += ni_profs[i]*massdict_amu[ion_labels[i]]
        total_ion_density_prof += ni_profs[i]
    
    #########################################################################################################
    ### See if sum of ion densities times ion charges matches electron density times -1:
    #########################################################################################################
    if np.max(np.abs(ne_check - ne_prof)/ne_prof) > 0.01:
        print(np.abs(ne_check - ne_prof)/ne_prof)
        print("On-axis charge density for electrons = "+str(ne_prof[0]))
        for i in range(len(ion_labels)):
            print("On-axis charge density for ion "+ion_labels[i]+" = "+str((ni_profs[i]*chargedict[ion_labels[i]])[0])," assuming complete ionization with charge "+str(chargedict[ion_labels[i]]))
        print("Sum of on-axis ion charge densities = "+str(ne_check[0]))
        if len(ion_labels) == 1:
            raise ValueError("Quasineutrality check failed for a single-ion-species plasma, something is wrong")
    
    #########################################################################################################
    # Run single_species_simplification if requested:
    #########################################################################################################

    if not single_species_simplification:
        profiles_xr = profiles_xr.assign_attrs({'single_species_model': False})
        return profiles_xr
    elif single_species_simplification and not mygs:
        print("Warning: single_species_simplification requested but mygs not provided, cannot compute flux surface averages, returning multi-species profiles only")
        return profiles_xr

    total_electron_density = mygs.flux_integral(psi_normalised,ne_prof) # in m^-3
    total_ion_density = mygs.flux_integral(psi_normalised,total_ion_density_prof) # in m^-3
    total_electron_pressure_prof = ne_prof*te_prof*kBoltz*1.E3 # # 1.602e-19 * [m^-3] * [eV] = [Pa]
    profiles_xr = profiles_xr.assign_attrs({'total_electron_density_m3': total_electron_density, 'total_ion_density_m3': total_ion_density})
    if len(ion_labels) == 1:
        ss_ion_charge = chargedict[ion_labels[0]]
        ss_ion_mass = massdict_amu[ion_labels[0]]
        profiles_xr = profiles_xr.assign({'ss_ion_charge': chargedict[ion_labels[0]], 'ss_ion_mass': massdict_amu[ion_labels[0]]})
        profiles_xr['ss_ion_density'] = profiles_xr['ni'].sel(species=ion_labels[0])
        profiles_xr['ss_ion_temp'] = profiles_xr['ti'].sel(species=ion_labels[0])
        total_ion_pressure_prof = (profiles_xr['ss_ion_density']*profiles_xr['ss_ion_temp']*kBoltz*1.E3).values # 1.602e-19 * [m^-3] * [eV] = [Pa]
        total_thermal_pressure_prof = total_ion_pressure_prof + total_electron_pressure_prof
    else:
        ### Effective charge:
        ion_effective_charge = total_electron_density/total_ion_density
        ### Effective mass:
        ion_effective_mass = mygs.flux_integral(psi_normalised,total_ion_mass_prof)/total_ion_density
        ### Add to profiles_xr:
        profiles_xr = profiles_xr.assign({'ss_ion_charge': ion_effective_charge, 'ss_ion_mass': ion_effective_mass})
        #########################################################################################################
        ### If the underlying ion species have different temperature profiles, we need to adjust to make sure the single-species Ti matches the pressure:
        #########################################################################################################
        # Compute total ion pressure from individual species:
        total_ion_pressure_prof = np.zeros(len_prof)
        for i in range(len(ion_labels)):
            total_ion_pressure_prof += ni_profs[i]*ti_profs[i]*kBoltz*1.E3 # 1.602e-19 * [m^-3] * [eV] = [Pa]
            if debug:
                print("On-axis ion pressure for ion "+ion_labels[i]+" = "+str((ni_profs[i]*ti_profs[i]*kBoltz*1.E3)[0])+" Pa")
        if debug:
            print("On-axis ion pressure for electrons = "+str((ne_prof*te_prof*kBoltz*1.E3)[0])+" Pa")
        # Total thermal pressure:
        total_thermal_pressure_prof = total_ion_pressure_prof + total_electron_pressure_prof
        # Check this is consistent with pax from omas_data:
        if np.abs(total_thermal_pressure_prof[0] - pax_profiles_1d)/pax_profiles_1d > 0.01: 
            print(
                "Warning: computed on-axis thermal pressure from kinetic profiles = "+str(total_thermal_pressure_prof[0])+" Pa does not match pax from core_profiles_1d = "+str(pax_profiles_1d)+" Pa, relative difference = "+str(np.abs(total_thermal_pressure_prof[0] - pax_profiles_1d)/pax_profiles_1d)
            )
            raise ValueError("Error: computed on-axis thermal pressure from kinetic profiles is inconsistent")
        # Now compute the Ti profile that would give this total thermal pressure assuming a single ion species with density total_ion_density_prof:
        ti_equiv_prof = (total_thermal_pressure_prof - total_electron_pressure_prof)/(total_ion_density_prof*kBoltz*1.E3) # in keV
        profiles_xr['ss_ion_temp'] = xr.DataArray(ti_equiv_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ss_ion_temp', attrs={'units': 'keV'})
        profiles_xr['ss_ion_density'] = xr.DataArray(total_ion_density_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ss_ion_density', attrs={'units': 'm^-3'})

    #########################################################################################################
    # Sanity checks on single species quasi neutrality, and pressure consistency:
    #########################################################################################################
    # Check quasi-neutrality:
    if np.max(np.abs(profiles_xr['ss_ion_density']*profiles_xr['ss_ion_charge'] - profiles_xr['ne'])/profiles_xr['ne']) > 0.01:
        raise ValueError("Quasineutrality check failed for single-ion-species model, something is wrong")
    # Check pressure consistency:
    ss_ion_pressure_prof = profiles_xr['ss_ion_density']*profiles_xr['ss_ion_temp']*kBoltz*1.E3 # 1.602e-19 * [m^-3] * [eV] = [Pa]
    total_thermal_pressure_prof_ss = ss_ion_pressure_prof + total_electron_pressure_prof
    if np.max(np.abs(total_thermal_pressure_prof_ss - total_thermal_pressure_prof)/total_thermal_pressure_prof) > 0.01:
        raise ValueError("Pressure consistency check failed for single-ion-species model, something is wrong")

    profiles_xr = profiles_xr.assign_attrs({'single_species_model': True})
    if eta_fac is not None:
        profiles_xr['eta_fac'] = eta_fac
    if Coulomb_logarithm is not None:
        profiles_xr['Coulomb_logarithm'] = Coulomb_logarithm
    return profiles_xr

def get_kinetic_profs_gacode(gacode_data=None, gacode_filename_for_profiles=None, mygs=None, single_species_simplification=True, eta_fac=None, Coulomb_logarithm=17, exclude_species=[], debug=True, return_thermal_pressure=False):
    """ Extract kinetic profiles from a GACODE output file and return as xarray Dataset.
        If mygs is provided, computes the equivalent single-ion-species model for the combined mass and 
        charge of the multiple ion species. """
    import xarray as xr
    import h5py
    kBoltz = 1.602e-19
    #########################################################################################################
    # Read in data: 
    #########################################################################################################
    if gacode_filename_for_profiles is None and gacode_data is None:
        raise ValueError("Either omas_data or json_for_profiles must be provided")
    if gacode_data is None:
        gacode_data = read_gacode(gacode_filename_for_profiles)
    #########################################################################################################
    ### Extract Electron info with unit checks:
    #########################################################################################################
    psi_normalised = np.array(gacode_data['polflux']/gacode_data['polflux'][-1]) # Normalized poloidal flux
    ne_prof = np.array(gacode_data['ne']) 
    # Check units:
    ne_units = gacode_data.get('ne_units', None)
    ne_prof = gacode_density_unit_check(ne_prof, ne_units)

    te_prof = np.array(gacode_data['te'])
    # Check units:
    te_units = gacode_data.get('te_units', None)
    te_prof = gacode_temperature_unit_check(te_prof, te_units)

    # Make into xarray DataArrays:
    te_da = xr.DataArray(te_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='te', attrs={'units': 'keV'})
    ne_da = xr.DataArray(ne_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ne', attrs={'units': 'm^-3'})
    profiles_xr = xr.Dataset({'te': te_da, 'ne': ne_da})

    #########################################################################################################
    ### Extract Ion info with unit checks:
    #########################################################################################################
    ion_labels=[]
    ni_profs=[]
    ti_profs=[]
    ion_datasets = []
    # Check if 'ni' is a dictionary (multiple ion species) or array (single ion species):
    if gacode_data['ni'] is None:
        raise ValueError("ni data not found in GACODE file")
    if isinstance(gacode_data['ni'], np.ndarray):
        # Single ion species case:
        if 'name' in gacode_data:
            # We assume it's a single ion species with this name:
            ion_labels.append(gacode_data['name'])
        ni_prof = np.array(gacode_data['ni'])
        # Check units:
        ni_units = gacode_data.get('ni_units', None)
        ni_prof = gacode_density_unit_check(ni_prof, ni_units)
        ni_profs.append(ni_prof)
        ti_prof = np.array(gacode_data['ti'])
        # Check units:
        ti_units = gacode_data.get('ti_units', None)
        ti_prof = gacode_temperature_unit_check(ti_prof, ti_units)
        ti_profs.append(ti_prof)
        ni_da = xr.DataArray(ni_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ni', attrs={'units': 'm^-3'})
        ti_da = xr.DataArray(ti_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ti', attrs={'units': 'keV'})
        charge = gacode_data.get('z', 1.0)
        mass_amu = gacode_data.get('mass', 1.0)
        ion_ds = xr.Dataset({'ni': ni_da, 'ti': ti_da, 'charge': charge, 'mass_amu': mass_amu})
        ion_datasets.append(ion_ds)
    elif isinstance(gacode_data['ni'], dict):
        # Multiple ion species case:
        # We make the sub-labels: col_1, col_2, ... for each key in the dictionary
        for key in gacode_data['ni']:
            if gacode_data['name'][key] in exclude_species:
                continue
            ion_labels.append(gacode_data['name'][key])
            ni_prof = np.array(gacode_data['ni'][key])
            # Check units:
            ni_units = gacode_data.get('ni_units', None)
            ni_prof = gacode_density_unit_check(ni_prof, ni_units)
            ni_profs.append(ni_prof)
            ti_prof = np.array(gacode_data['ti'][key])
            # Check units:
            ti_units = gacode_data.get('ti_units', None)
            ti_prof = gacode_temperature_unit_check(ti_prof, ti_units)
            ti_profs.append(ti_prof)
            ni_da = xr.DataArray(ni_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ni', attrs={'units': 'm^-3'})
            ti_da = xr.DataArray(ti_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ti', attrs={'units': 'keV'})
            charge = gacode_data['z'][key]
            mass_amu = gacode_data['mass'][key]
            ion_ds = xr.Dataset({'ni': ni_da, 'ti': ti_da, 'charge': charge, 'mass_amu': mass_amu})
            ion_datasets.append(ion_ds)
    else:
        raise ValueError("ni data in GACODE file is neither an array nor a dictionary, something is wrong")
    
    # Concatenate all ion datasets along species dimension:
    ion_ds_all = xr.concat(ion_datasets, dim='species')
    ion_ds_all = ion_ds_all.assign_coords({'species': ion_labels})
    # Merge with profiles_xr:
    profiles_xr = xr.merge([profiles_xr, ion_ds_all])

    #########################################################################################################
    ### Check all profiles have same length:
    #########################################################################################################
    len_prof = len(psi_normalised)
    if len(ne_prof) != len_prof:
        raise ValueError("ne profile length does not match psi length")
    if len(te_prof) != len_prof:
        raise ValueError("te profile length does not match psi length")
    for i in range(len(ion_labels)):
        if len(ni_profs[i]) != len_prof:
            raise ValueError("ni profile length for ion "+ion_labels[i]+" does not match psi length")
        if len(ti_profs[i]) != len_prof:
            raise ValueError("ti profile length for ion "+ion_labels[i]+" does not match psi length")
    
    #########################################################################################################
    ### Create total ion mass profile and ion number density profiles. Also check quasineutrality (full ionization, not accurate for high-Z impurities):
    #########################################################################################################

    ne_check = np.zeros(len_prof)
    total_ion_mass_prof = np.zeros(len_prof)
    total_ion_density_prof = np.zeros(len_prof)
    for i in range(len(ion_labels)):
        ne_check += ni_profs[i]*profiles_xr['charge'].sel(species=ion_labels[i]).values
        total_ion_mass_prof += ni_profs[i]*profiles_xr['mass_amu'].sel(species=ion_labels[i]).values
        total_ion_density_prof += ni_profs[i]

    #########################################################################################################
    ### See if sum of ion densities times ion charges matches electron density times -1:
    #########################################################################################################

    if np.max(np.abs(ne_check - ne_prof)/ne_prof) > 0.01:
        print(np.abs(ne_check - ne_prof)/ne_prof)
        print("On-axis charge density for electrons = "+str(ne_prof[0]))
        for i in range(len(ion_labels)):
            print("On-axis charge density for ion "+ion_labels[i]+" = "+str((ni_profs[i]*profiles_xr['charge'].sel(species=ion_labels[i]).values)[0])," assuming complete ionization with charge "+str(profiles_xr['charge'].sel(species=ion_labels[i]).values))
        print("Sum of on-axis ion charge densities = "+str(ne_check[0]))
        if len(ion_labels) == 1:
            raise ValueError("Quasineutrality check failed for a single-ion-species plasma, something is wrong")

    #########################################################################################################
    # Total pressure:
    #########################################################################################################

    total_electron_pressure_prof = ne_prof*te_prof*kBoltz*1.E3 # # 1.602e-19 * [m^-3] * [eV] = [Pa]
    # Compute total ion pressure from individual species:
    total_ion_pressure_prof = np.zeros(len_prof)
    for i in range(len(ion_labels)):
        total_ion_pressure_prof += ni_profs[i]*ti_profs[i]*kBoltz*1.E3 # 1.602e-19 * [m^-3] * [eV] = [Pa]
        if debug:
            print("On-axis ion pressure for ion "+ion_labels[i]+" = "+str((ni_profs[i]*ti_profs[i]*kBoltz*1.E3)[0])+" Pa")
    if debug:
        print("On-axis ion pressure for electrons = "+str((ne_prof*te_prof*kBoltz*1.E3)[0])+" Pa")
    # Total thermal pressure:
    total_thermal_pressure_prof = total_ion_pressure_prof + total_electron_pressure_prof
    # Check this is consistent with pressure from gacode_data:
    p_profiles_1d = np.array(gacode_data['ptot'])
    assert gacode_data['ptot_units'] == 'Pa', "Currently only Pa units for ptot supported"
    assert len(total_thermal_pressure_prof) == len(p_profiles_1d) 
    if np.max(np.abs(total_thermal_pressure_prof[0:10] - p_profiles_1d[0:10])/p_profiles_1d[0:10]) > 0.01: 
        print(
            "Warning: computed on-axis thermal pressure from kinetic profiles = "+str(total_thermal_pressure_prof[0:10])+" Pa does not match pax from core_profiles_1d = "+str(p_profiles_1d[0:10])+" Pa, relative difference = "+str(np.abs(total_thermal_pressure_prof[0:10] - p_profiles_1d[0:10])/p_profiles_1d[0:10])
        )
        if debug:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.plot(psi_normalised,total_thermal_pressure_prof,label='Kinetic pressure')
            plt.plot(psi_normalised,p_profiles_1d,label='total pressure')
            plt.legend()
            plt.savefig('gacode_pressure_comparison')
            raise ValueError("Error: computed on-axis thermal pressure from kinetic profiles is inconsistent")

    if return_thermal_pressure:
        return psi_normalised,total_thermal_pressure_prof

    #########################################################################################################
    # Run single_species_simplification if requested:
    #########################################################################################################

    if not single_species_simplification:
        profiles_xr = profiles_xr.assign_attrs({'single_species_model': False})
        return profiles_xr
    elif single_species_simplification and not mygs:
        print("Warning: single_species_simplification requested but mygs not provided, cannot compute flux surface averages, returning multi-species profiles only")
        return profiles_xr
    
    total_electron_density = mygs.flux_integral(psi_normalised,ne_prof) # in m^-3
    total_ion_density = mygs.flux_integral(psi_normalised,total_ion_density_prof) # in m^-3
    profiles_xr = profiles_xr.assign_attrs({'total_electron_density_m3': total_electron_density, 'total_ion_density_m3': total_ion_density})
    if len(ion_labels) == 1:
        ss_ion_charge = profiles_xr['charge'].sel(species=ion_labels[0]).values
        ss_ion_mass = profiles_xr['mass_amu'].sel(species=ion_labels[0]).values
        profiles_xr = profiles_xr.assign({'ss_ion_charge': ss_ion_charge, 'ss_ion_mass': ss_ion_mass})
        profiles_xr['ss_ion_density'] = profiles_xr['ni'].sel(species=ion_labels[0])
        profiles_xr['ss_ion_temp'] = profiles_xr['ti'].sel(species=ion_labels[0])
        total_ion_pressure_prof = (profiles_xr['ss_ion_density']*profiles_xr['ss_ion_temp']*kBoltz*1.E3).values # 1.602e-19 * [m^-3] * [eV] = [Pa]
        total_thermal_pressure_prof = total_ion_pressure_prof + total_electron_pressure_prof
    else:
        ### Effective charge:
        ion_effective_charge = total_electron_density/total_ion_density
        ### Effective mass:
        ion_effective_mass = mygs.flux_integral(psi_normalised,total_ion_mass_prof)/total_ion_density
        ### Add to profiles_xr:
        profiles_xr = profiles_xr.assign({'ss_ion_charge': ion_effective_charge, 'ss_ion_mass': ion_effective_mass})
        #########################################################################################################
        ### If the underlying ion species have different temperature profiles, we need to adjust to make sure the single-species Ti matches the pressure:
        #########################################################################################################
        # Now compute the Ti profile that would give this total thermal pressure assuming a single ion species with density total_ion_density_prof:
        ti_equiv_prof = (total_thermal_pressure_prof - total_electron_pressure_prof)/(total_ion_density_prof*kBoltz*1.E3) # in keV
        profiles_xr['ss_ion_temp'] = xr.DataArray(ti_equiv_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ss_ion_temp', attrs={'units': 'keV'})
        profiles_xr['ss_ion_density'] = xr.DataArray(total_ion_density_prof, coords=[psi_normalised], dims=['psi_n_1d'], name='ss_ion_density', attrs={'units': 'm^-3'})

    #########################################################################################################
    # Sanity checks on single species quasi neutrality, and pressure consistency:
    #########################################################################################################
    # Check quasi-neutrality:
    if np.max(np.abs(profiles_xr['ss_ion_density']*profiles_xr['ss_ion_charge'] - profiles_xr['ne'])/profiles_xr['ne']) > 0.01:
        raise ValueError("Quasineutrality check failed for single-ion-species model, something is wrong")
    # Check pressure consistency:
    ss_ion_pressure_prof = profiles_xr['ss_ion_density']*profiles_xr['ss_ion_temp']*kBoltz*1.E3 # 1.602e-19 * [m^-3] * [eV] = [Pa]
    total_thermal_pressure_prof_ss = ss_ion_pressure_prof + total_electron_pressure_prof
    if np.max(np.abs(total_thermal_pressure_prof_ss - total_thermal_pressure_prof)/total_thermal_pressure_prof) > 0.01:
        raise ValueError("Pressure consistency check failed for single-ion-species model, something is wrong")

    profiles_xr = profiles_xr.assign_attrs({'single_species_model': True})
    if eta_fac is not None:
        profiles_xr['eta_fac'] = eta_fac
    if Coulomb_logarithm is not None:
        profiles_xr['Coulomb_logarithm'] = Coulomb_logarithm
    return profiles_xr

def gacode_temperature_unit_check(temperature_array, temperature_units):
    """ Check and convert temperature units to keV """
    if temperature_units is not None:
        if 'kev' in temperature_units.lower():
            pass
        elif 'ev' in temperature_units.lower():
            temperature_array /= 1.E3
        else:
            print("Warning: assuming temperature units are keV")
    else:
        print("Warning: temperature units not found, assuming keV")
    return temperature_array

def gacode_density_unit_check(density_array, density_units):
    """ Check and convert density units to m^-3 """
    if density_units is not None:
        if '20' in density_units and 'm' in density_units:
            density_array *= 1.E20
        elif '19' in density_units and 'm' in density_units:
            density_array *= 1.E19
        elif '18' in density_units and 'm' in density_units:
            density_array *= 1.E18
        elif '17' in density_units and 'm' in density_units:
            density_array *= 1.E17
        elif '16' in density_units and 'm' in density_units:
            density_array *= 1.E16
        elif '15' in density_units and 'm' in density_units:
            density_array *= 1.E15
        elif '14' in density_units and 'm' in density_units:
            density_array *= 1.E14
        else:
            print("Warning: assuming density units are m^-3")
    else:
        print("Warning: density units not found, assuming m^-3")
    return density_array

def read_gacode(gacode_filename,verbose=False):
    """
        Read input.gacode file and return a dictionary containing all data.
        Parameters:
        -----------
        gacode_filename : str
            Path to the input.gacode file
        Returns:
        --------
        data : dict
            Dictionary containing all scalars and vectors from the file
    """
    data = {}
    with open(gacode_filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip comments and empty lines
        if (not line) or (not line.strip().startswith('#')):
            #print("stripped line:", line.strip()," T/F:", line.strip().startswith('#'))
            #print("Skipping line:", line)
            i += 1
            continue
        #i += 1
        # Check if this is a data label (starts with '# ' followed by a name)
        # Check if first non-whitespace character is '#'
        if line.strip().startswith('#'):
            if len(line.strip()) > 1:
                # Check what second non-whitespace character is
                line_without_hash = line.strip()[1:].lstrip()
                if line_without_hash.startswith('*'):
                    # It's a comment line, skip
                    #print("Skipping comment line:", line)
                    i += 1
                    continue
                if line_without_hash.isspace():
                    # It's an empty line, skip
                    i += 1
                    continue
            if len(line.strip()) == 1:
                # It's just a '#' line, skip
                i += 1
                continue
            # It's a data label
            #print("Processing line:", line)
            # Extract the label and units if present
            parts = line[2:].split('|')
            label = parts[0].strip()
            units = parts[1].strip() if len(parts) > 1 else None
            #print(f"Reading label: {label} with units: {units}")
            # Move to next line for data
            if verbose: print("Processing line:", lines[i])
            i += 1
            if verbose: print("Data line:", lines[i])
            # Read data lines until we hit another comment or empty line
            data_lines = []
            while i < len(lines):
                data_line = lines[i].strip()
                if (not data_line) or data_line.startswith('#'):
                    if verbose: print("NONSCALAR value for label", label, ":", lines[i], ":", lines[i].strip())
                    break
                data_lines.append(data_line)
                i += 1
            # Parse the data
            if verbose: print('data_lines:', data_lines, ' len:', len(data_lines))
            if len(data_lines) == 1:
                # Single value - check if it's scalar or vector
                values = data_lines[0].split()
                if verbose: print("values:", values)
                if len(values) == 1:
                    # Scalar value
                    if verbose: print("SCALAR value for label", label, ":", values[0])
                    try:
                        data[label] = float(values[0])
                    except ValueError:
                        data[label] = values[0]  # Keep as string if not a number
                else:
                    # Single line with multiple values (like name or type)
                    # Make a sub-dictionary with keys col_1, col_2, ..., where each key corresponds to that column's value
                    value_dict = {}
                    for idx, val in enumerate(values):
                        try:
                            value_dict[f'col_{idx+1}'] = float(val)
                        except ValueError:
                            value_dict[f'col_{idx+1}'] = val  # Keep as string if not a number
                    data[label] = value_dict
                    if not (units is None):
                        data[f'{label}_units'] = units
                    if verbose: print(" MULTICOLUMN value for label", label, ":", value_dict)
            else:
                # Multiple lines - vector data
                if verbose: print("MULTILINE value for label", label, ":", data_lines)
                if all(line_.split()[0].isdigit() for line_ in data_lines):
                    # Data with indices (like rho, ne, te, etc.)
                    if len(data_lines[0].split()) < 2:
                        if verbose: print(" skipping unrecognized line_:", line_)
                        continue
                    elif len(data_lines[0].split()) == 2:
                        # Two columns - single vector
                        indices = []
                        values = []
                        for line__ in data_lines:
                            parts = line__.split()
                            indices.append(int(parts[0]))
                            values.append(float(parts[1]))
                        data[label] = np.array(values)
                    else:
                        # More than two columns - multiple vectors. make a sub-dictionary with keys 1, 2, ..., 
                        # where each key corresponds to a numpy array of that column's values
                        data[label] = {}
                        for l in range(1, len(data_lines[0].split())):
                            col_values = []
                            for line__ in data_lines:
                                parts = line__.split()
                                col_values.append(float(parts[l]))
                            data[label][f'col_{l}'] = np.array(col_values)
                else:
                    # Data without indices
                    if verbose: print(" DATA WITHOUT INDICES for label", label, ":", data_lines)
                    data[label] = [line__ for line__ in data_lines]
            # Store units as metadata if present
            if not (units is None):
                data[f'{label}_units'] = units
        else:
            if verbose: print(" skipping unrecognized line:", line)
            i += 1
    return data

def get_profs(eqdsk_w_profiles,normalise=True,set_edge_zero=True,len_psi_sample=257,use_cubic_splines=False):
    ffprim = eqdsk_w_profiles['ffprim']
    pprime = eqdsk_w_profiles['pprime']
    psi_eqdsk = np.linspace(0.0,1.0,np.size(ffprim))
    # Check minimimum psi spacing:
    min_psi_diff = np.min(np.diff(psi_eqdsk))
    psi_sample = np.linspace(0.0,1.0,len_psi_sample)
    psi_prof = np.copy(psi_sample)
    if not use_cubic_splines:
        ffp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_eqdsk,ffprim)))).copy()
        pp_prof = np.transpose(np.vstack((psi_prof,np.interp(psi_sample,psi_eqdsk,pprime)))).copy()
    else:
        from scipy.interpolate import CubicSpline
        ffp_spline = CubicSpline(psi_eqdsk,ffprim,extrapolate=True)
        pp_spline = CubicSpline(psi_eqdsk,pprime,extrapolate=True)
        ffp_prof = np.transpose(np.vstack((psi_prof,ffp_spline(psi_sample)))).copy()
        pp_prof = np.transpose(np.vstack((psi_prof,pp_spline(psi_sample)))).copy()
    ffp_prof={'type': 'linterp', 'y': ffp_prof[:,1], 'x': psi_sample}
    pp_prof={'type': 'linterp', 'y': pp_prof[:,1], 'x': psi_sample}
    if normalise:
        ffp_prof['y'] /= np.abs(ffp_prof['y']).max() # Normalize profile (not required but convienient)
        pp_prof['y'] /= np.abs(pp_prof['y']).max() # Normalize profile (not required but convienient)
        if ffp_prof['y'][0]<0.0:
            ffp_prof['y'] *= -1.0 # Make sure ffp is positive at axis
        if pp_prof['y'][0]<0.0:
            pp_prof['y'] *= -1.0 # Make sure pp is positive at axis
    if set_edge_zero:
        ffp_prof['y'][-1] = 0.0 # Set edge to zero
        pp_prof['y'][-1] = 0.0 # Set edge to zero
    return ffp_prof, pp_prof

def one_well_hill_solver(mygs,wellhill_vec,
            psi_orig,
            jtor_abs_orig,
            q_orig,
            device_dict_orig,
            npsi=257,
            psi_pad=1.E-4,
            max_its=20,tol=0.002,
            nl_tol=1E-6,
            keep_same_Ip=True,
            q0lim=None):
    """ Solve equilibrium with a vector of well-hill current perturbations, adjusting to hit target q surfaces """

    #########################################################################################################
    # Check distance to nearest rational surfaces and reduce width of well-hill if necessary (and if 
    # variable=True):
    #########################################################################################################
    for i in wellhill_vec:
        new_width = check_distance_to_nearest_rationals(mygs, i['m'], i['n'], i['width'], psi_orig, q_orig)
        if i['variable']:
            i['width'] = new_width

    #########################################################################################################
    # Check if psi_orig is narrow enough:
    #########################################################################################################
    min_psi_orig_diff = np.min(np.diff(psi_orig))
    goal_min_psi_diff = None
    #What is the minimum width of the well-hills?
    widths = [i['width'] for i in wellhill_vec]
    min_width = min(widths)
    #Is this less than 3x the minimum psi grid spacing?
    if min_width < 3.0*min_psi_orig_diff:
        goal_min_psi_diff = min_width/3.0
    if goal_min_psi_diff is not None:
        print("Warning: minimum width of well-hill is very narrow compared to psi grid, consider increasing psi resolution to at least "+str(goal_min_psi_diff))
        return None,False
    # Convert goal_min_psi_diff to npsi:
    if goal_min_psi_diff is not None:
        goal_npsi = int(1.0/goal_min_psi_diff)+1
        npsi = max(npsi,goal_npsi)
            

    #########################################################################################################
    # Set up initial conditions:
    #########################################################################################################
    i = 0
    qvals = np.array([(i['m'])/i['n'] for i in wellhill_vec])
    psifsurfs = get_psis_from_qs(qvals,psi_orig,q_orig)
    # Check qvals are unique:
    if len(qvals) != len(set(qvals)):
        print("Error: q values for well-hills are not unique")
        raise ValueError("Non-unique q values found")

    #########################################################################################################
    # Run iteration:
    #########################################################################################################
    while i <= max_its:
        jtor = jtor_abs_orig.copy()
        for iw, w in enumerate(wellhill_vec):
            gaussian_bump = gaussian(psi_orig, amp=w['amp'], center=psifsurfs[iw], width=w['width'], offset=w['offset'], sk=w['skew'])
            jtor += gaussian_bump*np.max(jtor_abs_orig)
        jtor_prof = {'type': 'jphi-linterp', 'x': psi_orig, 'y': jtor}
        jtor_prof['y'] = jtor_prof['y']/np.max(jtor)
        mygs.set_profiles(ffp_prof=jtor_prof, pp_prof=device_dict['pp_prof'])
        mygs.settings.pm=True
        mygs.settings.nl_tol=nl_tol
        mygs.update_settings()
        if not keep_same_Ip:
            new_Ip = mygs.flux_integral(psi_orig,jtor)
            mygs.set_targets(Ip=new_Ip,pax=device_dict_orig['pax_target'])
            print("Ip adjusted to "+str(new_Ip)+" (compared to original target of "+str(device_dict_orig['Ip_target'])+") based on wellhill current")
        else:  
            mygs.set_targets(Ip=device_dict_orig['Ip_target'],pax=device_dict_orig['pax_target'])
        err_flag = mygs.solve()
        psi_qs,qs,ravgs,_,_,_ = mygs.get_q(npsi=npsi,psi_pad=psi_pad)
        #New psi_surf:
        new_psifsurfs = get_psis_from_qs(qvals,psi_qs,qs)
        print("Iteration "+str(i)+": q_surf = "+str(qvals)+" at psi = "+str(new_psifsurfs)+" (targets were "+str(psifsurfs)+")")
        if np.all(np.abs(new_psifsurfs - psifsurfs) < tol):
            print("Converged to within "+str(tol)+" in psi, finishing")
            return mygs, True
        else:
            psifsurfs = psifsurfs+0.2*(new_psifsurfs-psifsurfs)
        print("Iteration "+str(i))
        i += 1
    
    #########################################################################################################
    # Check if converged:
    #########################################################################################################
    if i > max_its:
        print("Warning: did not converge within "+str(max_its)+" iterations")
        print("  Final psi surfaces were "+str(new_psifsurfs)+" (targets were "+str(psifsurfs)+"), tolerance was "+str(tol))
        worked = False
    else:
        worked = True

    if q0lim is not None:
        if qs[0] < q0lim:
            print("Warning: q0 = "+str(qs[0])+" is below limit of "+str(q0lim)+", rejecting solution")
            worked = False
    
    return mygs, worked

def multi_well_hill_generation_names(filename_prefac,surf_vec,amps_vec):
    eq_list = []
    for amps in amps_vec:
        #Convert list of amps to string:
        amps_str = "_".join([str(a).replace('.','p') for a in amps])
        #Convert list of surf_vec to string:
        m_n_str = "_".join([str(surf['m'])+"m"+str(surf['n'])+"n" for surf in surf_vec])
        filename = filename_prefac+amps_str+"_"+m_n_str
        if os.path.exists(filename+".eqdsk"):
            eq_list.append(filename+".eqdsk")
    return eq_list

def multi_well_hill_generation(mygs,device_dict_orig,filename_prefac,surf_vec,amps_vec,plot_dir,npsi=257,psi_pad=1.E-4,hotstart=False,return_eq_list=False,profiles_xr=None,**kwargs):
    '''
    Provide list of surf_vec of the form [surf1,surf2,...] where each surfi is a dictionary {'m': 2, 'n': 1, 'width': 0.01, 'skew': 0.0, 'offset': 0.0, 'variable': False}
    Provide a list of amplitudes to amps_vec, one for each equilibrium run:
        Ie amps_vec = [[0.0,0.1], [0.0,0.2], [0.0,0.3], ...] where each sublist has the same length as surf_vec
    '''
    psi_orig,f,fp,p,pp,R_avg,one_over_R_avg,jtor,ffprime,ravgs,qorig = get_all_profiles(mygs, psi_pad=psi_pad, npsi=npsi)

    worked_vec = []
    eq_list = []
    for amps in amps_vec:
        #Convert list of amps to string:
        amps_str = "_".join([str(a).replace('.','p') for a in amps])
        #Convert list of surf_vec to string:
        m_n_str = "_".join([str(surf['m'])+"m"+str(surf['n'])+"n" for surf in surf_vec])
        filename = filename_prefac+amps_str+"_"+m_n_str

        if return_eq_list:
            if os.path.exists(filename+".eqdsk"):
                eq_list.append(filename+".eqdsk")
            #continue

        # Check if file is present:
        if hotstart and os.path.exists(filename+".eqdsk"):
            print(" Skipping "+filename+", already exists")
            continue
        if del_existing and os.path.exists(filename+".eqdsk"):
            os.remove(filename+".eqdsk")
            print(" Deleted existing "+filename)

        if len(amps) != len(surf_vec):
            raise ValueError("Length of amps_vec sublist does not match length of surf_vec")
        wellhill_vec = []
        for i,surf in enumerate(surf_vec):
            wellhill = {}
            wellhill['m'] = surf['m']
            wellhill['n'] = surf['n']
            wellhill['amp'] = amps[i]
            wellhill['skew'] =  surf.get('skew',0.0)
            wellhill['variable'] = surf.get('variable',True)
            wellhill['offset'] = surf.get('offset',0.0)
            wellhill['width'] = surf.get('width',0.01)
            wellhill_vec.append(wellhill)
        my_gs, worked = one_well_hill_solver(mygs,wellhill_vec,psi_orig,jtor,qorig,device_dict_orig,npsi=npsi,**kwargs)
        worked_vec.append(worked)
        if worked:
            psi_q,qs,ravgs,_,_,_ = mygs.get_q(npsi=10,psi_pad=0.005)
            mygs.save_eqdsk(filename+".eqdsk",nr=257,nz=257,lcfs_pad=0.0001)
            print("Saved equilibrium to "+filename+".eqdsk", "q0 = ",qs[0])
            if profiles_xr is not None:
                # Save kinetic profiles to netcdf:
                profiles_xr.to_netcdf(filename+"_profiles.nc")
            # Save figs:
            easy_plot(mygs,filename,plot_dir,profiles_xr=profiles_xr)

    #if return_eq_list:
    #    return eq_list
    if return_eq_list:
        return mygs,worked_vec,eq_list
    return mygs,worked_vec

def check_distance_to_nearest_rationals(mygs,m,n,width,psi_orig,q_orig,width_scale=0.9):
    """ Check distance to nearest rational surfaces and reduce width of well-hill if necessary """
    qsurf_vals = [(m-1)/n,(m)/n,(m+1)/n]
    psi_vals = get_psis_from_qs(qsurf_vals,psi_orig,q_orig)
    psi_val_diffs = np.diff(psi_vals)
    print("m, n = ", m, n, ": difference between surfaces = ", psi_val_diffs)
    if width > min(psi_val_diffs)/2.0:
        print("Warning: width is too large, reducing to fit between surfaces")
        width = width_scale*min(psi_val_diffs)/2.0
    return width