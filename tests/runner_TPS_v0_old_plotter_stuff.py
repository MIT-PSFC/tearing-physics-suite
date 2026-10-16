
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 1,
    },
    where_option_dict={
        'r_value': 2.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
        'r_unique': True,
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
    debug=False
)
total_finite = sum(np.isfinite(np.asarray(vals, dtype=float)).sum() for vals in plot_input['value_matrix'])
print(f"Total finite values in value_matrix: {total_finite}")
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 99],
    #y_range=[-50,50],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_21_unique'
)
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 1,
    },
    where_option_dict={
        'r_value': 2.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
total_finite = sum(np.isfinite(np.asarray(vals, dtype=float)).sum() for vals in plot_input['value_matrix'])
print(f"Total finite values in value_matrix: {total_finite}")
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 99],
    #y_range=[-50,50],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_21_all'
)



plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    'q',
    sel_option_dict={
        'code': 'rdcon',
        'nn': 1,
    }
)
prof_plotter('/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs','avg_mu0Jpara', times, xri_list)


# RDCON BATCH

var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 1,
    },
    where_option_dict={
        'r_value': 2.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_21'
)




var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 1,
    },
    where_option_dict={
        'r_value': 2.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_21_str'
)

var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 2,
    },
    where_option_dict={
        'r_value': 3.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_32'
)


var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 3,
    },
    where_option_dict={
        'r_value': 4.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 99],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_43'
)


var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 4,
    },
    where_option_dict={
        'r_value': 5.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 99],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_54'
)



var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 1,
    },
    where_option_dict={
        'r_value': 2.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_21'
)

var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 2,
    },
    where_option_dict={
        'r_value': 3.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_32'
)

var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 3,
    },
    where_option_dict={
        'r_value': 4.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_43'
)


var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'rdcon',
        'nn': 4,
    },
    where_option_dict={
        'r_value': 5.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_54'
)



# STRIDE BATCH:

var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 2,
    },
    where_option_dict={
        'r_value': 3.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_32_str'
)


var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 3,
    },
    where_option_dict={
        'r_value': 4.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 99],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_43_str'
)


var_name='Delta_prime_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 4,
    },
    where_option_dict={
        'r_value': 5.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 99],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='DP_54_str'
)



var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 1,
    },
    where_option_dict={
        'r_value': 2.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_21_str'
)

var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 2,
    },
    where_option_dict={
        'r_value': 3.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_32_str'
)

var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 3,
    },
    where_option_dict={
        'r_value': 4.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_43_str'
)


var_name='dwdtau_max_surf'
plot_input = dist_extractor(
    '/cscratch/benjamins/TMDB_v0/files/Laszlo_v0/test3/TPS_run_1/TPS_figs',
    xri,
    #'w_marg_surf',
    var_name,
    sel_option_dict={
        'code': 'stride',
        'nn': 4,
    },
    where_option_dict={
        'r_value': 5.,
        # list => multiple conditions on the same coordinate
        'psi_n_rational': [
            {'op': '<',  'val': 0.95},
        ],
    },
    isel_option_dict={
        'Delta_prime_type': 0,
    },
)
fig, ax, outputs = dist_plotter(
    **plot_input,
    percent_bounds=[1, 100],
    n_bins=250,
    point_size=3,
    angle_deg=0.0,
    range_buffer=0.3,
    save_name='dwdtmax_54_str'
)

