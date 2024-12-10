import time
import json
import logging
from copy import deepcopy

import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO,
)

from openquake.fnm.fault_modeler import (
    get_subsections_from_fault,
    simple_fault_from_feature,
    make_subfault_df,
    make_rupture_df,
)

from openquake.fnm.rupture_connections import (
    get_rupture_adjacency_matrix,
    get_multifault_ruptures,
    get_multifault_ruptures_fast,
    get_multifault_ruptures_numba,
    make_binary_adjacency_matrix,
    make_binary_adjacency_matrix_sparse,
    filter_bin_adj_matrix_by_rupture_angle,
)

from openquake.fnm.rupture_filtering import (
    get_rupture_plausibilities,
    filter_proportionally_to_plausibility,
)

from openquake.fnm.inversion.utils import (
    rup_df_to_rupture_dicts,
    subsection_df_to_fault_dicts,
    SHEAR_MODULUS,
    get_rup_rates_from_fault_slip_rates,
)

from openquake.fnm.inversion.soe_builder import make_eqns
from openquake.fnm.inversion.simulated_annealing import simulated_annealing

from openquake.fnm.exporter import (
    make_multifault_source,
    write_multifault_source,
)

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

default_settings = {
    'subsection_size': [15.0, 15.0],
    'edge_sd': 5.0,
    'dip_sd': 5.0,
    'max_jump_distance': 10.0,
    'max_sf_rups_per_mf_rup': 10,
    'rupture_angle_threshold': 60.0,
    'filter_by_plausibility': True,
    'filter_by_angle': True,
    'rupture_filtering_connection_distance_plausibility_threshold': 0.1,
    'skip_bad_faults': False,
    'shear_modulus': SHEAR_MODULUS,
    'fault_mfd_b_value': 1.0,
    'fault_mfd_type': 'TruncatedGRMFD',
    'seismic_fraction': 0.7,
    'rupture_set_for_rates_from_slip_rates': 'all',
    'plot_fault_moment_rates': False,
    'sparse_distance_matrix': True,
    'parallel_multifault_search': False,
    'full_fault_only_mf_ruptures': True,
    'calculate_rates_from_slip_rates': True,
    'surface_type': 'simple',
    'min_mag': None,
    'max_mag': None,
    "filter_seed": 69,
}


def build_fault_network(
    faults=None,
    fault_geojson=None,
    settings=None,
    return_faults_only=False,
    **kwargs,
):
    """
    Build a fault network from a list of faults or a fault geojson file. This
    is the main data preparatory step for building a fault-based seismic source
    model.

    Parameters
    ----------
    faults : list of fault dictionaries, optional
        List of faults in dictionary format. The default is None.
    fault_geojson : str, optional
        Path to a fault geojson file. The default is None.
    settings : dict, optional
        Settings for building the fault network. The default is None.
    surface_type : str, optional
        Type of surface to build from a fault. The default is 'simple'.
    filter_by_angle : bool, optional
        Whether to filter the fault network by rupture angle. The default is
        True.
    filter_by_plausibility : bool, optional
        Whether to filter the fault network by rupture plausibility. The
        default is True.
    **kwargs : dict
        Additional settings. These will overwrite the settings provided in the
        settings dictionary.

    Returns
    -------
    fault_network : dict
        Dictionary containing the fault network and rupture data. The keys are:
            - 'faults': list of fault dictionaries
            - 'subfaults': list of subfault dictionaries
            - 'single_rup_df': DataFrame of single-fault ruptures
            - 'dist_mat': continuous distance matrix
            - 'subfault_df': DataFrame of subfaults
            - 'multifault_inds': list of multifault rupture indices
            - 'rupture_df': DataFrame of all ruptures
            - 'plausibility': DataFrame of rupture plausibilities
            - 'rupture_df_keep': DataFrame of ruptures after filtering
    """
    build_settings = deepcopy(default_settings)
    if settings is not None:
        build_settings.update(settings)
    build_settings.update(kwargs)

    settings = build_settings

    fault_network = {}

    event_times = []

    t0 = time.time()
    event_times.append(t0)
    if faults is None:
        if settings['surface_type'] == 'simple':
            build_surface = simple_fault_from_feature
        else:
            raise NotImplementedError(
                f'Surface type {settings["surface_type"]} not implemented'
            )

        if fault_geojson is not None:
            logging.info("Building faults from geojson")
            with open(fault_geojson) as f:
                fault_gj = json.load(f)
            faults = []
            fault_fids = []
            for feature in fault_gj['features']:
                try:
                    surf = build_surface(
                        feature,
                        edge_sd=settings['edge_sd'],
                    )
                    faults.append(surf)
                    fault_fids.append(feature['properties']['fid'])
                except Exception as e:
                    logging.error(
                        f"Cannot build fault {feature['properties']['fid']}"
                    )
                    if settings["skip_bad_faults"] is True:
                        logging.error(
                            f"\tskipping fault {feature['properties']['fid']}"
                        )
                        logging.error(f"\t{e}")
                    else:
                        raise e

            duplicated_fids = [
                fid for fid in set(fault_fids) if fault_fids.count(fid) > 1
            ]
            if len(duplicated_fids) > 0:
                raise ValueError(f'Duplicated fault fids: {duplicated_fids}')
            logging.info(f"\t{len(faults)} faults built from geojson")

        else:
            raise ValueError('No faults provided')
    fault_network['faults'] = faults
    t1 = time.time()
    event_times.append(t1)
    logging.info(f"\tdone in {round(t1-t0, 1)} s")
    if return_faults_only:
        return fault_network

    logging.info("Making subfaults")
    fault_network['subfaults'] = []
    for i, fault in enumerate(faults):
        try:
            fault_network['subfaults'].append(
                get_subsections_from_fault(
                    fault,
                    subsection_size=build_settings['subsection_size'],
                    edge_sd=build_settings['edge_sd'],
                    dip_sd=build_settings['dip_sd'],
                    surface=fault['surface'],
                )
            )
        except Exception as e:
            logging.error(f"Error with fault {i}: {e}")
            # yield fault_network
            raise e
            # return faults

    n_subfaults = sum([len(sf) for sf in fault_network['subfaults']])
    t2 = time.time()
    event_times.append(t2)
    logging.info(f"\tdone in {round(t2-t1, 1)} s")
    logging.info(f"\t{n_subfaults} subfaults from {len(faults)} faults")

    logging.info("Making single fault rup df and distance matrix")
    (
        fault_network['single_rup_df'],
        fault_network['dist_mat'],
    ) = get_rupture_adjacency_matrix(
        faults,
        all_subfaults=fault_network['subfaults'],
        max_dist=settings['max_jump_distance'],
        sparse=settings['sparse_distance_matrix'],
        full_fault_only_mf_ruptures=settings['full_fault_only_mf_ruptures'],
    )
    t3 = time.time()
    event_times.append(t3)
    logging.info(f"\tdone in {round(t3-t2, 1)} s")
    logging.info(
        f"\t{'{:,}'.format(len(fault_network['single_rup_df']))} "
        + "single-fault ruptures"
    )

    if settings['sparse_distance_matrix'] is True:
        binary_adjacence_matrix = make_binary_adjacency_matrix_sparse(
            fault_network['dist_mat'], max_dist=settings['max_jump_distance']
        )
    else:
        binary_adjacence_matrix = make_binary_adjacency_matrix(
            fault_network['dist_mat'], max_dist=settings['max_jump_distance']
        )

    n_connections = binary_adjacence_matrix.sum()
    n_possible_connections = len(fault_network['dist_mat']) ** 2

    logging.info(
        f"\t{'{:,}'.format(n_connections)} "
        + "close ruptures out of "
        + f"{'{:,}'.format(n_possible_connections)} connections"
        + f" ({round(n_connections/n_possible_connections*100, 1)}%)"
    )

    if settings['filter_by_angle']:
        logging.info("  Filtering by rupture angle")
        binary_adjacence_matrix = filter_bin_adj_matrix_by_rupture_angle(
            fault_network['single_rup_df'],
            fault_network['subfaults'],
            binary_adjacence_matrix,
            threshold_angle=settings['rupture_angle_threshold'],
        )
        t3_ = time.time()
        event_times.append(t3_)
        logging.info(f"\tdone in {round(t3_-t3, 1)} s")
        n_connections = binary_adjacence_matrix.sum()
        logging.info(f"\t{'{:,}'.format(n_connections)} connections remaining")
        # filter continuous distance matrix
        fault_network['dist_mat'] *= binary_adjacence_matrix

    logging.info("Building subfault dataframe")
    fault_network['subfault_df'] = make_subfault_df(fault_network['subfaults'])
    t4 = time.time()
    event_times.append(t4)
    logging.info(f"\tdone in {round(t4-t3, 1)} s")

    logging.info("Getting multifault ruptures")
    # fault_network['multifault_inds'] = get_multifault_ruptures(
    fault_network['multifault_inds'] = get_multifault_ruptures_fast(
        # fault_network['multifault_inds'] = get_multifault_ruptures_numba(
        # fault_network['dist_mat'],
        binary_adjacence_matrix,
        # max_dist=settings['max_jump_distance'],
        max_sf_rups_per_mf_rup=settings['max_sf_rups_per_mf_rup'],
        parallel=settings['parallel_multifault_search'],
    )
    t5 = time.time()
    event_times.append(t5)
    logging.info(f"\tdone in {round(t5-t4, 1)} s")
    logging.info(
        f"\t{'{:,}'.format(len(fault_network['multifault_inds']))} "
        + "multifault ruptures"
    )

    logging.info("Making rupture dataframe")
    fault_network['rupture_df'] = make_rupture_df(
        fault_network['single_rup_df'],
        fault_network['multifault_inds'],
        fault_network['subfault_df'],
    )

    if settings['min_mag'] is not None:
        logging.info("Filtering ruptures by minimum magnitude")
        fault_network['rupture_df'] = fault_network['rupture_df'][
            fault_network['rupture_df']['mag'] >= settings['min_mag']
        ]

    if settings['max_mag'] is not None:
        logging.info("Filtering ruptures by maximum magnitude")
        fault_network['rupture_df'] = fault_network['rupture_df'][
            fault_network['rupture_df']['mag'] <= settings['max_mag']
        ]

    t6 = time.time()
    event_times.append(t6)
    logging.info(f"\tdone in {round(t6-t5, 1)} s")

    if settings['filter_by_plausibility']:
        t7 = time.time()
        event_times.append(t7)
        logging.info("Filtering ruptures by plausibility")
        fault_network['plausibility'] = get_rupture_plausibilities(
            fault_network['rupture_df'],
            distance_matrix=fault_network['dist_mat'],
            connection_distance_threshold=settings['max_jump_distance'],
            connection_distance_plausibility_threshold=settings[
                'rupture_filtering_connection_distance_plausibility_threshold'
            ],
        )

        fault_network['rupture_df_keep'] = (
            filter_proportionally_to_plausibility(
                fault_network['rupture_df'],
                fault_network['plausibility']['total'],
                seed=settings['filter_seed'],
            )
        )
        t8 = time.time()
        event_times.append(t8)
        n_rups_start = len(fault_network['rupture_df'])
        n_rups_filtered = len(fault_network['rupture_df_keep'])

        logging.info(f"\tdone in {round(t8-t7, 1)} s")
        logging.info(
            f"\t{'{:,}'.format(n_rups_filtered)} "
            + "ruptures remaining ("
            + f"{round(n_rups_filtered / n_rups_start*100, 1)} %)"
        )

    if settings['calculate_rates_from_slip_rates']:
        t_slip_rate_start = time.time()
        logging.info("Calculating rates from slip rates")
        if settings['rupture_set_for_rates_from_slip_rates'] == 'filtered':
            rup_df_key = 'rupture_df_keep'
        elif settings['rupture_set_for_rates_from_slip_rates'] == 'all':
            rup_df_key = 'rupture_df'

        rupture_rates = get_rup_rates_from_fault_slip_rates(
            fault_network,
            b_val=settings['fault_mfd_b_value'],
            mfd_type=settings['fault_mfd_type'],
            seismic_fraction=settings['seismic_fraction'],
            rupture_set_for_rates_from_slip_rates=settings[
                'rupture_set_for_rates_from_slip_rates'
            ],
            plot_fault_moment_rates=settings['plot_fault_moment_rates'],
        )
        fault_network[rup_df_key]['annual_occurrence_rate'] = rupture_rates
        t_slip_rate_end = time.time()
        event_times.append(t_slip_rate_end)
        logging.info(
            f"\tdone in {round(t_slip_rate_end-t_slip_rate_start, 1)} s"
        )

    fault_network['bin_dist_mat'] = binary_adjacence_matrix
    logging.info(f"total time: {round(event_times[-1]-event_times[0], 1)} s")
    return fault_network


def build_system_of_equations(
    rup_df,
    subsection_df,
    mag_col='mag',
    subfaults_col='subfaults',
    displacement_col='displacement',
    slip_rate_col='net_slip_rate',
    slip_rate_err_col='net_slip_rate_err',
    **soe_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a system of linear equations to solve in order to estimate
    the annual occurrence rate for each rupture, from the fault slip
    rates and magnitude-frequency distribution information.

    Parameters
    ----------
    rup_df : pd.DataFrame
        DataFrame containing information about each rupture.
        See `make_rupture_df` for more information on the format.
    subsection_df : pd.DataFrame
        DataFrame containing information about each subfault.
        See `make_subfault_df` for more information on the format.
    mag_col : str
        Name of the column in `rup_df` containing the rupture magnitudes.
    subfaults_col : str
        Name of the column in `rup_df` containing the subfault indices
        for each rupture.
    displacement_col : str
        Name of the column in `rup_df` containing the rupture displacements.
    slip_rate_col : str
        Name of the column in `subsection_df` containing the slip rates.
    slip_rate_err_col : str
        Name of the column in `subsection_df` containing the slip rate errors.
    soe_kwargs : dict
        Additional keyword arguments to pass to `openquake.fnm.soe.make_eqns`,
        with (for example) magnitude-frequency distribution information.

    Returns
    -------
    lhs : np.ndarray
        Left-hand side of the system of equations, i.e. the equations,
        of shape (m,n) where m is the number of constraints and n is
        the number of ruptures. The rows correspond to the ruptures and the
        columns correspond to the constraints.
    rhs : np.ndarray
        Right-hand side of the system of equations, i.e. the data. The shape
        is (m,1) where m is the number of constraints.
    errs : np.ndarray
        Errors for each equation. These are the standard devations of the
        data or analogous uncertainties that are used to weight the
        inversion. The shape is (m,1) where m is the number of constraints.
    """
    ruptures = rup_df_to_rupture_dicts(
        rup_df,
        mag_col=mag_col,
        displacement_col=displacement_col,
        subfaults_col=subfaults_col,
    )
    faults = subsection_df_to_fault_dicts(
        subsection_df,
        slip_rate_col=slip_rate_col,
        slip_rate_err_col=slip_rate_err_col,
    )

    lhs, rhs, errs = make_eqns(ruptures, faults, **soe_kwargs)

    return lhs, rhs, errs
