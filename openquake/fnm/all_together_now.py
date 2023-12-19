import time
import json
import logging
from copy import deepcopy

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from openquake.fnm.once_more_with_feeling import (
    get_subsections_from_fault,
    simple_fault_from_feature,
    make_subfault_df,
    make_rupture_df,
)


from openquake.fnm.ships_in_the_night import (
    get_rupture_adjacency_matrix,
    get_multifault_ruptures,
    make_binary_adjacency_matrix,
    filter_bin_adj_matrix_by_rupture_angle,
)


default_settings = {
    'subsection_size': [15.0, 15.0],
    'edge_sd': 2.0,
    'dip_sd': 2.0,
    'max_jump_distance': 10.0,
    'max_sf_rups_per_mf_rup': 10,
    'rupture_angle_threshold': 60.0,
}


def build_fault_network(
    faults=None,
    fault_geojson=None,
    settings=None,
    surface_type='simple',
    filter_by_angle=True,
    **kwargs,
):
    build_settings = deepcopy(default_settings)
    if settings is not None:
        build_settings.update(settings)
    build_settings.update(kwargs)
    if settings is None:
        settings = build_settings

    fault_network = {}

    t0 = time.time()
    if faults is None:
        if surface_type == 'simple':
            build_surface = simple_fault_from_feature
        else:
            raise NotImplementedError(
                f'Surface type {surface_type} not implemented'
            )

        if fault_geojson is not None:
            logging.info("Building faults from geojson")
            with open(fault_geojson) as f:
                fault_gj = json.load(f)
            faults = [
                build_surface(feature) for feature in fault_gj['features']
            ]

            fault_fids = [fault['fid'] for fault in faults]
            duplicated_fids = [
                fid for fid in set(fault_fids) if fault_fids.count(fid) > 1
            ]
            if len(duplicated_fids) > 0:
                raise ValueError(f'Duplicated fault fids: {duplicated_fids}')

        else:
            raise ValueError('No faults provided')
    t1 = time.time()
    logging.info(f"\tdone in {round(t1-t0, 1)} s")

    logging.info("Making subfaults")
    fault_network['subfaults'] = [
        get_subsections_from_fault(
            fault,
            subsection_size=build_settings['subsection_size'],
            edge_sd=build_settings['edge_sd'],
            dip_sd=build_settings['dip_sd'],
            surface=fault['surface'],
        )
        for fault in faults
    ]

    n_subfaults = sum([len(sf) for sf in fault_network['subfaults']])
    t2 = time.time()
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
    )
    t3 = time.time()
    logging.info(f"\tdone in {round(t3-t2, 1)} s")
    logging.info(
        f"\t{'{:,}'.format(len(fault_network['single_rup_df']))} single-fault ruptures"
    )

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

    if filter_by_angle:
        logging.info("  Filtering by rupture angle")
        binary_adjacence_matrix = filter_bin_adj_matrix_by_rupture_angle(
            fault_network['single_rup_df'],
            fault_network['subfaults'],
            binary_adjacence_matrix,
            threshold_angle=settings['rupture_angle_threshold'],
        )
        t3 = time.time()
        # logging.info(f"\tdone in {round(t3-t2, 1)} s")
        n_connections = binary_adjacence_matrix.sum()
        logging.info(f"\t{'{:,}'.format(n_connections)} connections remaining")
        # filter continuous distance matrix
        fault_network['dist_mat'] *= binary_adjacence_matrix

    logging.info("Building subfault dataframe")
    fault_network['subfault_df'] = make_subfault_df(fault_network['subfaults'])
    t4 = time.time()
    logging.info(f"\tdone in {round(t4-t3, 1)} s")

    logging.info("Getting multifault ruptures")
    fault_network['multifault_inds'] = get_multifault_ruptures(
        fault_network['dist_mat'],
        max_dist=settings['max_jump_distance'],
        max_sf_rups_per_mf_rup=settings['max_sf_rups_per_mf_rup'],
    )
    t5 = time.time()
    logging.info(f"\tdone in {round(t5-t4, 1)} s")
    logging.info(
        f"\t{'{:,}'.format(len(fault_network['multifault_inds']))} multifault ruptures"
    )

    logging.info(f"total time: {round(t5-t0, 1)} s")
    return fault_network
