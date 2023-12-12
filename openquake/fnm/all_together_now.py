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
    get_simple_fault_from_feature,
)


from openquake.fnm.ships_in_the_night import (
    get_rupture_adjacency_matrix,
    get_multifault_ruptures,
)


default_settings = {
    'subsection_size': [15.0, 15.0],
    'edge_sd': 2.0,
    'dip_sd': 2.0,
    'max_jump_distance': 10.0,
    'max_sf_rups_per_mf_rup': 10,
}


def build_fault_network(
    faults=None,
    fault_geojson=None,
    settings=None,
    surface_type='simple',
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
            build_surface = get_simple_fault_from_feature
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
    t2 = time.time()
    logging.info(f"\tdone in {round(t2-t1, 1)} s")

    logging.info("making single fault rup df and distance matrix")
    fault_network['single_rup_df'], dist_mat = get_rupture_adjacency_matrix(
        faults,
        all_subfaults=fault_network['subfaults'],
        max_dist=settings['max_jump_distance'],
    )
    t3 = time.time()
    logging.info(f"\tdone in {round(t3-t2, 1)} s")

    logging.info("getting multifaults")
    fault_network['multifault_inds'] = get_multifault_ruptures(
        dist_mat,
        max_sf_rups_per_mf_rup=settings['max_sf_rups_per_mf_rup'],
    )
    t4 = time.time()
    logging.info(f"\tdone in {round(t4-t3, 1)} s")

    logging.info(f"total time: {round(t4-t0, 1)} s")
    return fault_network
