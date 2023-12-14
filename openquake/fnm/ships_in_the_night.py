import logging
from typing import Optional
from multiprocessing import Pool, Array

import numpy as np
import igraph as ig
import pandas as pd
from numba import jit

from openquake.hazardlib.geo.geodetic import spherical_to_cartesian

from openquake.aft.rupture_distances import (
    get_sequence_pairwise_dists,
    RupDistType,
)

from openquake.fnm.connections import get_angles
from openquake.fnm.once_more_with_feeling import (
    get_subsections_from_fault,
    simple_fault_from_feature,
    make_sf_rupture_meshes,
)


def get_bb_from_surface(surface):
    sphere_bb = surface.get_bounding_box()

    return np.array(
        [
            [sphere_bb.north, sphere_bb.east],
            [sphere_bb.north, sphere_bb.west],
            [sphere_bb.south, sphere_bb.west],
            [sphere_bb.south, sphere_bb.east],
        ]
    )


def get_bounding_box_distances(
    fault_surfaces, max_dist: Optional[float] = None
):
    poly_pts = [get_bb_from_surface(surface) for surface in fault_surfaces]
    cart_pts = [
        spherical_to_cartesian(pts[:, 0], pts[:, 1]) for pts in poly_pts
    ]

    fault_bb_dists = get_sequence_pairwise_dists(cart_pts, cart_pts)

    logging.info("  Filtering fault adjacence by distance")
    if max_dist is not None:
        fault_bb_dists = fault_bb_dists[fault_bb_dists["d"] <= max_dist]

    return fault_bb_dists


def get_close_faults(faults, max_dist: Optional[float] = None):
    surfaces = [fault['surface'] for fault in faults]
    fault_bb_dists = get_bounding_box_distances(surfaces, max_dist=max_dist)

    return fault_bb_dists


def get_rupture_patches_from_single_fault(
    subfaults,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
):
    num_rows = len(np.unique([sf['fault_position'][0] for sf in subfaults]))
    num_cols = len(np.unique([sf['fault_position'][1] for sf in subfaults]))

    subfault_quick_lookup = {
        sf['fault_position']: i for i, sf in enumerate(subfaults)
    }

    identifier = subfaults[0]['fid']
    sub_length = subfaults[0]['length']
    sub_width = subfaults[0]['width']

    single_fault_rup_indices = get_all_contiguous_subfaults(
        num_cols,
        num_rows,
        s_length=sub_length,
        d_length=sub_width,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    single_fault_rups = [
        [subfault_quick_lookup[pos] for pos in rup]
        for rup in single_fault_rup_indices
    ]

    return {identifier: single_fault_rups}


# @jit(nopython=True)
def get_all_contiguous_subfaults(
    NS: int,
    ND: int,
    s_length: float = 10.0,
    d_length: float = 10.0,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
):
    subarrays = []
    if NS == 1:  # single column
        return [[(d, 0)] for d in range(ND)]
    for row_start in range(ND):
        for col_start in range(NS):
            for row_end in range(row_start, ND):
                for col_end in range(col_start, NS):
                    n_rows = row_end - row_start + 1
                    n_cols = col_end - col_start + 1

                    along_strike_length = n_cols * s_length
                    down_dip_width = n_rows * d_length

                    aspect_ratio = along_strike_length / down_dip_width

                    if (
                        (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio)
                        or (min_aspect_ratio <= aspect_ratio and n_rows == ND)
                        or (n_rows == 1 and n_cols == 1)
                    ):
                        subarray = [
                            (r, c)
                            for r in range(row_start, row_end + 1)
                            for c in range(col_start, col_end + 1)
                        ]
                        subarrays.append(subarray)
    return subarrays


def subfaults_are_adjacent(subfault_1, subfault_2):
    pos1 = subfault_1['fault_position']
    pos2 = subfault_2['fault_position']

    if (pos1[0] == pos2[0] and np.abs(pos1[1] - pos2[1]) == 1) or (
        pos1[1] == pos2[1] and np.abs(pos1[0] - pos2[0]) == 1
    ):
        return True


def get_single_fault_rupture_coordinates(rupture, single_fault_subfaults):
    if len(rupture) == 1:
        return single_fault_subfaults[rupture[0]]['surface'].mesh.xyz
    else:
        return np.vstack(
            [single_fault_subfaults[i]['surface'].mesh.xyz for i in rupture]
        )


def get_single_fault_rups(subfaults, subfault_index_start: int = 0):
    fault_rups = get_rupture_patches_from_single_fault(subfaults)
    rup_patches = list(fault_rups.values())[0]
    rup_subfaults = [
        [rp + subfault_index_start for rp in rup] for rup in rup_patches
    ]
    num_rups = len(rup_patches)

    rupture_df = pd.DataFrame(
        index=np.arange(num_rups) + subfault_index_start,
        data={
            'fault_rup': np.arange(num_rups),
            'patches': rup_patches,
            'subfaults': rup_subfaults,
        },
    )
    rupture_df['fault'] = list(fault_rups.keys())[0]

    return rupture_df


def get_all_single_fault_rups(all_subfaults):
    single_fault_rups = []
    rup_dfs = []
    all_rup_count = 0
    subfault_count = 0

    for subfaults in all_subfaults:
        rupture_df = get_single_fault_rups(
            subfaults, subfault_index_start=subfault_count
        )

        num_rups = rupture_df.shape[0]
        single_fault_rups.append(rupture_df['patches'].values.tolist())
        rup_dfs.append(rupture_df)
        all_rup_count += num_rups
        subfault_count += len(subfaults)

    rup_df = pd.concat(rup_dfs, axis=0).reset_index(drop=True)

    return single_fault_rups, rup_df


def get_rupture_adjacency_matrix(
    faults,
    all_subfaults=None,
    multifaults_on_same_fault: bool = False,
    max_dist: float = 20.0,
):
    if all_subfaults is None:
        all_subfaults = [
            get_subsections_from_fault(fault, surface=fault['surface'])
            for fault in faults
        ]

    single_fault_rups, single_fault_rup_df = get_all_single_fault_rups(
        all_subfaults
    )

    single_fault_rup_coords = [
        [
            get_single_fault_rupture_coordinates(rup, subfaults)
            for rup in single_fault_rups[i]
        ]
        for i, subfaults in enumerate(all_subfaults)
    ]

    nrups = single_fault_rup_df.shape[0]

    fault_dists = get_close_faults(faults)

    fault_dists = {(i, j): d for i, j, d in fault_dists}

    dist_adj_matrix = np.zeros((nrups, nrups), dtype=np.float32)

    row_count = 0
    for i, f1 in enumerate(faults):
        nrows = len(single_fault_rups[i])
        col_count = 0
        for j, f2 in enumerate(faults):
            ncols = len(single_fault_rups[j])
            pw_source_dists = None
            if i == j:
                if multifaults_on_same_fault:
                    pw_source_dists = get_sequence_pairwise_dists(
                        single_fault_rup_coords[i],
                        single_fault_rup_coords[j],
                    )
                else:
                    pass
            elif i < j:
                if fault_dists[(i, j)] <= max_dist:
                    pw_source_dists = get_sequence_pairwise_dists(
                        single_fault_rup_coords[i],
                        single_fault_rup_coords[j],
                    )
            else:
                pass

            if pw_source_dists is not None:
                dist_adj_matrix[
                    row_count : row_count + nrows,
                    col_count : col_count + ncols,
                ] = rdist_to_dist_matrix(pw_source_dists, nrows, ncols)

            col_count += ncols
        row_count += nrows

    return single_fault_rup_df, dist_adj_matrix


def make_binary_distance_matrix(dist_matrix, max_dist: float = 10.0):
    binary_dist_matrix = np.zeros(dist_matrix.shape, dtype=np.int32)
    binary_dist_matrix[(dist_matrix > 0.0) & (dist_matrix <= max_dist)] = 1

    return binary_dist_matrix


def get_proximal_rup_angles(sf_meshes, binary_distance_matrix):
    rup_angles = {}
    for i, mesh_0 in enumerate(sf_meshes):
        for j, mesh_1 in enumerate(sf_meshes):
            if binary_distance_matrix[i, j] == 1:
                rup_angles[(i, j)] = get_angles(mesh_0, mesh_1)
    return rup_angles


def filter_bin_adj_matrix_by_rupture_angle(
    single_rup_df,
    subfaults,
    binary_adjacence_matrix,
    threshold_angle=60.0,
    angle_type='trace',
):
    if angle_type == 'trace':
        angle_index = 1
    elif angle_type == 'dihedral':
        angle_index = 0

    sf_meshes = make_sf_rupture_meshes(
        single_rup_df['patches'], single_rup_df['fault'], subfaults
    )

    rup_angles = get_proximal_rup_angles(sf_meshes, binary_adjacence_matrix)

    for (i, j), angles in rup_angles.items():
        if angles[angle_index] < threshold_angle:
            binary_adjacence_matrix[i, j] = 0

    return binary_adjacence_matrix


def get_multifault_ruptures(
    dist_adj_matrix,
    max_dist: float = 10.0,
    check_unique: bool = False,
    max_sf_rups_per_mf_rup: int = 10,
):
    n_rups = dist_adj_matrix.shape[0]

    if max_sf_rups_per_mf_rup == -1:
        max_sf_rups_per_mf_rup = n_rups
    elif max_sf_rups_per_mf_rup > n_rups:
        max_sf_rups_per_mf_rup = n_rups

    dist_adj_binary = make_binary_distance_matrix(dist_adj_matrix, max_dist)

    graph = ig.Graph.Adjacency(dist_adj_binary)

    paths = []
    for i in range(n_rups):
        ps = graph.get_all_simple_paths(
            i, to=None, cutoff=max_sf_rups_per_mf_rup, mode='out'
        )
        paths.extend(ps)

    paths = [(p) for p in paths if len(p) > 1]

    if check_unique:
        paths = [frozenset(p) for p in paths if len(p) > 1]
        paths = set(paths)
        paths = [list(p) for p in paths]
    return paths


@jit(nopython=True)
def rdist_to_dist_matrix(rdist: RupDistType, nrows: int = -1, ncols: int = -1):
    if nrows == -1:
        nrows = np.max(rdist['r1'])

    if ncols == -1:
        ncols = np.max(rdist['r2'])

    dist_matrix = np.zeros((nrows, ncols), dtype=np.float32)

    for row in rdist:
        i = int(row[0])
        j = int(row[1])
        dist_matrix[i, j] = row[2]

    return dist_matrix


def get_mf_distances_from_adj_matrix(mf, dist_adj_matrix):
    distances = np.empty(len(mf) - 1)
    for i in range(len(distances)):
        distances[i] = dist_adj_matrix[mf[i], mf[i + 1]]
    return distances
