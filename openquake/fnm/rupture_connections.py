import logging
from typing import Optional, Dict, List, Mapping, Sequence, Set
from functools import partial
from collections import deque
from multiprocessing import Pool, Array

import numpy as np
import igraph as ig
import pandas as pd
from numba import jit, njit, int16, int32
from numba.typed import List
from scipy.sparse import dok_array, issparse, csr_matrix, isspmatrix_csr

from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.geodetic import (
    spherical_to_cartesian,
    azimuth,
    geodetic_distance,
)

from openquake.aft.rupture_distances import (
    get_sequence_pairwise_dists,
    calc_pairwise_distances,
    RupDistType,
)

from openquake.fnm.connections import get_angles
from openquake.fnm.fault_modeler import (
    get_subsections_from_fault,
    simple_fault_from_feature,
    make_sf_rupture_meshes,
    get_trace_from_sf_rupture,
    get_trace_from_mesh,
)


from openquake.fnm.mesh import get_mesh_bb
from openquake.fnm.bbox import get_bb_distance

# these don't work well
# from openquake.fnm.multifault_parallel import (
#        find_connected_subsets_parallel,
#        find_connected_subsets_parallel_py,
#        )


def get_bb_from_surface(surface):
    sphere_bb = surface.get_bounding_box()

    return np.array(
        [
            [sphere_bb[3], sphere_bb[0]],
            [sphere_bb[3], sphere_bb[1]],
            [sphere_bb[2], sphere_bb[1]],
            [sphere_bb[2], sphere_bb[0]],
        ]
    )


def get_bounding_box_distances(
    fault_surfaces, max_dist: Optional[float] = None
) -> RupDistType:
    """
    Finds the distances in km between the bounding boxes of a set of fault
    surfaces. If `max_dist` is set, then only distances less than or equal to
    `max_dist` are returned.

    Parameters
    ----------
    fault_surfaces : list of openquake.hazardlib.geo.surface.base.BaseSurface
        List of fault surfaces.
    max_dist : float, optional
        Maximum distance between bounding boxes. The default is None.

    Returns
    -------
    RupDistType
        A structured array containing the distances between bounding boxes.
        The fields are:
            r1: index of the first bounding box (int)
            r2: index of the second bounding box (int)
            d: distance between the bounding boxes (float)
    """
    poly_pts = [get_bb_from_surface(surface) for surface in fault_surfaces]
    cart_pts = [
        spherical_to_cartesian(pts[:, 0], pts[:, 1]) for pts in poly_pts
    ]

    fault_bb_dists = get_sequence_pairwise_dists(cart_pts, cart_pts)

    if max_dist is not None:
        logging.info("  Filtering fault adjacence by distance")
        fault_bb_dists = fault_bb_dists[fault_bb_dists["d"] <= max_dist]

    return fault_bb_dists


def get_bounding_box_distances_marco(
    fault_surfaces, max_dist: Optional[float] = None
):
    mesh_bbs = [get_mesh_bb(surface.mesh) for surface in fault_surfaces]
    dists = {}
    for i in range(len(mesh_bbs)):
        for j in range(i + 1, len(mesh_bbs)):
            dist = get_bb_distance(mesh_bbs[i], mesh_bbs[j])
            if max_dist is None or dist <= max_dist:
                dists[(i, j)] = dist
                dists[(j, i)] = dist
    return dists


def get_close_faults(
    faults: list[dict], max_dist: Optional[float] = None
) -> RupDistType:
    """
    Finds the distances in km between the bounding boxes of a set of faults.
    If `max_dist` is set, then only distances less than or equal to `max_dist`
    are returned.

    Parameters
    ----------
    faults : list of fault dictionaries
        List of fault dictionaries.
    max_dist : float, optional
        Maximum distance between bounding boxes. The default is None.

    Returns
    -------
    RupDistType
        A structured array containing the distances between bounding boxes.
        The fields are:
            r1: index of the first bounding box (int)
            r2: index of the second bounding box (int)
            d: distance between the bounding boxes (float)
    """
    surfaces = [fault['surface'] for fault in faults]
    # fault_bb_dists_ = get_bounding_box_distances(surfaces, max_dist=max_dist)

    # fault_bb_dists = {(int(f0), int(f1)): float(d)
    #                  for f0, f1, d in fault_bb_dists_
    #                  if f0 != f1}

    fault_bb_dists = get_bounding_box_distances_marco(
        surfaces, max_dist=max_dist
    )

    return fault_bb_dists


def get_rupture_patches_from_single_fault(
    subfaults,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
) -> dict:
    """
    Get all possible contiguous subfaults from a single fault, within
    the specified aspect ratio bounds.

    Parameters
    ----------
    subfaults : list of dictionaries
        List of subfault dictionaries.
    min_aspect_ratio : float, optional
        Minimum aspect ratio of the rupture. The default is 0.8.
    max_aspect_ratio : float, optional
        Maximum aspect ratio of the rupture. The default is 3.0.

    Returns
    -------
    dict
        Dictionary of ruptures. The keys are the fault identifiers, and the
        values are lists of lists of subfault indices.
    """
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


def get_all_contiguous_subfaults(
    NS: int,
    ND: int,
    s_length: float = 10.0,
    d_length: float = 10.0,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
) -> list[list[tuple[int, int]]]:
    """
    Get all possible contiguous subfaults from a single fault, within
    the specified aspect ratio bounds.

    Parameters
    ----------
    NS : int
        Number of subfaults along strike.
    ND : int
        Number of subfaults down dip.
    s_length : float, optional
        Length of subfaults along strike. The default is 10.0.
    d_length : float, optional
        Length of subfaults down dip. The default is 10.0.
    min_aspect_ratio : float, optional
        Minimum aspect ratio of the rupture. The default is 0.8.
    max_aspect_ratio : float, optional
        Maximum aspect ratio of the rupture. The default is 3.0.

    Returns
    -------
    list[list[tuple[int, int]]]
        List of ruptures. Each rupture is a list of tuples, where each tuple
        contains the row and column index of a subfault.
    """
    subarrays = []
    if NS == 1:  # single column
        subarrays = [[(d, 0)] for d in range(ND)]
        if ND > 1:
            subarrays = subarrays + [[(d, 0) for d in range(ND)]]
        return subarrays
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
                        or (n_rows == ND and n_cols == NS)
                    ):
                        subarray = [
                            (r, c)
                            for r in range(row_start, row_end + 1)
                            for c in range(col_start, col_end + 1)
                        ]
                        subarrays.append(subarray)
    return subarrays


def subfaults_are_adjacent(subfault_1, subfault_2) -> bool:
    """
    Check if two subfaults are adjacent.

    Parameters
    ----------
    subfault_1 : dict
        Dictionary of subfault properties.
    subfault_2 : dict
        Dictionary of subfault properties.

    Returns
    -------
    bool
        True if the subfaults are adjacent, False otherwise.
    """
    pos1 = subfault_1['fault_position']
    pos2 = subfault_2['fault_position']

    if (pos1[0] == pos2[0] and np.abs(pos1[1] - pos2[1]) == 1) or (
        pos1[1] == pos2[1] and np.abs(pos1[0] - pos2[0]) == 1
    ):
        return True
    else:
        return False


def get_single_fault_rupture_coordinates(
    rupture, single_fault_subfaults
) -> np.ndarray:
    """
    Get the Euclidean coordinates of a single-fault rupture.

    Parameters
    ----------
    rupture : list of int
        List of subfault indices.
    single_fault_subfaults : list of dictionaries
        List of subfault dictionaries.

    Returns
    -------
    np.ndarray
        Array of subfault coordinates with a shape of (n,3), where each
        row is a point with x,y,z coordinates (in km from the Earth's center).
    """

    if len(rupture) == 1:
        return single_fault_subfaults[rupture[0]]['surface'].mesh.xyz
    else:
        coords = np.vstack(
            [single_fault_subfaults[i]['surface'].mesh.xyz for i in rupture]
        )
        return np.unique(coords, axis=0)


def get_single_fault_rups(
    subfaults,
    subfault_index_start: int = 0,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
    fault_group: int | None = None,
) -> pd.DataFrame:
    """
    Get all possible ruptures from a single fault.

    Parameters
    ----------
    subfaults : list of dictionaries
        List of subfault dictionaries.
    subfault_index_start : int, optional
        Index of the first subfault. The default is 0.
    fault_group: index of fault_group, optional.
    min_aspect_ratio : float, optional
        Minimum aspect ratio of the rupture. The default is 0.8.
    max_aspect_ratio : float, optional
        Maximum aspect ratio of the rupture. The default is 3.0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the rupture information. The columns are:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
    """
    num_subfaults = len(subfaults)
    fault_rups = get_rupture_patches_from_single_fault(
        subfaults,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )
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
    rupture_df['full_fault_rupture'] = [
        len(rup) == num_subfaults for rup in rup_patches
    ]
    if fault_group is not None:
        rupture_df['fault_group'] = fault_group

    return rupture_df


def get_all_single_fault_rups(
    all_subfaults,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
) -> tuple[list[list[int]], pd.DataFrame]:
    """
    Get all possible single-fault ruptures from a set of subfaults.

    Parameters
    ----------
    all_subfaults : list of lists of dictionaries
        List of lists of subfault dictionaries. Each list of subfaults
        is derived from a single, contiguous fault.
    min_aspect_ratio : float, optional
        Minimum aspect ratio of the rupture. The default is 0.8.
    max_aspect_ratio : float, optional
        Maximum aspect ratio of the rupture. The default is 3.0.

    Returns
    -------
    tuple[list[list[int]], pd.DataFrame]
        Tuple containing a list of single-fault ruptures and a DataFrame
        containing the rupture information. The columns are:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
    """
    single_fault_rups = []
    rup_dfs = []
    all_rup_count = 0
    subfault_count = 0
    fault_count = 0

    for subfaults in all_subfaults:
        rupture_df = get_single_fault_rups(
            subfaults,
            subfault_index_start=subfault_count,
            # fault_group=fault_count,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )

        num_rups = rupture_df.shape[0]
        single_fault_rups.append(rupture_df['patches'].values.tolist())
        rup_dfs.append(rupture_df)
        all_rup_count += num_rups
        subfault_count += len(subfaults)
        fault_count += 1

    rup_df = pd.concat(rup_dfs, axis=0).reset_index(drop=True)

    return single_fault_rups, rup_df


def get_rupture_adjacency_matrix(
    faults,
    all_subfaults=None,
    multifaults_on_same_fault: bool = False,
    max_dist: Optional[float] = 20.0,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
    sparse: bool = True,
    full_fault_only_mf_ruptures: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Get the rupture adjacency matrix for a set of faults. Adjacency values
    are the distances between the ruptures in km. The distance value for
    non-adjacent ruptures is 0.0, which is probably a bad idea and will
    likely be changed in the future.

    Parameters
    ----------
    faults : list of dictionaries
        List of fault dictionaries.
    all_subfaults : list of lists of dictionaries, optional
        List of lists of subfault dictionaries. Each list of subfaults
        is derived from a single, contiguous fault. The default is None.
    multifaults_on_same_fault : bool, optional
        Whether to compute the rupture distances between ruptures on the same
        fault. The default is False.
    max_dist : float, optional
        Maximum distance between ruptures in km. The default is 20.0.
        If this value is not None, then the rupture adjacency matrix will
        be filtered by distance, so that ruptures farther apart than this
        distance will be given the null (0.0) distance value.
    min_aspect_ratio : float, optional
        Minimum aspect ratio of the rupture. The default is 0.8.
    max_aspect_ratio : float, optional
        Maximum aspect ratio of the rupture. The default is 3.0.
    sparse: bool.
        Whether to return a sparse adjacency matrix.
    full_fault_only_mf_ruptures: bool
        Only use full-fault ruptures to assemble multifault ruptures. Must
        be `True` for regional-scale or larger PSHA models, otherwise
        billions or trillions of multifault ruptures will be returned.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Tuple containing a DataFrame of rupture information and the rupture
        adjacency matrix. The rupture DataFrame contains the following columns:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
        The rupture adjacency matrix is a square matrix with dimensions
        equal to the number of ruptures. The values are the distances between
        the ruptures in km.
    """

    def sparsify_maybe(mat, sparse: bool):
        if sparse:
            return dok_array(mat, dtype=mat.dtype)
        return mat

    if all_subfaults is None:
        logging.info("  getting subsections from faults")
        all_subfaults = [
            get_subsections_from_fault(fault, surface=fault['surface'])
            for fault in faults
        ]

    logging.info("  making single-fault ruptures")
    single_fault_rups, single_fault_rup_df = get_all_single_fault_rups(
        all_subfaults,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    nrups = single_fault_rup_df.shape[0]

    # increasing distance to make up for discrepancy between bb dist and
    # actual rupture distances; this is a filtering step.
    logging.info("  calculating fault distances")
    fault_dists = get_close_faults(faults, max_dist=max_dist * 1.5)

    # fault_dists = {(i, j): d for i, j, d in fault_dists}

    logging.info(f"  making dist_adj_matrix {(nrups, nrups)}")
    if sparse:
        dist_adj_matrix = dok_array((nrups, nrups), dtype=np.float32)
    else:
        dist_adj_matrix = np.zeros((nrups, nrups), dtype=np.float32)

    if max_dist is None:
        max_dist = np.inf

    logging.info("  filtering and calculating pairwise rupture distances")
    if full_fault_only_mf_ruptures:
        fault_lookup = {fault['fid']: i for i, fault in enumerate(faults)}

        full_fault_ruptures = sorted(
            set(
                [
                    i
                    for i, rup in single_fault_rup_df.iterrows()
                    if rup['full_fault_rupture']
                ]
            )
        )
        full_fault_rup_coords = {}
        for ff in full_fault_ruptures:
            fault_idx = fault_lookup[single_fault_rup_df.loc[ff, 'fault']]
            full_fault_rup_coords[ff] = get_single_fault_rupture_coordinates(
                single_fault_rup_df.loc[ff, 'patches'],
                all_subfaults[fault_idx],
            )

        for row_ff in full_fault_ruptures:
            row_fault_ind = fault_lookup[
                single_fault_rup_df.loc[row_ff, 'fault']
            ]
            for col_ff in full_fault_ruptures:
                col_fault_ind = fault_lookup[
                    single_fault_rup_df.loc[col_ff, 'fault']
                ]

                if (row_ff < col_ff) and (
                    row_fault_ind,
                    col_fault_ind,
                ) in fault_dists:
                    dist = np.min(
                        calc_pairwise_distances(
                            full_fault_rup_coords[row_ff],
                            full_fault_rup_coords[col_ff],
                        )
                    )
                    if dist <= max_dist:
                        dist_adj_matrix[row_ff, col_ff] = dist

    else:
        single_fault_rup_coords = [
            [
                get_single_fault_rupture_coordinates(rup, subfaults)
                for rup in single_fault_rups[i]
            ]
            for i, subfaults in enumerate(all_subfaults)
        ]

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
                elif i < j:  # upper triangular matrix
                    if (i, j) not in fault_dists:
                        pass
                    elif fault_dists[(i, j)] <= max_dist * 1.5:
                        pw_source_dists = get_sequence_pairwise_dists(
                            single_fault_rup_coords[i],
                            single_fault_rup_coords[j],
                        )
                # else:
                #    pass

                if pw_source_dists is not None:
                    local_dist_matrix = rdist_to_dist_matrix(
                        pw_source_dists, nrows, ncols
                    )
                    local_dist_matrix[local_dist_matrix > max_dist] = 0.0

                    dist_adj_matrix[
                        row_count : row_count + nrows,
                        col_count : col_count + ncols,
                    ] = sparsify_maybe(local_dist_matrix, sparse)

                col_count += ncols
            row_count += nrows

    logging.debug(" dist matrix type: %s", type(dist_adj_matrix))

    # make the matrix symmetric
    dist_adj_matrix += dist_adj_matrix.T

    return single_fault_rup_df, dist_adj_matrix


def get_rupture_grouping(faults, single_rup_df):
    fault_lookup = {fault['fid']: i for i, fault in enumerate(faults)}
    rup_flt_lookup = {i: rup['fault'] for i, rup in single_rup_df.iterrows()}
    rup_flt_index_lookup = {
        k: fault_lookup[v] for k, v in rup_flt_lookup.items()
    }

    return rup_flt_index_lookup


def make_binary_adjacency_matrix(
    dist_matrix, max_dist: float = 10.0
) -> np.ndarray:
    """
    Make a binary adjacency matrix from a distance matrix. The binary
    adjacency matrix is 1 if the distance is less than or equal to
    `max_dist`, and 0 otherwise.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix.
    max_dist : float, optional
        Maximum distance between ruptures in km. The default is 10.0.

    Returns
    -------
    np.ndarray
        Binary adjacency matrix.
    """
    binary_dist_matrix = np.zeros(dist_matrix.shape, dtype=np.int32)
    binary_dist_matrix[(dist_matrix > 0.0) & (dist_matrix <= max_dist)] = 1

    return binary_dist_matrix


def make_binary_adjacency_matrix_sparse(
    dist_matrix, max_dist: float = 10.0
) -> dok_array:
    """
    Make a binary adjacency matrix from a distance matrix. The binary
    adjacency matrix is 1 if the distance is less than or equal to
    `max_dist`, and 0 otherwise.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix.
    max_dist : float, optional
        Maximum distance between ruptures in km. The default is 10.0.

    Returns
    -------
    dok_array
        Binary adjacency matrix.
    """
    dist_matrix = dist_matrix.todok()
    binary_dist_matrix = dok_array(dist_matrix.shape, dtype=np.int32)
    for (row, col), distance in dist_matrix.items():
        if 0.0 < distance <= max_dist:
            binary_dist_matrix[row, col] = 1
    # binary_dist_matrix[(dist_matrix > 0.0) & (dist_matrix <= max_dist)] = 1

    return binary_dist_matrix


def _lonlat_to_xyz(lon, lat):
    phi = np.radians(lat)
    lamb = np.radians(lon)

    return np.array(
        [np.cos(phi) * np.cos(lamb), np.cos(phi) * np.sin(lamb), np.sin(phi)]
    )


def _xyz_to_lonlat(sp):
    x, y, z = sp
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    lamb = np.arctan2(y, x)

    lon, lat = np.degrees([lamb, phi])
    return lon, lat


def _geog_vec_to_xyz(lon, lat, bearing):
    phi, lamb, theta = np.radians([lat, lon, bearing])

    C = np.array(
        [
            np.sin(lamb) * np.cos(theta)
            - np.sin(phi) * np.cos(lamb) * np.sin(theta),
            -np.cos(lamb) * np.cos(theta)
            - np.sin(phi) * np.sin(lamb) * np.sin(theta),
            np.cos(phi) * np.sin(theta),
        ]
    )
    return C


def intersection_pt(
    lon_a, lat_a, strike_a, lon_b, lat_b, strike_b, return_closest=True
):
    p_a = _lonlat_to_xyz(lon_a, lat_a)
    p_b = _lonlat_to_xyz(lon_b, lat_b)

    c_a = _geog_vec_to_xyz(lon_a, lat_a, strike_a)
    c_b = _geog_vec_to_xyz(lon_b, lat_b, strike_b)

    n1 = np.cross(c_a, c_b)
    n2 = np.cross(c_b, c_a)

    pt_1 = _xyz_to_lonlat(n1)
    pt_2 = _xyz_to_lonlat(n2)

    if return_closest:
        d1 = np.arccos(p_a.dot(n1))
        d2 = np.arccos(p_b.dot(n2))
        if d1 <= d2:
            return pt_1
        else:
            return pt_2
    else:
        return pt_1, pt_2


def find_intersection_angle(trace_1, trace_2, az_1=None, az_2=None):
    if not isinstance(trace_1, Line):
        tr1 = Line([Point(*coords) for coords in trace_1])
        tr2 = Line([Point(*coords) for coords in trace_2])
    else:
        tr1 = trace_1
        tr2 = trace_2

    if az_1 is None:
        az_1 = tr1.average_azimuth()
    if az_2 is None:
        az_2 = tr2.average_azimuth()

    pt_1_x, pt_1_y = tr1[0].x, tr1[0].y
    pt_2_x, pt_2_y = tr2[0].x, tr2[0].y

    int_pt = intersection_pt(pt_1_x, pt_1_y, az_1, pt_2_x, pt_2_y, az_2)

    # use the farther point from int_pt on each trace to calculate the angle
    dists_1 = geodetic_distance(
        int_pt[0],
        int_pt[1],
        np.array([c[0] for c in tr1.coo]),
        np.array([c[1] for c in tr1.coo]),
    )

    dists_2 = geodetic_distance(
        int_pt[0],
        int_pt[1],
        np.array([c[0] for c in tr2.coo]),
        np.array([c[1] for c in tr2.coo]),
    )

    pt_1_x, pt_1_y = (
        trace_1[np.argmax(dists_1)].x,
        trace_1[np.argmax(dists_1)].y,
    )
    pt_2_x, pt_2_y = (
        trace_2[np.argmax(dists_2)].x,
        trace_2[np.argmax(dists_2)].y,
    )

    az_pt_1, az_pt_2 = azimuth(
        [pt_1_x, pt_2_x], [pt_1_y, pt_2_y], int_pt[0], int_pt[1]
    )

    angle_between = np.abs(az_pt_1 - az_pt_2) % 360

    if angle_between > 180.0:
        angle_between = 360 - angle_between

    return int_pt, angle_between


def get_proximal_rup_angles(
    sf_traces, binary_distance_matrix, verbose=False
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Get the rupture angles between proximal ruptures (i.e., those considered
    close by the adjacency matrix). The rupture angles are given by a
    dictionary item where the key is a tuple of the indices of each rupture
    and the value is a tuple of the rupture angles in degrees. The first
    angle is the dihedral angle between the two ruptures, and the second
    angle is the trace angle between the two ruptures.

    Parameters
    ----------
    sf_meshes : list of lists of Surfaces
        List of lists of fault surfaces.
    binary_distance_matrix : scipy.sparse.dok_matrix
        Binary adjacency matrix in DOK format.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    dict[tuple[int, int], tuple[float, float]]
        Dictionary of rupture angles. The keys are tuples of rupture indices,
        and the values are tuples of rupture angles in degrees.
    """
    n_faults = len(sf_traces)
    pad_width = len(str(n_faults))

    # Get the non-zero entries directly from the DOK matrix
    nonzero_pairs = list(binary_distance_matrix.keys())
    total_pairs = len(nonzero_pairs)

    convert_from_coords = not isinstance(sf_traces[0], Line)
    if convert_from_coords and verbose:
        print(" getting traces from faults")

    proximal_traces: dict[int, Line] = {}
    for i, j in nonzero_pairs:
        for k in (i, j):
            if k not in proximal_traces:
                trace = sf_traces[k]
                if convert_from_coords:
                    proximal_traces[k] = Line(
                        [Point(*coords) for coords in trace]
                    )
                else:
                    proximal_traces[k] = trace

    rup_angles = {}

    for idx, (i, j) in enumerate(nonzero_pairs, 1):
        if verbose:
            if idx < total_pairs:
                print(
                    f"  doing pair {str(idx).zfill(pad_width)} out of {total_pairs}",
                    end="\r",
                )
            else:
                print(f"  doing pair {idx} out of {total_pairs}")

        trace_0 = proximal_traces[i]
        trace_1 = proximal_traces[j]
        rup_angles[(i, j)] = find_intersection_angle(trace_0, trace_1)

    return rup_angles


def angle_difference(angle_1, angle_2):
    return np.abs(angle_1 - angle_2) % 360


def get_dists_and_azimuths_from_pt(trace, pt):
    endpts_x = np.array([trace[0].x, trace[-1].x])
    endpts_y = np.array([trace[0].y, trace[-1].y])

    dists = geodetic_distance(pt[0], pt[1], endpts_x, endpts_y)
    azimuths = azimuth(pt[0], pt[1], endpts_x, endpts_y)

    return dists, azimuths


def adjust_distances_based_on_azimuth(pairs):
    # Convert list of tuples to a NumPy array for easier manipulation
    data = np.array(pairs)

    # Normalize azimuths to a 0-360 range
    azimuths = data[:, 1] % 360

    # Initialize variables to track the best fit
    best_fit_count = -1
    best_fit_start = None

    # Check ranges starting from each azimuth
    for azimuth in azimuths:
        start = azimuth
        end = (start + 90) % 360

        if end > start:
            count = np.sum((azimuths >= start) & (azimuths < end))
        else:  # Handle wrap-around
            count = np.sum((azimuths >= start) | (azimuths < end))

        # Update if this range fits more azimuths
        if count > best_fit_count:
            best_fit_count = count
            best_fit_start = start

    # Adjust distances based on the identified range
    adjusted_pairs = []
    for distance, azimuth in pairs:
        azimuth_normalized = azimuth % 360
        if best_fit_start is not None:
            end = (best_fit_start + 90) % 360
            if end > best_fit_start:
                if not (best_fit_start <= azimuth_normalized < end):
                    distance = -distance
            else:  # Handle wrap-around
                if not (
                    azimuth_normalized >= best_fit_start
                    or azimuth_normalized < end
                ):
                    distance = -distance
        adjusted_pairs.append((distance, azimuth))

    return adjusted_pairs


def _check_overlap(segment1, segment2):
    # Convert segments to numpy arrays for easier manipulation
    A = np.asarray(sorted(segment1))
    B = np.asarray(sorted(segment2))

    # Determine the maximum start point and minimum end point
    overlap_start = np.max([A[0], B[0]])
    overlap_end = np.min([A[1], B[1]])

    # Calculate the overlap length
    overlap_length = np.max([0, overlap_end - overlap_start])

    # Categorize the overlap
    if overlap_length == 0.0:
        return ("none", 0.0)
    elif A[0] == B[0] and A[1] == B[1]:
        return ("equality", overlap_length)
    elif (A[0] >= B[0] and A[1] <= B[1]) or (B[0] >= A[0] and B[1] <= A[1]):
        return ("full", overlap_length)
    else:
        return ("partial", overlap_length)


def calc_rupture_overlap(trace_1, trace_2, intersection_pt=None):
    if not isinstance(trace_1, Line):
        tr1 = Line([Point(*coords) for coords in trace_1])
        tr2 = Line([Point(*coords) for coords in trace_2])
    else:
        tr1 = trace_1
        tr2 = trace_2

    if intersection_pt is None:
        intersection_pt = find_intersection_angle(tr1, tr2)[0]

    dists_1, az_1 = get_dists_and_azimuths_from_pt(tr1, intersection_pt)
    dists_2, az_2 = get_dists_and_azimuths_from_pt(tr2, intersection_pt)

    dist_az_pairs = list(zip(dists_1, az_1))
    dist_az_pairs.extend(zip(dists_2, az_2))

    dist_az_pairs = adjust_distances_based_on_azimuth(dist_az_pairs)

    dists_1 = [dist_az_pairs[0][0], dist_az_pairs[1][0]]
    dists_2 = [dist_az_pairs[2][0], dist_az_pairs[3][0]]

    overlap_type, overlap_length = _check_overlap(dists_1, dists_2)
    return overlap_type, overlap_length


def _is_strike_slip(rake):
    return (
        (-45.0 < rake < 45.0)
        or (135.0 < rake < 180.0)
        or (-180.0 < rake < -135.0)
    )


def filter_bin_adj_matrix_by_rupture_overlap(
    single_rup_df,
    subfaults,
    binary_adjacence_matrix,
    strike_slip_only: bool = True,
    threshold_overlap: float = 20.0,
    threshold_angle: float = 45.0,
):

    sf_traces = get_trace_from_sf_rupture(single_rup_df, subfaults)
    logging.info("   Getting proximal rup angles")
    rup_angles = get_proximal_rup_angles(sf_traces, binary_adjacence_matrix)
    fault_rake_lookup = {ff[0]['fid']: ff[0]['rake'] for ff in subfaults}
    rakes = {
        i: fault_rake_lookup[ff] for i, ff in enumerate(single_rup_df['fault'])
    }

    if strike_slip_only:
        strike_slip_filter = {}
        for (i, j), _ in rup_angles.items():
            if _is_strike_slip(rakes[i]) and _is_strike_slip(rakes[j]):
                strike_slip_filter[(i, j)] = True
            else:
                strike_slip_filter[(i, j)] = False
    else:
        strike_slip_filter = {k: True for k in rup_angles.keys()}

    logging.info("   Calculating overlap")
    for (i, j), (int_pt, angle) in rup_angles.items():
        if angle < threshold_angle:
            if strike_slip_filter[(i, j)]:
                overlap = calc_rupture_overlap(
                    sf_traces[i], sf_traces[j], intersection_pt=int_pt
                )
                if overlap[1] > threshold_overlap:
                    if issparse(binary_adjacence_matrix):
                        del binary_adjacence_matrix[i, j]
                    else:
                        binary_adjacence_matrix[i, j] = 0

    return binary_adjacence_matrix, rup_angles


def filter_bin_adj_matrix_by_rupture_angle(
    single_rup_df,
    subfaults,
    binary_adjacence_matrix,
    threshold_angle=60.0,
) -> np.ndarray:
    """
    Filter the rupture adjacency matrix by rupture angle, so that only
    adjacent ruptures with angles greater than a threshold angle are retained.
    The rupture adjacency matrix is filtered in-place.

    Parameters
    ----------
    single_rup_df : pd.DataFrame
        DataFrame containing the rupture information. The columns are:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
    subfaults : list of lists of dictionaries
        List of lists of subfault dictionaries. Each list of subfaults
        is derived from a single, contiguous fault.
    binary_adjacence_matrix : np.ndarray
        Binary adjacency matrix.
    threshold_angle : float, optional
        Threshold angle in degrees. The default is 60.0.

    Returns
    -------
    np.ndarray
        Filtered binary adjacency matrix.
    """
    # if angle_type == 'trace':
    #    angle_index = 1
    # elif angle_type == 'dihedral':
    #    angle_index = 0

    sf_meshes = make_sf_rupture_meshes(
        single_rup_df['patches'], single_rup_df['fault'], subfaults
    )

    sf_traces = [get_trace_from_mesh(mesh) for mesh in sf_meshes]

    rup_angles = get_proximal_rup_angles(sf_traces, binary_adjacence_matrix)

    for (i, j), (int_pt, angle) in rup_angles.items():
        # for (i, j), angles in rup_angles.items():
        # if angles[angle_index] < threshold_angle:
        if angle < threshold_angle:
            if issparse(binary_adjacence_matrix):
                del binary_adjacence_matrix[i, j]
            else:
                binary_adjacence_matrix[i, j] = 0

    return binary_adjacence_matrix


def get_multifault_ruptures(
    dist_adj_binary,
    max_dist: float = 10.0,
    check_unique: bool = True,
    max_sf_rups_per_mf_rup: int = 10,
) -> list[list[int]]:
    """
    Get all possible multifault ruptures from a rupture adjacency matrix,
    by finding all of the simple paths from a graph represented by the
    adjacency matrix.

    Parameters
    ----------
    dist_adj_binary : np.ndarray
        Rupture adjacency matrix.
    max_dist : float, optional
        Maximum distance between ruptures in km. The default is 10.0.
    check_unique : bool, optional
        Whether to check for unique ruptures. The default is False.
    max_sf_rups_per_mf_rup : int, optional
        Maximum number of single-fault ruptures per multifault rupture.
        The default is 10.

    Returns
    -------
    list[list[int]]
        List of multifault ruptures. Each rupture is a list of rupture
        indices.
    """
    n_rups = dist_adj_binary.shape[0]

    if max_sf_rups_per_mf_rup == -1:
        max_sf_rups_per_mf_rup = n_rups
    elif max_sf_rups_per_mf_rup > n_rups:
        max_sf_rups_per_mf_rup = n_rups

    if issparse(dist_adj_binary):
        dist_adj_binary = dist_adj_binary.tocoo()

    logging.info("\tmaking graph")
    graph = ig.Graph.Adjacency(dist_adj_binary, mode="undirected")

    logging.info("\tgetting all simple paths")
    paths = []
    for i in range(n_rups):
        for j in range(i + 1, n_rups):
            ps = graph.get_all_simple_paths(
                i, to=j, cutoff=max_sf_rups_per_mf_rup - 1, mode="out"
            )
            paths.extend(ps)
        # ps = graph.get_all_simple_paths(
        #    i, to=None, cutoff=max_sf_rups_per_mf_rup, mode='all'
        # )
        paths.extend(ps)

    paths = [(p) for p in paths if len(p) > 1]
    logging.info("\t%d paths found", len(paths))

    if check_unique:
        paths = [frozenset(p) for p in paths if len(p) > 1]
        paths = set(paths)
        paths = [list(p) for p in paths]

    logging.info("\t%d unique paths found", len(paths))
    return paths


def sparse_to_adjlist(sparse_matrix):
    """
    Converts a scipy.sparse adjacency matrix to an adjacency list.

    Parameters:
    - sparse_matrix: The sparse adjacency matrix (csr_matrix).

    Returns:
    - adj_list: A list of lists, where each sublist represents the neighbors of a vertex.
    """
    adj_list = []
    n = sparse_matrix.shape[0]  # Number of vertices

    for i in range(n):
        # Extract indices of non-zero elements in row i, which are the neighbors
        neighbors = sparse_matrix[[i], :].nonzero()[1].tolist()
        adj_list.append(neighbors)

    return adj_list


def sparse_to_adj_dict_old(sparse_matrix):
    """
    Converts a scipy.sparse adjacency matrix to an adjacency dictionary.

    Parameters:
    - sparse_matrix: The sparse adjacency matrix (csr_matrix).

    Returns:
    - adj_dict: A dictionary, where the keys are the vertices and the values are the neighbors.
    """
    adj_dict = {}
    n = sparse_matrix.shape[0]  # Number of vertices
    if n < 32767:
        int_type = np.int16
    else:
        int_type = np.int32

    for i in range(n):
        # Extract indices of non-zero elements in row i, which are the neighbors
        neighbors = sparse_matrix[[i], :].nonzero()[1].tolist()
        adj_dict[i] = [int_type(neighbor) for neighbor in neighbors]

    return adj_dict


def sparse_to_adj_dict(sparse_matrix: csr_matrix):
    """
    Converts a scipy.sparse adjacency matrix to an adjacency dictionary.

    Parameters:
    - sparse_matrix: The sparse adjacency matrix (csr_matrix).

    Returns:
    - adj_dict: A dictionary, where the keys are the vertices and the values are the neighbors.
    """
    logging.info("\tmaking adjacency dictionary")
    if not isspmatrix_csr(sparse_matrix):
        logging.debug("\t\tconverting adj matrix to CSR")
        mat = sparse_matrix.tocsr()
    else:
        mat = sparse_matrix

    indptr = mat.indptr
    indices = mat.indices
    n = mat.shape[0]

    # Choose an integer dtype for compactness (optional)
    int_type = np.int16 if n < 32767 else np.int32

    # Build the dict – still in Python, but with no expensive matrix ops
    return {
        i: indices[indptr[i] : indptr[i + 1]].astype(int_type).tolist()
        for i in range(n)
    }


# def get_unique_vertex_sets(adj_list, max_vertices=10):
#    """
#    Get all unique vertex sets from an adjacency list.
#
#    Parameters:
#    - adj_list: A list of lists, where each sublist represents the neighbors of a vertex.
#
#    Returns:
#    - vertex_sets: A list of lists, where each sublist is a unique vertex set.
#    """
#    all_subsets = set()
#    n = len(adj_list)  # Number of vertices
#
#    def explore(vertex, current_set):
#        if 1 < len(current_set) <= max_vertices:
#            all_subsets.add(frozenset(current_set))
#
#        if len(current_set) == max_vertices:
#            return
#
#        for neighbor in adj_list[vertex]:
#            if neighbor not in current_set:
#                explore(neighbor, current_set | {neighbor})
#
#    logging.info("\tgetting all contiguous vertex sets")
#    for vertex in range(n):
#        explore(vertex, {vertex})
#
#    vertex_sets = sorted([sorted(list(subset)) for subset in all_subsets])
#    logging.info("\t%d unique vertex sets found", len(vertex_sets))
#    return vertex_sets


def find_connected_subsets_dfs(adj_dict, max_vertices=10):
    all_subsets = set()

    def explore(vertex, current_set):
        if 1 < len(current_set) <= max_vertices:
            # Add the current set as an immutable set to avoid duplicates
            all_subsets.add(frozenset(current_set))

        if len(current_set) == max_vertices:
            return

        for neighbor in adj_dict[vertex]:
            if neighbor not in current_set:
                # Explore further only if the neighbor is not already in the current set
                explore(neighbor, current_set | {neighbor})

    for vertex in adj_dict.keys():
        explore(vertex, {vertex})

    return [list(subset) for subset in all_subsets]


def find_connected_subsets_bfs(adj_dict, max_vertices=10):
    """
    Find all connected subsets of vertices up to a specified size using BFS.

    Parameters:
    - adj_dict: A dictionary where each key is a vertex and the value is a list of neighbors.
    - cutoff: The maximum size of vertex subsets to find.

    Returns:
    - A list of sets, each representing a connected subset of vertices up to the cutoff size.
    """
    all_subsets = set()  # Use a set to store unique subsets

    for start_vertex in adj_dict.keys():
        # Queue items are tuples (vertex, subset) where 'subset' includes 'vertex'
        queue = deque([(start_vertex, frozenset([start_vertex]))])

        while queue:
            vertex, current_set = queue.popleft()

            # Add the current subset to the collection if it's within the size limit
            if 1 < len(current_set) <= max_vertices:
                all_subsets.add(current_set)

            if len(current_set) < max_vertices:
                # Only consider neighbors not already in the current subset
                for neighbor in adj_dict[vertex]:
                    if neighbor not in current_set:
                        new_set = current_set | frozenset([neighbor])
                        queue.append((neighbor, new_set))

    # Convert frozensets back to lists or regular sets if necessary
    return [set(subset) for subset in all_subsets]


def find_connected_components(adj_dict):
    """
    Find connected components in a graph represented as an adjacency dictionary.

    Parameters:
    - adj_dict: A dictionary where each key is a vertex and the value is a list of neighbors.

    Returns:
    - A list of sets, where each set contains the vertices of a connected component.
    """
    visited = set()  # To keep track of visited vertices
    connected_components = []

    def dfs(vertex, component):
        """
        Depth-First Search to explore and mark vertices of the current component.
        """
        visited.add(vertex)
        component.add(vertex)
        for neighbor in adj_dict[vertex]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for vertex in adj_dict.keys():
        if vertex not in visited:
            component = set()
            dfs(vertex, component)
            connected_components.append(component)

    return connected_components


def subgraphs_from_connected_components(adj_dict, connected_components):
    subgraphs = []
    for component in connected_components:
        subgraph = {
            v: [n for n in adj_dict[v] if n in component] for v in component
        }
        subgraphs.append(subgraph)

    return subgraphs


def find_connected_subgraphs(adj_dict, filter=True):
    connected_components = find_connected_components(adj_dict)
    subgraphs = subgraphs_from_connected_components(
        adj_dict, connected_components
    )
    if filter:
        subgraphs = [sg for sg in subgraphs if len(sg) > 1]
    return subgraphs


def find_connected_subsets_dfs_serial(
    adj_dict: Dict[int, Sequence[int]],
    group_of: Mapping[int, int],  #  vertex → group‑id  lookup
    max_vertices: int = 10,
) -> List[List[int]]:
    """
    Enumerate all connected vertex sets (size 2…max_vertices) such that
    no two vertices belong to the same group.

    Parameters
    ----------
    adj_dict
        Undirected adjacency list (neighbours do not have to be sets,
        just iterables).
    group_of
        `vertex  ->  group‑id` mapping.  Only O(1) read access is needed.
    max_vertices
        Largest set size to return.

    Returns
    -------
    list[list[int]]
        Each inner list is a connected, group‑unique vertex set.
    """
    all_subsets: Set[frozenset[int]] = set()

    def explore(
        vertex: int, current_set: Set[int], used_groups: Set[int]
    ) -> None:

        if 1 < len(current_set) <= max_vertices:
            all_subsets.add(frozenset(current_set))

        if len(current_set) == max_vertices:
            return

        for nbr in adj_dict[vertex]:
            if nbr in current_set:  # already inside
                continue
            g = group_of[nbr]
            if g in used_groups:  # would repeat a group
                continue
            explore(
                nbr,
                current_set | {nbr},
                used_groups | {g},
            )

    for v in adj_dict:
        explore(v, {v}, {group_of[v]})

    # convert back to mutable containers if that is what the caller wants
    return [list(s) for s in all_subsets]


def get_multifault_ruptures_fast(
    dist_adj_binary,
    max_sf_rups_per_mf_rup: int = 10,
    rup_groups=None,
    parallel=False,
    min_parallel_subgraphs: int = 0,
) -> list[list[int]]:

    parallel = False

    n = dist_adj_binary.shape[0]  # Number of vertices

    adj_dict = sparse_to_adj_dict(dist_adj_binary)

    logging.info("\tfinding connected subgraphs")
    try:
        subgraphs = find_connected_subgraphs(adj_dict, filter=True)
        logging.info("\t%d connected subgraphs found", len(subgraphs))
    except RecursionError:
        logging.info(
            "\tRecursion depth exceeded; working on whole model instead"
        )
        subgraphs = [adj_dict]

    logging.info("\tgetting all contiguous vertex sets")
    if parallel and len(subgraphs) > min_parallel_subgraphs:
        # vertex_sets = get_multifault_ruptures_parallel(
        #    subgraphs, max_sf_rups_per_mf_rup
        # )
        vertex_sets = find_connected_subsets_parallel(
            dist_adj_binary,  # adj_dict,
            # vertex_sets = find_connected_subsets_parallel_py(
            #    adj_dict,
            rup_groups,
            max_vertices=max_sf_rups_per_mf_rup,
        )
    else:
        vertex_sets = []
        for i, subgraph in enumerate(subgraphs):
            vertex_sets.extend(
                find_connected_subsets_dfs_serial(
                    subgraph, rup_groups, max_sf_rups_per_mf_rup
                )
            )
            logging.info("\t\tfinished subgraph %d", i + 1)
    return vertex_sets


def get_multifault_ruptures_parallel(subgraphs, max_sf_rups_per_mf_rup=10):
    with Pool() as pool:
        vertex_sets = pool.map(
            partial(
                find_connected_subsets_dfs, max_vertices=max_sf_rups_per_mf_rup
            ),
            subgraphs,
        )
    return vertex_sets


# def convert_to_numba_list_of_arrays(adj_list):
#    """
#    Convert a Python list of lists to a Numba typed list of numpy arrays.
#    """
#    # if int_len == 32:
#    #    dtype = np.int32
#    # elif int_len == 16:
#    #    dtype = np.int16
#    numba_list = List()
#    for sublist in adj_list:
#        numba_list.append(np.array(sublist, dtype=np.int32))
#    return numba_list
#
#
# @njit
# def explore(vertex, current_set, adj_list, results, cutoff):
#    # if int_len == 32:
#    #    dtype = np.int32
#    # elif int_len == 16:
#    #    dtype = np.int16
#    dtype = np.int32
#    if 1 < len(current_set) <= cutoff:
#        results.append(np.array(list(current_set), dtype=dtype))
#
#    if len(current_set) == cutoff:
#        return
#
#    for neighbor in adj_list[vertex]:
#        if neighbor not in current_set:
#            new_set = current_set.copy()
#            new_set.add(neighbor)
#            explore(neighbor, new_set, adj_list, results, cutoff)
#
#
# @jit
# def find_connected_subsets(adj_list, cutoff, results):
#    for vertex in range(len(adj_list)):
#        explore(vertex, set([vertex]), adj_list, results, cutoff)
#
#    unique_results = set([frozenset(res) for res in results])
#    return unique_results


def get_multifault_ruptures_numba(
    dist_adj_binary, max_sf_rups_per_mf_rup: int = 10
) -> list[list[int]]:
    # if dist_adj_binary.shape[0] < 32767:
    # if dist_adj_binary.shape[0] < 1:
    #    int_len = 16
    #    int_type = int16
    # else:
    #    int_len = 32
    #    int_type = int32

    adj_list = sparse_to_adjlist(dist_adj_binary)
    numba_adj_list = convert_to_numba_list_of_arrays(adj_list)
    results = List.empty_list(int32[:])
    logging.info("\tgetting all contiguous vertex sets")
    unique_results = find_connected_subsets(
        numba_adj_list, max_sf_rups_per_mf_rup, results
    )
    logging.info("\t%d unique vertex sets found", len(unique_results))
    return [list(res) for res in unique_results]


@jit(nopython=True)
def rdist_to_dist_matrix(
    rdist: RupDistType, nrows: int = -1, ncols: int = -1
) -> np.ndarray:
    """
    Convert a rupture distance array to a rupture distance matrix. If
    `nrows` and `ncols` are not provided, then the maximum row and column
    indices are used to determine the dimensions of the matrix.

    Parameters
    ----------
    rdist : RupDistType
        Rupture distance array.
    nrows : int, optional
        Number of rows in the rupture distance matrix. The default is -1.
    ncols : int, optional
        Number of columns in the rupture distance matrix. The default is -1.

    Returns
    -------
    np.ndarray
        Rupture distance matrix.
    """
    # can't raise informative error in nopython mode
    if nrows == -1:
        nrows = np.max(rdist['r1']) + 1
    else:
        row_max = np.max(rdist['r1'])
        if nrows < row_max + 1:
            raise ValueError
            # raise ValueError(
            #    "nrows (",
            #    nrows,
            #    ") must be larger than the max row index ",
            #    row_max,
            # )

    if ncols == -1:
        ncols = np.max(rdist['r2']) + 1
    else:
        col_max = np.max(rdist['r2'])
        if ncols < col_max + 1:
            raise ValueError
            # raise ValueError(
            #    "ncols (",
            #    ncols,
            #    ") must be larger than the max col index ",
            #    col_max,
            # )

    dist_matrix = np.zeros((nrows, ncols), dtype=np.float32)

    for row in rdist:
        i = int(row[0])
        j = int(row[1])
        dist_matrix[i, j] = row[2]

    return dist_matrix


def get_mf_distances_from_adj_matrix(mf, dist_adj_matrix) -> np.ndarray:
    """
    Get the distances between the ruptures in a multifault rupture.

    Parameters
    ----------
    mf : list of int
        List of rupture indices.
    dist_adj_matrix : np.ndarray
        Rupture adjacency matrix.

    Returns
    -------
    np.ndarray
        Array of distances between the ruptures in the multifault rupture.
    """
    if len(mf) == 1:
        return np.zeros(1)
    distances = np.empty(len(mf) - 1)
    for i in range(len(distances)):
        distances[i] = dist_adj_matrix[mf[i], mf[i + 1]]
    return distances


def get_multifault_rupture_distances(rupture_df, distance_matrix) -> pd.Series:
    """
    Get the distances between the ruptures in a multifault rupture.
    The distances are given as a Pandas Series, where the index is the rupture
    index and the values are the distances.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        DataFrame containing the rupture information. The columns are:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
    distance_matrix : np.ndarray
        Rupture adjacency matrix.

    Returns
    -------
    pd.Series
        Series of distances between the ruptures in the multifault rupture.
    """
    distances = pd.Series(
        data=[
            get_mf_distances_from_adj_matrix(row.ruptures, distance_matrix)
            for row in rupture_df.itertuples()
        ],
        index=rupture_df.index,
    )

    return distances


def get_subfaults_on_each_fault(subfault_df: pd.DataFrame) -> Dict:
    faults = {fid: [] for fid in subfault_df['fid'].unique()}
    for sf_id, subfault in subfault_df.iterrows():
        faults[subfault['fid']].append(str(sf_id))

    faults = {fid: tuple(vals) for fid, vals in faults.items()}

    return faults


def get_full_fault_indices(df: pd.DataFrame) -> pd.Series:
    """
    For each rupture, return:
      - the index of its full-fault rupture (same 'fault'), OR
      - a unique negative ID (-1, -2, ...) for faults that have no full-fault
        rupture.
    """

    # 1. Collect full-fault rupture indices
    full_idx_by_fault = {}
    for row in df[df["full_fault_rupture"]].itertuples():
        idx = row.Index
        f = row.fault
        if f not in full_idx_by_fault:
            full_idx_by_fault[f] = idx

    # 2. Identify faults missing a full-fault rupture
    all_faults = df["fault"].unique()
    no_full_faults = [f for f in all_faults if f not in full_idx_by_fault]

    # 3. Assign negative IDs to faults lacking a full rupture
    neg_id_by_fault = {f: -(i + 1) for i, f in enumerate(no_full_faults)}

    # 4. Build output
    out = []
    for row in df.itertuples():
        idx = row.Index
        f = row.fault
        if f in full_idx_by_fault:
            out.append(full_idx_by_fault[f])  # full-fault rupture index
        else:
            logging.warning(
                f"rup {idx} from fault {f} has no associated full-fault"
                + " rupture"
            )
            out.append(neg_id_by_fault[f])  # negative group ID

    return pd.Series(out, index=df.index, name="full_fault_index")


def get_fault_groups(fault_network):
    fn = fault_network
    partial_rup_to_full_map = get_full_fault_indices(fn['single_rup_df'])

    # these are connected full-fault ruptures (unless partial rups are allowed
    # in the multifault ruptures)
    conn_subs = find_connected_subgraphs(
        sparse_to_adj_dict(fn['bin_dist_mat']), filter=True
    )

    rup_groups = {k: set(v.keys()) for k, v in enumerate(conn_subs)}

    fullfault_to_group = {
        ff: group for group, ff_set in rup_groups.items() for ff in ff_set
    }

    next_group_id = max(fullfault_to_group.values(), default=-1) + 1
    full_fault_group_map = {}

    for rup, full_fault_rup in partial_rup_to_full_map.items():
        if rup in fullfault_to_group:
            pass
        else:
            if full_fault_rup in fullfault_to_group:
                fullfault_to_group[rup] = fullfault_to_group[full_fault_rup]
            else:
                if full_fault_rup in full_fault_group_map:
                    fullfault_to_group[rup] = full_fault_group_map[
                        full_fault_rup
                    ]
                else:
                    fullfault_to_group[rup] = next_group_id
                    full_fault_group_map[full_fault_rup] = next_group_id
                    next_group_id += 1

    fault_groups = fn['rupture_df'].ruptures.apply(
        lambda r: fullfault_to_group[r[0]]
    )

    fn['rupture_df']['fault_group'] = fault_groups
