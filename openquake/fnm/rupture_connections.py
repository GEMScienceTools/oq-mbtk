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
from openquake.fnm.fault_modeler import (
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
    fault_bb_dists = get_bounding_box_distances(surfaces, max_dist=max_dist)

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
        return np.vstack(
            [single_fault_subfaults[i]['surface'].mesh.xyz for i in rupture]
        )


def get_single_fault_rups(
    subfaults, subfault_index_start: int = 0
) -> pd.DataFrame:
    """
    Get all possible ruptures from a single fault.

    Parameters
    ----------
    subfaults : list of dictionaries
        List of subfault dictionaries.
    subfault_index_start : int, optional
        Index of the first subfault. The default is 0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the rupture information. The columns are:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
    """
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


def get_all_single_fault_rups(
    all_subfaults,
) -> tuple[list[list[int]], pd.DataFrame]:
    """
    Get all possible single-fault ruptures from a set of subfaults.

    Parameters
    ----------
    all_subfaults : list of lists of dictionaries
        List of lists of subfault dictionaries. Each list of subfaults
        is derived from a single, contiguous fault.

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
    max_dist: Optional[float] = 20.0,
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

    # increasing distance to make up for discrepancy between bb dist and
    # actual rupture distances; this is a filtering step.
    fault_dists = get_close_faults(faults, max_dist=max_dist * 1.5)

    fault_dists = {(i, j): d for i, j, d in fault_dists}

    dist_adj_matrix = np.zeros((nrups, nrups), dtype=np.float32)

    if max_dist is None:
        max_dist = np.inf

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
                if (i, j) not in fault_dists:
                    pass
                elif fault_dists[(i, j)] <= max_dist * 1.5:
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

    dist_adj_matrix[dist_adj_matrix > max_dist] = 0.0

    return single_fault_rup_df, dist_adj_matrix


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


def get_proximal_rup_angles(
    sf_meshes, binary_distance_matrix
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
    binary_distance_matrix : np.ndarray
        Binary adjacency matrix.

    Returns
    -------
    dict[tuple[int, int], tuple[float, float]]
        Dictionary of rupture angles. The keys are tuples of rupture indices,
        and the values are tuples of rupture angles in degrees.
    """
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
    angle_type : str, optional
        Type of angle to use for filtering. The default is 'trace', but
        'dihedral' is also supported.

    Returns
    -------
    np.ndarray
        Filtered binary adjacency matrix.
    """
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
) -> list[list[int]]:
    """
    Get all possible multifault ruptures from a rupture adjacency matrix,
    by finding all of the simple paths from a graph represented by the
    adjacency matrix.

    Parameters
    ----------
    dist_adj_matrix : np.ndarray
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
    n_rups = dist_adj_matrix.shape[0]

    if max_sf_rups_per_mf_rup == -1:
        max_sf_rups_per_mf_rup = n_rups
    elif max_sf_rups_per_mf_rup > n_rups:
        max_sf_rups_per_mf_rup = n_rups

    dist_adj_binary = make_binary_adjacency_matrix(dist_adj_matrix, max_dist)

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