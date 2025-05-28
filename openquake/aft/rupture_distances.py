# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022-2023 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import logging
from typing import Dict, Sequence, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit, prange, njit
from scipy.spatial.distance import cdist

from openquake.hazardlib.source import PointSource
from openquake.hazardlib.geo.geodetic import spherical_to_cartesian

# typing
from openquake.hazardlib.source import BaseSeismicSource

# A Numpy dtype with fields for rupture indices and distances
RupDistType = np.dtype([("r1", np.int32), ("r2", np.int32), ("d", np.single)])


def calc_min_source_dist(
    source1: BaseSeismicSource, source2: BaseSeismicSource
) -> float:
    """
    Calculates the minimum distance between two sources, based on the polygons
    defining the bounds of the sources.

    Returns a scalar distance in km.
    """
    pts1 = np.array(source1.polygon.coords)
    cpts1 = spherical_to_cartesian(pts1[:, 0], pts1[:, 1])

    pts2 = np.array(source2.polygon.coords)
    cpts2 = spherical_to_cartesian(pts2[:, 0], pts2[:, 1])

    dist_mat = cdist(cpts1, cpts2)
    return np.min(dist_mat)


def get_close_source_pairs_slow(
    source_list: Sequence[BaseSeismicSource],
    dist_threshold: Optional[float] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Calculates the pairwise minimum distance between all sources in
    `source_list` (including each source to itself), and optionally only
    returns those with distances less than `dist_threshold`.

    :param source_list: Sequence of sources.

    :param dist_threshold: Maximum distance (in km) between sources to retain.

    :returns:
        Dictionary with tuples of source IDs for keys, and distances (in km)
        as values.
    """
    source_pair_dists = {
        (s_i.source_id, s_j.source_id): calc_min_source_dist(s_i, s_j)
        for i, s_i in enumerate(tqdm(source_list, leave=True))
        for j, s_j in enumerate(source_list)
    }

    if dist_threshold:
        source_pair_dists = {
            k: v for k, v in source_pair_dists.items() if v <= dist_threshold
        }

    return source_pair_dists


def get_source_points(source):
    if isinstance(source, PointSource):
        return np.array([[source.location.x, source.location.y]])
    else:
        return np.array(source.polygon.coords)


def get_close_source_pairs(
    source_list: Sequence[BaseSeismicSource],
    max_dist: Optional[float] = 1000,
    source_len_limit: Optional[int] = 2000,
    n_procs: Optional[int] = 4,
):

    poly_pts = [get_source_points(source) for source in source_list]
    cart_pts = [
        spherical_to_cartesian(pts[:, 0], pts[:, 1]) for pts in poly_pts
    ]

    source_pair_dists = get_sequence_pairwise_dists(cart_pts, cart_pts)

    logging.info(" filtering source pairs by distance")
    if max_dist:
        source_pair_dists = source_pair_dists[
            source_pair_dists["d"] <= max_dist
        ]

    return source_pair_dists


@njit(fastmath=True, parallel=True)
def calc_pairwise_distances(
    vec_1: np.ndarray, vec_2: np.ndarray
) -> np.ndarray:
    """
    Calculates the pairwise Cartesian distance between two 3D vectors.  Runs in
    parallel over `vec_1`.

    :returns:
        An array of floats of shape (N x M), where N is the length of
    `vec_1` and M is the length of `vec_2`.

    """
    res = np.empty((vec_1.shape[0], vec_2.shape[0]), dtype=vec_1.dtype)
    for i in prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i, j] = np.sqrt(
                (vec_1[i, 0] - vec_2[j, 0]) ** 2
                + (vec_1[i, 1] - vec_2[j, 1]) ** 2
                + (vec_1[i, 2] - vec_2[j, 2]) ** 2
            )

    return res


@njit(parallel=True)
def min_reduce(
    arr: np.ndarray, row_inds: Sequence[int], col_inds: Sequence[int]
) -> np.ndarray:
    """
    Calculates the minima of sub-blocks of `arr` (a 2D array), where the
    sub-block boundaries are given by `row_inds` and `col_inds`. Essentially
    a 2D version of numpy.minimum.reduceat(), and indexing should be the same.

    :param arr:
        Array to be subset and minimized.

    :param row_inds:
        Sequence of the first (starting) row index of each sub-block.

    :param col_inds:
        Sequence of the first (starting) column index of each sub-block.

    :returns:
        A smaller 2D array of the minimum in each sub-block.
    """
    reduced = np.zeros((len(row_inds), len(col_inds)), dtype=arr.dtype)

    ris = np.append(row_inds, arr.shape[0])
    cis = np.append(col_inds, arr.shape[1])

    for i in prange(len(row_inds)):
        r_slice = slice(ris[i], ris[i + 1])
        for j in range(len(col_inds)):

            c_slice = slice(cis[j], cis[j + 1])
            reduced[i, j] = np.min(arr[r_slice, c_slice])

    return reduced


def stack_sequences(sequences: Sequence) -> Tuple[np.ndarray]:
    """
    Takes a sequence (list, array, etc.) of sequences of values, and returns
    an array with the starting indices of each of the original sequences of
    values, and single stacked array of the values themselves.

    In the context of the intended use for this rather general function, each
    sequence contains the XYZ coordinates of points representing a rupture
    mesh. These are all joined into a single array to rapidly calculate
    pairwise distances between many ruptures. The indices denoting the bounds
    between the ruptures must be preserved, and so are returned as well.

    :param sequences:
        Sequence of sequences of values to be stacked. These must have the same
        number of columns.

    :returns:
        An array with the starting indices of each original sequence (in
        int32), and a single stacked array with the original values (in
        float32).

    >>> stack_sequences(([[0,0],[0,0],[0,0]],
                         [[1, 1],[1,1]],
                         [[2,2],[2,2],[2,2]]))
    (array([0, 3, 5], dtype=int32),
     array([[0., 0.],
            [0., 0.],
            [0., 0.],
            [1., 1.],
            [1., 1.],
            [2., 2.],
            [2., 2.],
            [2., 2.]], dtype=float32))

    """
    array_stack = np.single(np.vstack(sequences))

    id_stack = np.cumsum([len(rup_xyz) for rup_xyz in sequences][:-1])
    id_stack = np.int32(np.insert(id_stack, 0, 0))

    return id_stack, array_stack


def split_rows(
    row_ids: Sequence[int], stacked_array: np.ndarray, n_splits: int = 20
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Splits an array into `n_splits` chunks along rows, preserving the
    boundaries from `stack_sequences`. Splits are as evenly distributed as
    possible given these boundaries. If `n_splits` is greater than
    len(row_ids), then len(row_ids) blocks are returned.

    :param row_ids:
        Sequence of indices denoting the first row of blocks of the pre-stacked
        sequences

    :param stacked_array:
        Array to be split

    :param n_splits:
        Number of splits.

    :returns:
       Dictionary of dictionaries. Outer dictionary has keys representing
       the first `row_id` of the split. Each inner dict (value) has an
       `array_stack` item split from `stacked_array`, and a `row_idxs` item
       that gives the boundaries of each block within the `array_stack`.
    """
    n_rows = len(stacked_array)

    splits = np.array_split(np.arange(n_rows), n_splits)
    splits = [split for split in splits if len(split) > 0]
    split_starts = [split[0] for split in splits]

    closest_first_inds = [
        np.argmin(np.abs(row_ids - split_start))
        for split_start in split_starts
    ]

    closest_first_inds = np.unique(closest_first_inds)

    data_splits = {}

    for i in range(len(closest_first_inds)):
        first_row = closest_first_inds[i]
        data_splits[first_row] = {}
        this = data_splits[first_row]

        if i < len(closest_first_inds) - 1:
            start = closest_first_inds[i]
            stop = closest_first_inds[i + 1]
            block_idx = row_ids[start:stop]

            this["array_stack"] = stacked_array[
                row_ids[start] : row_ids[stop], :
            ]

            this["row_idxs"] = block_idx - block_idx[0]

        else:
            start = closest_first_inds[i]
            block_idx = row_ids[start:]

            this["array_stack"] = stacked_array[row_ids[start] :, :]
            this["row_idxs"] = block_idx - block_idx[0]

    return data_splits


def get_min_rup_dists(
    dists: np.ndarray,
    row_inds: Sequence[int],
    col_inds: Sequence[int],
    row_offset: int = 0,
    col_offset: int = 0,
) -> np.ndarray:
    """
    Gets the minimum distance between ruptures.

    :param dists:
        Matrix of pairwise distances between all points in two sets of
        ruptures.

    :param row_inds:
        Array of starting indices for the rows of each rupture (from the first
        source). See `min_reduce` for details.

    :param col_inds:
        Array of starting indices for the columns of each rupture (from the
        second source). See `min_reduce` for details.

    :param row_offset:

    :param col_offset:

    :returns:
        Array of rupture distances (of RupDistType type).
    """
    min_reduced = min_reduce(dists, row_inds, col_inds)

    r1s = (
        np.tile(
            np.arange(len(row_inds), dtype=np.int32), (len(col_inds), 1)
        ).ravel(order="F")
        + row_offset
    )
    r2s = (
        np.tile(
            np.arange(len(col_inds), dtype=np.int32), (1, len(row_inds))
        ).ravel(order="C")
        + col_offset
    )

    min_dists = np.zeros(len(r1s), dtype=RupDistType)
    min_dists["r1"] = r1s
    min_dists["r2"] = r2s
    min_dists["d"] = min_reduced.ravel()

    return min_dists


@njit(fastmath=True)
def check_dists_by_mag(
    dist: Union[float, np.ndarray],
    mag: Union[float, np.ndarray],
    dist_constant: float = 4.0,
) -> Union[bool, np.ndarray]:
    """
    Checks to see that a distance value `dist` is less than the rupture
    magnitude `mag` scaled by a length-to-distance scaling relationship
    from Leonard (2014) times a constant (`dist_constant`). Arguments can be
    any mix of scalars or arrays of the same length/shape.

    Note that because the function is jit compiled with Numba, we can't use
    the scaling relationships defined in `openquake.hazardlib.scalerel`
    or any other method that allows for the selection of different
    scaling relationships.

    :param mag:
        Scalar or array of moment magnitudes.

    :param dist:
        Scalar or array of distances (in km).

    :param dist_constant:
        Coefficient to multiply the length (scaled by magnitude) as the
        maximum distance threshold.

    :returns:
        Boolean scalar or array.
    """
    return dist <= dist_constant * 10.0 ** (-2.943 + 0.681 * mag)


def filter_dists_by_mag(
    min_rup_dists: RupDistType,
    mags: Sequence[float],
    dist_constant: float = 4.0,
) -> np.ndarray:
    """
    Filters pairwise distance arrays (from `get_min_rup_dists`) based on
    the magnitude and a magnitude-length scaling.

    :param min_rup_dists:
        Array (of RupDistType) with rupture indices and distances; this
        should be produced by `min_reduce`.

    :param mags:
        Sequence of moment magnitudes for each rupture in Source 1.

    :param dist_constant:
        Coefficient to multiply the length (scaled by magnitude) as the
        maximum distance threshold.

    :returns:
        Array (of RupDistType) pairwise rupture distances that are
        less than or equal to the rupture length for the source rupture
        (from Source 1) times a `dist_constant`.
    """
    # It is possible that by making a new, jit-compiled function for just
    # the for loop part, the rest can be done with simple numpy and then
    # we could pass a scaling relationship function to this function as
    # an argument that would then get passed to `check_dists_by_mag` and
    # not result in any real slowdown.  This would give much more flexibility
    # for the filtering.

    mags_array = np.empty(len(min_rup_dists), dtype=np.float32)

    for i, rup_i in enumerate(min_rup_dists["r1"]):
        mags_array[i] = mags[rup_i]

    return min_rup_dists[
        check_dists_by_mag(
            min_rup_dists["d"], mags_array, dist_constant=dist_constant
        )
    ]


def filter_dists_by_dist(
    min_rup_dists: RupDistType,
    dist_threshold=4.0,
) -> np.ndarray:
    pass


def get_sequence_pairwise_dists(seq_1, seq_2, max_block_ram=20.0):
    r_id_1, array_stack_1 = stack_sequences(seq_1)
    r_id_2, array_stack_2 = stack_sequences(seq_2)

    if (
        array_stack_1.shape[0] * array_stack_2.shape[0] * 8
        < max_block_ram * 1e9
    ):
        dists = calc_pairwise_distances(array_stack_1, array_stack_2)
        min_rup_dists = get_min_rup_dists(dists, r_id_1, r_id_2)

    else:
        min_rup_dist_list = []
        n_splits = int(
            np.ceil(
                (array_stack_1.shape[0] * array_stack_2.shape[0] * 8)
                / (max_block_ram * 1e9)
            )
        )
        row_splits = split_rows(r_id_1, array_stack_1, n_splits)

        for start_rup, row_split in row_splits.items():
            dists = calc_pairwise_distances(
                row_split["array_stack"], array_stack_2
            )
            min_rup_dist_list.append(
                get_min_rup_dists(
                    dists, row_split["row_idxs"], r_id_2, row_offset=start_rup
                )
            )
        min_rup_dists = np.hstack(min_rup_dist_list)

    return min_rup_dists


def get_rup_dist_pairs(
    source_id_0: str,
    source_id_1: str,
    rup_df: pd.DataFrame,
    source_groups: pd.core.groupby.generic.DataFrameGroupBy,
    dist_constant: float = 4.0,
    max_block_ram=20.0,
) -> np.ndarray:
    """
    Calculates pairwise rupture distances for pairs of sources. The pairwise
    distance matrix that results is asymmetrical, as it is filtered
    by the distance from the source rupture to the receiver rupture times
    a constant (`dist_constant`); this distance depends on the magnitude of the
    source rupture.

    The initial pairwise distance calculation is done for each point in
    the mesh of each rupture, which can be extremely RAM intensive for
    sources with a big number of large ruptures (i.e. subduction zones).
    In this instance, these calculations are broken into blocks, of
    of approximate size `max_block_ram`.

    :param source_id_0: This is the `source_id` for the... source source.

    :param source_id_1: This is the `source_id` for the target source.

    :param rup_df:
        DataFrame that contains all of the ruptures from all
        sources as rows, with `rupture`, `source`, `xyz` (from
        `rupture.surface.mesh.xyz`), and `mag` columns.

    :param source_groups:
        This is a Pandas Groupby object that describes the
        grouping of `rup_gdf` by the `source` column.

    :param dist_constant:
        Coefficient to multiply the length (scaled by magnitude) as the
        maximum distance threshold.

    :param max_block_ram:
        Maximum about of RAM (in GB) to be used for a single block
        of pairwise distances. This value is to be based on the
        user's RAM and can vary greatly depending on the system
        (old laptop vs. cluster).

    :returns:
        Pairwise distance matrix for each rupture in both sources,
        which is filtered to only retain the close ruptures. This
        results in a sparse matrix with a format like `[r1, r2, distance]`
        for each row, where `r1` and `r1` are the indices of the
        ruptures for `source_id_0` and `source_id_1`, respectively.
    """
    rups1 = rup_df.iloc[source_groups.groups[source_id_0]]
    rups2 = rup_df.iloc[source_groups.groups[source_id_1]]

    # r_id_1, array_stack_1 = stack_sequences(rups1.xyz)
    # r_id_2, array_stack_2 = stack_sequences(rups2.xyz)

    # if (
    #    array_stack_1.shape[0] * array_stack_2.shape[0] * 8
    #    < max_block_ram * 1e9
    # ):
    #    dists = calc_pairwise_distances(array_stack_1, array_stack_2)
    #    min_rup_dists = get_min_rup_dists(dists, r_id_1, r_id_2)

    # else:
    #    min_rup_dist_list = []
    #    n_splits = int(
    #        np.ceil(
    #            (array_stack_1.shape[0] * array_stack_2.shape[0] * 8)
    #            / (max_block_ram * 1e9)
    #        )
    #    )
    #    row_splits = split_rows(r_id_1, array_stack_1, n_splits)

    #    for start_rup, row_split in row_splits.items():
    #        dists = calc_pairwise_distances(
    #            row_split["array_stack"], array_stack_2
    #        )
    #        min_rup_dist_list.append(
    #            get_min_rup_dists(
    #                dists, row_split["row_idxs"], r_id_2, row_offset=start_rup
    #            )
    #        )
    #    min_rup_dists = np.hstack(min_rup_dist_list)

    min_rup_dists = get_sequence_pairwise_dists(
        rups1.xyz, rups2.xyz, max_block_ram=max_block_ram
    )

    min_rup_dists = filter_dists_by_mag(
        min_rup_dists, rups1.mag.values, dist_constant=dist_constant
    )

    return min_rup_dists


def process_source_pair(
    source_pair: Tuple[str],
    rup_adj_dict: Union[Dict[str, Dict[str, np.ndarray]], h5py.Dataset],
    rup_df: pd.DataFrame,
    source_groups: pd.core.groupby.generic.DataFrameGroupBy,
    h5_file: Optional[str] = None,
    dist_constant: float = 4.0,
    max_block_ram: float = 20.0,
) -> None:
    """
    Calculates the pairwise rupture distances for a pair of sources, and
    saves the results in the `rup_adj_dict` (which is modified in-place).

    :param source_pair:
        Tuple of strings holding the `source_id`s of the source pair.

    :param rup_adj_dict:
        Dictionary (or HDF5 dataset) that holds the pairwise rupture
        distances. This has the form
        `{source_0: {source_1: dist_matrix, source_2: dist_matrix}}`.

    :param rup_df:
        DataFrame that contains all of the ruptures from all
        sources as rows, with `rupture`, `source`, `xyz` (from
        `rupture.surface.mesh.xyz`), and `mag` columns.

    :param source_groups:
        This is a Pandas Groupby object that describes the
        grouping of `rup_gdf` by the `source` column.

    :param h5_file:
        Optional filepath of HDF5 file that holds the rup_adj_dict. This
        function does not open the HDF5 file, but simply uses it as a flag
        to determine how to treat the `rup_adj_dict`.  If `h5_file == None`,
        the `rup_adj_dict` is treated as an in-memory Python dictionary,
        and if `h5_file != None`, then `rup_adj_dict` is treated as an
        HDF5 dataset.

    :param dist_constant:
        Coefficient to multiply the length (scaled by magnitude) as the
        maximum distance threshold.

    :param max_block_ram:
        Maximum about of RAM (in GB) to be used for a single block
        of pairwise distances. This value is to be based on the
        user's RAM and can vary greatly depending on the system
        (old laptop vs. cluster).

    :returns:
        None. `rup_adj_dict` is modified in-place.
    """
    source_id_0 = source_pair[0]
    source_id_1 = source_pair[1]

    min_rup_dists = get_rup_dist_pairs(
        source_id_0,
        source_id_1,
        rup_df=rup_df,
        source_groups=source_groups,
        dist_constant=dist_constant,
        max_block_ram=max_block_ram,
    )

    if source_id_0 not in rup_adj_dict.keys():
        if h5_file:
            rup_adj_dict.create_group(source_id_0)
        else:
            rup_adj_dict[source_id_0] = {}
    rup_adj_dict[source_id_0][source_id_1] = min_rup_dists


def calc_rupture_adjacence_dict_all_sources(
    source_pairs: Sequence[Tuple[str]],
    rup_df: pd.DataFrame,
    source_groups: pd.core.groupby.generic.DataFrameGroupBy,
    h5_file: Optional[str] = None,
    dist_constant: float = 4.0,
    max_block_ram: float = 20.0,
) -> Union[Dict[str, Dict[str, np.ndarray]], None]:
    """
    Calculates pairwise distances for ruptures from a sequence of source pairs.
    To save memory, these distances are filtered with a magnitude-distance
    filter, so that only distances less than the 'source' rupture length times
    the `dist_constant` (where the rupture length is determined based on
    the rupture magnitude and a scaling relationship) are saved.

    The results are a nested dictionary or HDF5 file, with a structure like:
    `{source_0: {source_1: dist_matrix, source_2: dist_matrix}}`.

    If an HDF5 file is given, the function will write all of the results
    to that file; otherwise, the function will return an in-memory dictionary.

    The initial pairwise distance calculation is done for each point in
    the mesh of each rupture (per source pair), which can be extremely RAM
    intensive for sources with a big number of large ruptures (i.e.
    subduction zones). In this instance, these calculations are broken into
    blocks, of of approximate size `max_block_ram`.


    :param source_pairs:
        Sequence of pairs (tuples) of `source_id`s for each pair of sources
        to calculate pairwise rupture distances for.

    :param rup_df:
        DataFrame that contains all of the ruptures from all
        sources as rows, with `rupture`, `source`, `xyz` (from
        `rupture.surface.mesh.xyz`), and `mag` columns.

    :param source_groups:
        This is a Pandas Groupby object that describes the
        grouping of `rup_gdf` by the `source` column.

    :param h5_file:
        Optional filepath of HDF5 file that holds the pairwise distances.
        If this is `None`, then the function will return an in-memory
        dictionary.

    :param dist_constant:
        Coefficient to multiply the length (scaled by magnitude) as the
        maximum distance threshold.

    :param max_block_ram:
        Maximum about of RAM (in GB) to be used for a single block
        of pairwise distances. This value is to be based on the
        user's RAM and can vary greatly depending on the system
        (old laptop vs. cluster).
    """
    if h5_file:
        rup_adj_dict = h5py.File(h5_file, "a")
    else:
        rup_adj_dict = {}

    pbar = tqdm(source_pairs)

    for source_pair in pbar:
        source_pair_str = f"{source_pair[0]}, {source_pair[1]}"
        process_source_pair(
            source_pair,
            rup_adj_dict,
            rup_df,
            source_groups,
            h5_file=h5_file,
            dist_constant=dist_constant,
        )
        pbar.set_postfix({"sp": source_pair_str})

    if h5_file:
        rup_adj_dict.close()

    if not h5_file:
        return rup_adj_dict
