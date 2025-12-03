# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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
import time
import logging

import pandas as pd
import numpy as np

from scipy.sparse import dok_array, issparse

import numba
from numba import jit, prange
from numba.typed import Dict, List
from numba.core import types

from openquake.fnm.inversion.utils import slip_vector_azimuth

from openquake.fnm.fault_modeler import (
    make_sf_rupture_meshes,
    get_trace_from_mesh,
)

from openquake.fnm.rupture_connections import (
    get_multifault_rupture_distances,
    get_proximal_rup_angles,
)


def logistic(x, k=1.0, x0=0.0, L=1.0):
    return L / (1 + np.exp(-k * (x - x0)))


def compact_cosine_sigmoid(angle, midpoint):
    cutoff = 2 * midpoint
    is_scalar = np.isscalar(angle)
    angle_arr = np.asarray(angle, dtype=float)
    vals = 0.5 * (1 + np.cos(np.pi * angle_arr / cutoff))
    vals = np.where(angle_arr >= cutoff, 0.0, vals)
    if is_scalar:
        return float(vals)
    return vals


def connection_angle_plausibility(
    connection_angles,
    function_type="cosine",
    no_connection_val=1,
    midpoint=90.0,
):
    conns = np.array(connection_angles)
    conns[conns == no_connection_val] = 0.0

    if function_type == "cosine":
        plausibilities = compact_cosine_sigmoid(conns, midpoint)

    else:
        raise NotImplementedError(
            f"Function type {function_type} not implemented."
        )

    total_plaus = np.prod(plausibilities)
    return total_plaus


def find_decay_exponent(y, x):
    """
    Finds the exponent (lambda) of the decay equation y = e^(-lambda x)
    given y and x.

    Parameters:

    y (float): The value of y in the equation. Must be positive.

    x (float): The value of x in the equation. Must be non-zero.


    Returns:

    float: The value of lambda in the equation.

    """
    if y <= 0 or x == 0:
        raise ValueError("y must be positive and x must be non-zero.")

    return -np.log(y) / x


def connection_distance_plausibility(
    connection_distances,
    function_type="exponent",
    no_connection_val=-1,
    midpoint=None,
):
    conns = np.array(connection_distances)
    conns[conns == no_connection_val] = 0.0
    if midpoint is None:
        midpoint = dist_threshold

    if function_type == "exponent":
        decay_exp = np.log(2.0) / midpoint
        plausibilities = np.exp(-decay_exp * conns)

    elif function_type == "linear":
        plausibilities = 1 - (conns / midpoint)
    elif function_type == "cosine":
        plausibilities = compact_cosine_sigmoid(conns, midpoint)

    else:
        raise NotImplementedError(
            f"Function type {function_type} not implemented."
        )

    total_plaus = np.prod(plausibilities)
    return total_plaus


def slip_azimith_plausibility(
    slip_azimuths,
    function_type="cosine",
    midpoint=90.0,
):
    if midpoint is None:
        return 1.0

    if len(slip_azimuths) == 1:
        return 1.0

    slip_azimuths = np.array(sorted(slip_azimuths))
    slip_azimuth_diffs = np.diff(slip_azimuths)

    if function_type == "cosine":
        plausibilities = compact_cosine_sigmoid(
            np.abs(slip_azimuth_diffs), midpoint
        )

    else:
        raise NotImplementedError(
            f"Function type {function_type} not implemented."
        )

    total_prob = np.prod(plausibilities)
    return total_prob


def _padded_array_from_sequences(
    sequences, dtype=np.float64, fill_value=0.0, replace=None
):
    seqs = list(sequences)
    n_rups = len(seqs)
    max_len = 0
    for seq in seqs:
        if seq is None:
            continue
        if not isinstance(seq, (list, tuple, np.ndarray)):
            seq = [seq]
        L = len(seq)
        if L > max_len:
            max_len = L

    arr = np.full((n_rups, max_len), fill_value, dtype=dtype)
    if max_len == 0:
        return arr

    for i, seq in enumerate(seqs):
        if seq is None:
            continue
        if not isinstance(seq, (list, tuple, np.ndarray)):
            seq = [seq]
        seq_arr = np.asarray(seq, dtype=dtype)
        if replace is not None:
            old_val, new_val = replace
            seq_arr = seq_arr.copy()
            seq_arr[seq_arr == old_val] = new_val
        if seq_arr.size:
            arr[i, : seq_arr.size] = seq_arr

    return arr


def _to_binary_sparse_matrix(matrix):
    if matrix is None:
        return None
    if issparse(matrix):
        binary = matrix.todok().copy()
        for key in list(binary.keys()):
            binary[key] = 1
        return binary

    dense = np.asarray(matrix)
    if dense.size == 0:
        return dok_array(dense.shape, dtype=np.int8)

    rows, cols = np.nonzero(dense)
    binary = dok_array(dense.shape, dtype=np.int8)
    for r, c in zip(rows, cols):
        binary[int(r), int(c)] = 1
    return binary


def _matrix_to_rupture_series(rupture_df, matrix):
    if matrix is None:
        return None

    mat = matrix.todok() if issparse(matrix) else np.asarray(matrix)
    is_sparse = issparse(mat)
    seqs = []
    for row in rupture_df.itertuples():
        rup_indices = getattr(row, "ruptures")
        if len(rup_indices) == 1:
            seqs.append(np.zeros(1, dtype=np.float64))
            continue

        vals = np.empty(len(rup_indices) - 1, dtype=np.float64)
        for idx in range(len(rup_indices) - 1):
            i, j = rup_indices[idx], rup_indices[idx + 1]
            if is_sparse:
                vals[idx] = float(mat[i, j])
            else:
                vals[idx] = float(mat[i][j])
        seqs.append(vals)

    return pd.Series(seqs, index=rupture_df.index)


def _build_angle_matrix_from_pairs(single_rup_df, subfaults, binary_matrix):
    if binary_matrix is None:
        return None

    binary = (
        binary_matrix.todok().copy()
        if issparse(binary_matrix)
        else _to_binary_sparse_matrix(binary_matrix)
    )

    sf_meshes = make_sf_rupture_meshes(
        single_rup_df['patches'], single_rup_df['fault'], subfaults
    )
    sf_traces = [get_trace_from_mesh(mesh) for mesh in sf_meshes]

    rup_angles = get_proximal_rup_angles(sf_traces, binary)

    angle_matrix = dok_array(binary.shape, dtype=np.float64)
    for (i, j), angle_data in rup_angles.items():
        angle_val = (
            angle_data[1] if isinstance(angle_data, tuple) else angle_data
        )
        angle_matrix[i, j] = angle_val
        angle_matrix[j, i] = angle_val

    return angle_matrix


def get_single_rupture_plausibilities(
    rupture,
    connection_angle_function="cosine",
    connection_distance_function="exponent",
    slip_azimuth_function="cosine",
    connection_angle_threshold=1.0,
    connection_distance_midpoint=15.0,
    connection_angle_midpoint=90.0,
    slip_azimuth_midpoint=90.0,
):
    plausibilities = {}

    if connection_angle_midpoint is not None:
        plausibilities["connection_angle"] = connection_angle_plausibility(
            rupture["connection_angles"],
            function_type=connection_angle_function,
            no_connection_val=connection_angle_threshold,
            midpoint=connection_angle_midpoint,
        )
    else:
        plausibilities["connection_angle"] = 1.0

    if connection_distance_midpoint is not None:
        plausibilities["connection_distance"] = (
            connection_distance_plausibility(
                rupture["connection_distances"],
                function_type=connection_distance_function,
                midpoint=connection_distance_midpoint,
            )
        )
    else:
        plausibilities["connection_distance"] = 1.0

    if slip_azimuth_midpoint is not None:
        plausibilities["slip_azimuth"] = slip_azimith_plausibility(
            rupture["slip_azimuths"],
            function_type=slip_azimuth_function,
            midpoint=slip_azimuth_midpoint,
        )
    else:
        plausibilities["slip_azimuth"] = 1.0

    plausibilities["total"] = np.prod(list(plausibilities.values()))

    return plausibilities


def get_rupture_plausibilities(
    rupture_df,
    distances=None,
    distance_matrix=None,
    connection_angle_function="cosine",
    connection_distance_function="exponent",
    slip_azimuth_function="cosine",
    connection_angle_threshold=1.0,
    angles=None,
    angle_matrix=None,
    binary_distance_matrix=None,
    single_rup_df=None,
    subfaults=None,
    connection_angle_midpoint=90.0,
    connection_distance_midpoint=15.0,
    slip_azimuth_midpoint=90.0,
):

    if distances is None:
        if distance_matrix is None:
            raise ValueError(
                "Either distances or distance_matrix must be provided."
            )
        distances = get_multifault_rupture_distances(
            rupture_df, distance_matrix
        )

    # distances is expected to be a 1-D pandas Series of per-rupture sequences
    # (e.g. lists/arrays of connection distances).
    # We build a padded 2-D array so we can do vectorized math.
    no_connection_val = -1.0

    # Convert to a plain list of sequences to avoid pandas overhead
    dist_seqs = list(distances.values)
    n_rups = len(dist_seqs)
    columns = [
        "connection_angle",
        "connection_distance",
        "slip_azimuth",
        "total",
    ]

    if n_rups == 0:
        return pd.DataFrame(
            index=rupture_df.index,
            columns=columns,
            dtype=np.float64,
        )

    dist_arr = _padded_array_from_sequences(
        dist_seqs,
        dtype=np.float64,
        fill_value=0.0,
        replace=(no_connection_val, 0.0),
    )

    if connection_distance_midpoint is None or dist_arr.shape[1] == 0:
        conn_total = np.ones(n_rups, dtype=np.float64)
    else:
        # Vectorized connection-distance plausibility
        if connection_distance_function == "exponent":
            decay_exp = np.log(2.0) / connection_distance_midpoint
            conn_plaus = np.exp(-decay_exp * dist_arr)
        elif connection_distance_function == "linear":
            conn_plaus = 1.0 - (dist_arr / connection_distance_midpoint)
        elif connection_distance_function == "cosine":
            conn_plaus = compact_cosine_sigmoid(
                dist_arr, connection_distance_midpoint
            )
        else:
            raise NotImplementedError(
                f"Function type {connection_distance_function} not implemented."
            )

        conn_total = np.prod(conn_plaus, axis=1)

    angle_series = angles
    angle_matrix_input = angle_matrix

    if angle_series is None and angle_matrix_input is None:
        if "connection_angles" in rupture_df.columns:
            angle_series = rupture_df["connection_angles"]
        else:
            binary_for_angles = None
            if binary_distance_matrix is not None:
                binary_for_angles = _to_binary_sparse_matrix(
                    binary_distance_matrix
                )
            elif distance_matrix is not None:
                binary_for_angles = _to_binary_sparse_matrix(distance_matrix)

            if (
                binary_for_angles is not None
                and single_rup_df is not None
                and subfaults is not None
            ):
                angle_matrix_input = _build_angle_matrix_from_pairs(
                    single_rup_df, subfaults, binary_for_angles
                )

    if angle_series is None and angle_matrix_input is not None:
        if issparse(angle_matrix_input):
            angle_matrix_input = angle_matrix_input.todok()
        else:
            angle_matrix_input = dok_array(
                np.asarray(angle_matrix_input), dtype=np.float64
            )
        angle_series = _matrix_to_rupture_series(
            rupture_df, angle_matrix_input
        )

    if angle_series is not None and connection_angle_midpoint is not None:
        angle_arr = _padded_array_from_sequences(
            list(angle_series.values),
            dtype=np.float64,
            fill_value=connection_angle_threshold,
        )

        if angle_arr.size == 0 or angle_arr.shape[1] == 0:
            conn_angle_total = np.ones(n_rups, dtype=np.float64)
        else:
            angle_arr[angle_arr == connection_angle_threshold] = 0.0
            if connection_angle_function == "cosine":
                conn_angle_plaus = compact_cosine_sigmoid(
                    angle_arr, connection_angle_midpoint
                )
            else:
                raise NotImplementedError(
                    f"Function type {connection_angle_function} not implemented."
                )
            conn_angle_total = np.prod(conn_angle_plaus, axis=1)
    else:
        conn_angle_total = np.ones(n_rups, dtype=np.float64)

    plausibilities = pd.DataFrame(
        index=rupture_df.index,
        columns=columns,
        dtype=np.float64,
    )
    plausibilities["connection_angle"] = conn_angle_total
    plausibilities["connection_distance"] = conn_total

    # rupture_df["slip_azimuth"] is assumed to hold per-rupture sequences
    if slip_azimuth_midpoint is not None:
        plausibilities["slip_azimuth"] = rupture_df["slip_azimuth"].apply(
            slip_azimith_plausibility,
            function_type=slip_azimuth_function,
            midpoint=slip_azimuth_midpoint,
        )
    else:
        plausibilities["slip_azimuth"] = 1.0

    plausibilities["total"] = (
        plausibilities["connection_angle"]
        * plausibilities["connection_distance"]
        * plausibilities["slip_azimuth"]
    )

    return plausibilities


KeyType = types.UniTuple(types.int64, 2)


# @jit(nopython=False)
@jit(nopython=True, parallel=True)
def calculate_similarities(ruptures, non_zero_pairs):
    similarities = Dict.empty(
        key_type=KeyType,
        value_type=types.float64,
    )
    for k in prange(non_zero_pairs.shape[0]):
        i, j = non_zero_pairs[k]
        # Extracting the subsections and removing the -1 padding values
        rup_i = ruptures[i]
        rup_j = ruptures[j]

        # Calculating the common subsections, max length, and similarity
        common_subsections = np.intersect1d(rup_i, rup_j).shape[0]
        max_len = max(rup_i.shape[0], rup_j.shape[0])
        similarity = common_subsections / max_len
        if similarity > 0.0:
            similarities[(i, j)] = similarity

    return similarities


ListType = types.ListType(types.int64)


@jit(nopython=True)
def get_non_zero_pairs(ruptures):
    subsection_map = Dict.empty(
        key_type=types.int64,
        value_type=ListType,
    )

    for rup_index in range(len(ruptures)):
        for subsection in ruptures[rup_index]:
            if subsection not in subsection_map:
                subsection_map[subsection] = List.empty_list(types.int64)
            subsection_map[subsection].append(rup_index)

    non_zero_pairs = []
    for rupture_indices in subsection_map.values():
        n = len(rupture_indices)
        for a in range(n):
            for b in range(a + 1, n):
                non_zero_pairs.append((rupture_indices[a], rupture_indices[b]))

    return np.array(non_zero_pairs, dtype=np.int64)


def get_rup_similarity_matrix(rup_df):
    print("Calculating rupture similarity matrix...")
    print("preprocessing data")

    n_rups = rup_df.shape[0]
    n_rup_str = len(str(n_rups))
    cols = 70  # just for printing

    if n_rups <= cols:
        unit = 1
    else:
        unit = int(np.floor(n_rups / cols))

    n_dots = 0

    ruptures = List()
    for i, rup_list in enumerate(rup_df["subsections"].values):
        if i % unit == 0:
            n_dots += 1
            i_str = (
                "." * n_dots
                + " " * (cols - n_dots)
                + f"{str(i+1).zfill(n_rup_str)} / {n_rups}"
            )
            print(i_str, end="\r")

        # numba_list = List(rup_list)
        numba_list = np.array(rup_list, dtype=np.int64)
        # [numba_list.append(val) for val in rup_list]

        ruptures.append(numba_list)

    print("")
    print("calculating non-zero pairs")
    non_zero_pairs = get_non_zero_pairs(ruptures)

    # similarities = {}

    print(f"{len(non_zero_pairs)} non-zero pairs")

    print("calculating similarities")
    similarities = calculate_similarities(ruptures, non_zero_pairs)

    # print("converting to dict")
    similarities = {k: v for k, v in similarities.items()}

    return similarities


def filter_proportionally_to_plausibility(rup_df, plausibility, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rnds = np.random.rand(rup_df.shape[0])
    rup_df["plausibility"] = plausibility

    keep_df = rup_df[plausibility >= rnds]

    return keep_df
