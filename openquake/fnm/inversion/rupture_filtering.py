import pandas as pd
import numpy as np

import numba
from numba import jit, prange
from numba.typed import Dict, List
from numba.core import types


def connection_angle_plausibility(
    connection_angles, function_type="cosine", no_connection_val=1
):
    conns = np.array(connection_angles)
    conns[conns == no_connection_val] = 0.0

    if function_type == "cosine":
        # values between 1 and 0 for angles 0-180.
        plausibilities = np.cos(np.radians(conns / 2.0))

    else:
        raise NotImplementedError(
            f"Function type {function_type} not implemented."
        )

    total_plaus = np.prod(plausibilities)
    return total_plaus


def find_decay_exponent(y, x):
    """
    Finds the exponent (lambda) of the decay equation y = e^(-lambda x) given y and x.

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
    dist_threshold=15.0,
    prob_threshold=0.1,
    no_connection_val=-1,
):
    conns = np.array(connection_distances)
    conns[conns == no_connection_val] = 0.0

    if function_type == "exponent":
        decay_exp = find_decay_exponent(prob_threshold, dist_threshold)
        plausibilities = np.exp(-decay_exp * conns)

    elif function_type == "linear":
        plausibilities = 1 - (conns / dist_threshold)

    else:
        raise NotImplementedError(
            f"Function type {function_type} not implemented."
        )

    total_plaus = np.prod(plausibilities)
    return total_plaus


def slip_azimith_plausibility(
    slip_azimuths,
    function_type="cosine",
):
    if len(slip_azimuths) == 1:
        return 1.0

    slip_azimuths = np.array(sorted(slip_azimuths))
    slip_azimuth_diffs = np.diff(slip_azimuths)

    if function_type == "cosine":
        # allow
        plausibilities = np.abs(np.cos(np.radians(slip_azimuth_diffs)))

    else:
        raise NotImplementedError(
            f"Function type {function_type} not implemented."
        )

    total_prob = np.prod(plausibilities)
    return total_prob


def get_single_rupture_plausibilities(
    rupture,
    connection_angle_function="cosine",
    connection_distance_function="exponent",
    slip_azimuth_function="cosine",
    connection_angle_threshold=1.0,
    connection_distance_threshold=15.0,
    connection_distance_plausibility_threshold=0.1,
):
    plausibilities = {}

    plausibilities["connection_angle"] = connection_angle_plausibility(
        rupture["connection_angles"],
        function_type=connection_angle_function,
        no_connection_val=connection_angle_threshold,
    )

    plausibilities["connection_distance"] = connection_distance_plausibility(
        rupture["connection_distances"],
        function_type=connection_distance_function,
        dist_threshold=connection_distance_threshold,
        prob_threshold=connection_distance_plausibility_threshold,
    )

    plausibilities["slip_azimuth"] = slip_azimith_plausibility(
        rupture["slip_azimuths"], function_type=slip_azimuth_function
    )

    plausibilities["total"] = np.prod(list(plausibilities.values()))

    return plausibilities


def get_rupture_plausibilities(
    ruptures,
    connection_angle_function="cosine",
    connection_distance_function="exponent",
    slip_azimuth_function="cosine",
    connection_angle_threshold=1.0,
    connection_distance_threshold=15.0,
    connection_distance_plausibility_threshold=0.1,
):
    plausibilities = pd.DataFrame(
        index=ruptures.index,
        columns=[
            "connection_angle",
            "connection_distance",
            "slip_azimuth",
            "total",
        ],
    )

    for i, rupture in ruptures.iterrows():
        plausibilities.loc[i][
            "connection_angle"
        ] = connection_angle_plausibility(
            rupture["connection_angles"],
            function_type=connection_angle_function,
            no_connection_val=connection_angle_threshold,
        )

        plausibilities.loc[i][
            "connection_distance"
        ] = connection_distance_plausibility(
            rupture["connection_distances"],
            function_type=connection_distance_function,
            dist_threshold=connection_distance_threshold,
            prob_threshold=connection_distance_plausibility_threshold,
        )

        plausibilities.loc[i]["slip_azimuth"] = slip_azimith_plausibility(
            rupture["slip_azimuths"], function_type=slip_azimuth_function
        )

    plausibilities["total"] = (
        plausibilities.connection_angle
        * plausibilities.connection_distance
        * plausibilities.slip_azimuth
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
