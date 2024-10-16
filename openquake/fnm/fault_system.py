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


from typing import Tuple
import numpy as np
import igraph as ig
import numpy.typing as npt

from openquake.fnm.mesh import get_mesh_bb
from openquake.fnm.connections import get_connections
from openquake.fnm.bbox import get_bb_distance_matrix
from openquake.fnm.section import split_into_subsections
from openquake.fnm.rupture import (
    _check_rupture_has_connections,
    _get_ruptures_first_level,
    get_ruptures_area,
    get_mags_and_areas,
)


def get_fault_system(
    surfs: list, subs_size: Tuple[list, npt.ArrayLike]
) -> list:
    """
    Computes the fault system i.e. the geometry of each section, its
    subdivision into subsections and the size of subsections.

    :param surfs:
        A list of :class:`openquake.hazardlib.geo.surface.KiteFaultSurface`
    :param subs_size:
        A tuple with the initial size (in cells) of subsections
    :returns:
        A list where each element contains (1) the surface of the section and
        (2) an array where each row includes the indexes of the UL corner of a
        subsection and the number of cells along strike and dip (the shape of
        this array is <num_subs_along_dip> x <num_subs_along_strike> x 4)
    """
    fsys = []
    siss = split_into_subsections
    for i, surf in enumerate(surfs):
        try:
            sbs = siss(surf.mesh, subs_size[0], subs_size[1])
            fsys.append([surf, sbs])
        except ValueError:
            print(f"Error while splitting section {i}")
    return fsys


def get_connection_rupture_table(rups, conns: npt.ArrayLike) -> npt.ArrayLike:
    """
    Creates a table containing the primary ruptures (i.e. occurring just
    on one section) involved in a connection.

    :returns:
        A :class:`numpy.ndarray` instance with four columns and N rows. Each
        row contains the following information:
        - The index of the section
        - The index of the rupture in its section
        - The index of the connection in the fault system
        - The index of the rupture in the rupture array
    """
    data = []
    for i_rup, rup in enumerate(rups):
        # Search for connections involving this rupture. `found_connections`
        # contains:
        # - A boolean indicating if the subsection contains a given connection
        # - The index of the other section
        # - A boolean. When True the other component of the connection is the
        #   first one provided (otherwise it's the second one)
        # - The connection index (incremental). Can be used to select
        #   connections from the initial `connection` array
        found_connections = _check_rupture_has_connections(conns, rup)

        for conn in found_connections:
            if not conn[0]:
                continue
            tmp = [rup[6], rup[7], conn[3], i_rup]
            if tmp in data:
                continue
            # Each row of the `data` list contains three indexes:
            # - The index of the section
            # - The index of the rupture in its section
            # - The index of the connection in the fault system
            data.append(tmp)
        # This can be used for testing purposes
        # if np.sum(found_connections[:, 0]):
        #    print(rup)
    return np.array(data)


def get_multi_fault_adjacency_mtx(
    fault_system: list, connections: npt.ArrayLike, aratios: npt.ArrayLike
) -> Tuple[npt.ArrayLike, npt.ArrayLike, list, npt.ArrayLike]:
    """
    :param fault_system:
        The fault system
    :param connections:
        The :class:`numpy.ndarray` connection table
    :param aratios:
        The :class:`numpy.ndarray` instance with aspec ratio table. This is
        used to create the single-section ruptures.
    :returns:
        A :class:`numpy.ndarray` instance with size N x N where N is the number
        of single-section ruptures connected with other single-section
        rutpures, the rupture-connection matrix, the list of single-section
        ruptures and a :class:`numpy.ndarray` instance of the same size of
        `adjmtx` with the indexes of connections
    """

    # Get first level ruptures i.e. ruptures on individual sections and update
    # the archive with the list of rupture IDs
    rups = _get_ruptures_first_level(fault_system, aratios)

    # Set arrays type
    rups = rups.astype(int)
    connections = connections.astype(int)

    # Create the rupture-connection table. In the output array, the first
    # column contains the index of the section, the second the rupture index,
    # the third one the index of the connection and the fourth one the index
    # of the rupture. Note that we consider here only single-section ruptures.
    rupcon = get_connection_rupture_table(rups, connections)

    # The number of unique elements in the 4th column of `rupcon` is the number
    # of single-section (SS) ruptures
    rupidx = np.unique(rupcon[:, 3])
    num_connected_rups = len(rupidx)
    adjmtx = np.zeros((num_connected_rups, num_connected_rups))
    conmtx = np.ones((num_connected_rups, num_connected_rups)) * -1
    adjmtx = adjmtx.astype(int)
    conmtx = conmtx.astype(int)

    for i_rup_1 in range(len(rupcon[:, 3])):
        i1 = np.where(rupidx == rupcon[i_rup_1, 3])

        for i_rup_2 in range(i_rup_1 + 1, len(rupcon[:, 3])):
            i2 = np.where(rupidx == rupcon[i_rup_2, 3])

            # If the ruptures belong to the same section, continue
            if rupcon[i_rup_1, 0:1] == rupcon[i_rup_2, 0:1]:
                continue

            # If the ruptures do not share the same connection, continue
            if rupcon[i_rup_1, 2] != rupcon[i_rup_2, 2]:
                continue

            # Set the value of the connection for the combination of
            # single-section ruptures
            adjmtx[i1, i2] = 1
            adjmtx[i2, i1] = 1
            conmtx[i1, i2] = rupcon[i_rup_1, 2]
            conmtx[i2, i1] = rupcon[i_rup_1, 2]

    return adjmtx, rupcon, rups, conmtx


def get_rups_fsys(surfs: list, settings: dict):
    """
    Computes all the ruptures admitted by the fault system given the parameters
    included in the settings.

    :param surfs:
        The surfaces of the sections
    :param settings:
        A dictionary containing all the settings and plausibility criteria
    :returns:
        1. all_rups: A list of lists Each element contains a set of integers
            that is the indexes of the single-section ruptures forming complex
            ruptures
        2. mags: A :class:`numpy.ndarray` instance with the values of magnitude
            for each of the ruptures
        3. single_sec_rups: A :class:`numpy.ndarray` instance with the
            description of the section ruptures
        4. fault_sys: See :method:`openquake.fnm.fault_system.get_fault_system`
        5. all_areas: A :class:`numpy.ndarray` instance with the areas of the
            ruptures
        5. frac_areas: A list of list where each element contains fraction
            of the total area for each of the single-section ruptures forming
            a rupture
        6. rups_sect_idxs: A list of list where each element contains the
            indexes of the sections containing the single-section ruptures
            forming a rupture
    """

    # Settings and plausibility criteria
    criteria = settings["connections"]
    aratios = np.array(settings["ruptures"]["aspect_ratios"])
    subs_size = np.array(settings["general"]["subsection_size"])

    # Get fault system and sections' connection
    print("Getting fault system components")
    flt_sys, conns, dists, angls = _get_components(surfs, subs_size, criteria)
    flt_sys = np.array(flt_sys, dtype=object)

    print("Making adjacency matrix")
    # Adjacency matrix, ruptures connection matrix and ruptures at first level.
    # The `rupcon` array contains four columns with the index of the section,
    # the index of the rupture in this section, the index of the connection,
    # and index of the rupture
    adjm, rupcon, single_sec_rups, conm = get_multi_fault_adjacency_mtx(
        flt_sys, conns, aratios
    )

    # Get single-section rupture areas
    print("Getting single-section areas")
    msr_key = settings["ruptures"]["aspect_ratios"]
    areas = get_ruptures_area(surfs, single_sec_rups)
    print(len(areas), "areas")

    print("Preparing input for simple path calculation")
    # Get upper triangular mtx of adjacency and create the graph instance
    tru = np.triu(adjm)
    g = ig.Graph.Adjacency(tru)
    multi_section_rup_ids = np.unique(rupcon[:, 3])
    g.vs["id"] = multi_section_rup_ids

    """ for documentation purpouses
    layout = g.layout("kk")
    g.vs["label"] = g.vs["id"]
    ig.plot(g, "graph.pdf", layout=layout)
    """

    print("Getting ruptures as simple paths")
    all_rups = []
    all_cons = []
    all_rups.extend([[int(i)] for i in single_sec_rups[:, 4]])
    all_cons.extend([[] for i in single_sec_rups[:, 4]])
    n_ms_rups = len(multi_section_rup_ids)

    for i_rup in range(n_ms_rups):
        if i_rup == n_ms_rups - 1:
            end = "\n"
        else:
            end = "\r"
        try:
            msg = f"rupture {str(i_rup).zfill(len(str(n_ms_rups)))}"
            msg += f"{n_ms_rups}"
            print(msg, end="\r", flush=True)
            rupsm = g.get_all_simple_paths(
                i_rup, to=None, cutoff=-1, mode="out"
            )

            # Updating the list with the indexes of the connections for
            # each rupture
            new_cons = []
            for rup_idxs in rupsm:
                if len(rup_idxs) == 1:
                    new_cons.append([])
                tmp = []
                for irup1 in rup_idxs:
                    for irup2 in rup_idxs:
                        if conm[irup1, irup2] > -1:
                            tmp.append(conm[irup1, irup2])
                new_cons.append(np.unique(tmp))

            # Remapping indexes of multi fault ruptures
            new_rups = _remap_indexes(rupsm, multi_section_rup_ids)

            # Checking
            assert len(all_cons) == len(all_rups)
            assert len(new_rups) == len(new_cons)
            all_rups.extend(new_rups)
            all_cons.extend(new_cons)

        except ValueError:
            print(f"Error while getting rupture {i_rup}")
            print(" " * 80)

    # Compute the magnitude for all the ruptures
    print("Getting rupture magnitudes")
    msr_key = settings["ruptures"]["magnitude_scaling_rel"]
    mags, all_areas = get_mags_and_areas(all_rups, areas, msr_key)
    print(len(mags), "mags")

    # Get fraction of rupture on each subsection
    frac_areas = _get_area_fraction(all_rups, areas)

    # Get indexes of sections composing each rupture
    rups_sect_idxs = _get_section_indexes_per_rupt(single_sec_rups, all_rups)

    #    return {
    #        "rupture_sub_sections": all_rups,
    #        "mags": mags,
    #        "single_sec_rups": single_sec_rups,
    #        "fault_sys": fault_sys,
    #        "areas": all_areas,
    #        "frac_areas": frac_areas,
    #        "rup_sec_idxs": rups_sect_idxs,
    #    }

    # Find the distances and angles between the connections of multi-fault
    # ruptures. `rupcon` contains the single-section ruptures that are also
    # part of multi-fault ruptures
    print("Getting distances and angles between sections in m-fault rups")
    rdists, rangls = _get_dists_angls_multifault(all_cons, conns, dists, angls)

    results = {
        "ruptures_single_section_indexes": all_rups,
        "magnitudes": mags,
        "areas": all_areas,
        "ruptures_single_section": single_sec_rups,
        "fault_system": flt_sys,
        "rupture_fractional_area": frac_areas,
        "ruptures_indexes_of_sections_involved": rups_sect_idxs,
        "ruptures_connection_distances": rdists,
        "ruptures_connection_angles": rangls,
    }

    return results


def _get_dists_angls_multifault(
    all_cons, conns, dists, angls
) -> Tuple[list, list]:
    """
    :param all_cons:
    :param conns:
    :param dists:
    :param angls:
    """
    out_angls = []
    out_dists = []
    for conns in all_cons:
        if len(conns) < 1:
            out_angls.append([-1])
            out_dists.append([-1])
            continue
        else:
            tmp_dists = []
            tmp_angls = []
            for idx in conns:
                tmp_dists.append(dists[idx])
                tmp_angls.append(angls[idx])
        # Update the list
        out_angls.append(tmp_angls)
        out_dists.append(tmp_dists)
        assert len(out_angls[-1]) == len(conns)
    return out_dists, out_angls


def _get_section_indexes_per_rupt(rups1: npt.ArrayLike, rupsa: list) -> list:
    """
    :param rups1:
    :param rupsa:
    """
    out = []
    for rup in rupsa:
        out.append([rups1[i, 6] for i in rup])
    return out


def _get_area_fraction(all_rups: list, areas: npt.ArrayLike) -> list:
    """
    Computes the fraction of area on each section involved in a rupture

    :param all_rups:
        A list of lists with the indexes of the single-section
    :param areas:
        A numpy array with the areas of all the single-section ruptures
    :returns:
        A list of lists were each element defines the fraction of the total
        area covered by a single-section rupture
    """
    fractions = []
    for i_rup, rup in enumerate(all_rups):
        tmp = [areas[idx] for idx in rup]

        # Rounding
        tmp /= np.sum(tmp)
        tmp = [float(f"{f:.3f}") for f in tmp]
        last = 1.0 - np.sum(tmp[:-1])
        tmp[-1] = float(f"{last:.3f}")

        # Checking
        assert np.abs(1.0 - np.sum(tmp)) < 1e-5

        # Updating output
        fractions.append(tmp)

    return fractions


def _remap_indexes(rups, idxs):
    """ """
    out = []
    for lst in rups:
        out.append([idxs[i] for i in lst])
    return out


def _get_components(surfs, subs_size, criteria):
    # Get the threshold distance. This is used for finding the bounding boxes
    # that might be connected
    key = "min_distance_between_subsections"
    sub_key = "threshold_distance"
    if (key in criteria) and (sub_key in criteria[key]):
        threshold = criteria["min_distance_between_subsections"][sub_key]
    else:
        msg = "Please add a threshold distance to the criteria:\n"
        msg += "criteria['min_distance_between_subsections'][sub_key] = 1"
        raise ValueError(msg)

    # Get the fault system i.e. the description of the surfaces, their
    # subdivision into subsections and the shape of each subsection
    bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]
    fsys = get_fault_system(surfs, subs_size)

    # Get the bboxes distance matrix. The binary matrix `binm` is true when
    # the distance between the bounding boxes for two sections is shorter
    # than the threshold distance
    dmtx = get_bb_distance_matrix(bboxes)
    binm = np.zeros_like(dmtx)
    binm[dmtx < threshold] = 1

    # Get the connections
    conns, dists, angls = get_connections(fsys, binm, criteria)

    return fsys, conns, dists, angls
