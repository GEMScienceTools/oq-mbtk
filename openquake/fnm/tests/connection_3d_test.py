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

import pathlib
import unittest
import numpy as np

from openquake.fnm.plot import plot
from openquake.fnm.mesh import get_mesh_bb
from openquake.fnm.connections import get_connections
from openquake.fnm.fault_system import get_fault_system
from openquake.fnm.bbox import get_bb_distance_matrix

from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface, get_profiles_from_simple_fault_data)

PLOTTING = False
HERE = pathlib.Path(__file__).parent


def _get_surfs_3d():

    mesh_spacing = 2.0
    profile_sd = 1.0
    edge_sd = 1.0

    # Create the Kite Fault Surface - Almost vertical fault
    usd = 0.0
    lsd = 30.0
    dip = 80.0
    fault_trace = Line([Point(11.0, 45.0), Point(10.0, 45.0)])
    profiles = get_profiles_from_simple_fault_data(
        fault_trace, usd, lsd, dip, mesh_spacing)
    surf0 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    # Create the Kite Fault Surface - Dipping fault
    usd = 5.0
    lsd = 30.0
    dip = 30.0
    fault_trace = Line([Point(8.90, 45.2), Point(9.98, 45.1)])
    profiles = get_profiles_from_simple_fault_data(
        fault_trace, usd, lsd, dip, mesh_spacing)
    surf1 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    return [surf0, surf1]


class Test3DConnections(unittest.TestCase):

    def test_connection_by_distance(self):
        """Test connections by distance"""

        # Set the size of subsections. To create the subsections we set their
        # length and width in [km]
        subs_size = [10, 10]

        # Get the surfaces representing sections
        surfs = _get_surfs_3d()

        # Compute the bounding boxes
        bboxes = [get_mesh_bb(surf.mesh) for surf in surfs]

        # Computing the fault system
        fsys = get_fault_system(surfs, subs_size)

        # Get the bboxes distance matrix. The binary matrix `binm` is true when
        # the distance between the bounding boxes for two sections is shorter
        # than the threshold distance
        dmtx = get_bb_distance_matrix(bboxes)
        binm = np.zeros_like(dmtx)
        threshold = 20.0  # Threshold distance in km
        binm[dmtx < threshold] = 1

        # Get the connections
        criteria = {'min_distance_between_subsections':
                    {'threshold_distance': 20., 'shortest_only': False},
                    'only_connections_on_edge': True}

        # Get the connections
        conns, _, _ = get_connections(fsys, binm, criteria)

        # Expected connection
        expected = np.array([[0,  1,  0, 70,  8, 10,  0, 80,  5, 10]])

        # Test
        np.testing.assert_array_equal(conns, expected)

        if PLOTTING:
            meshes = [s.mesh for s in surfs]
            plot(meshes, connections=conns, fsys=fsys)
