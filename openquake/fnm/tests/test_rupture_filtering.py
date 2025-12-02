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

import unittest
import numpy as np
import pandas as pd

from scipy.sparse import dok_array
from openquake.hazardlib.geo import Line, Point

from openquake.fnm.rupture_connections import get_proximal_rup_angles
from openquake.fnm.rupture_filtering import (
    find_decay_exponent,
    get_rupture_plausibilities,
)


class TestRupturePlausibilities(unittest.TestCase):
    def test_connection_angle_plausibility_from_sparse_angles(self):
        distance_matrix = np.array([[0.0, 5.0], [5.0, 0.0]])
        binary = dok_array(distance_matrix.shape, dtype=np.int8)
        binary[0, 1] = 1
        binary[1, 0] = 1

        traces = [
            Line([Point(0.0, 0.0), Point(1.0, 0.0)]),
            Line([Point(0.0, 0.0), Point(0.0, 1.0)]),
        ]
        angles_dict = get_proximal_rup_angles(traces, binary)

        angle_matrix = dok_array(binary.shape, dtype=np.float64)
        for (i, j), (_, ang) in angles_dict.items():
            angle_matrix[i, j] = ang
            angle_matrix[j, i] = ang

        angle_val = list(angles_dict.values())[0][1]

        rupture_df = pd.DataFrame(
            {
                "ruptures": [[0], [1], [0, 1]],
                "slip_azimuth": [[0.0], [0.0], [0.0, 0.0]],
            }
        )

        plausibilities = get_rupture_plausibilities(
            rupture_df,
            distance_matrix=distance_matrix,
            angle_matrix=angle_matrix,
        )

        expected_angle_plaus = np.cos(np.radians(angle_val / 2.0))
        decay = find_decay_exponent(0.1, 15.0)
        expected_distance_plaus = float(np.exp(-decay * distance_matrix[0, 1]))

        self.assertIn("connection_angle", plausibilities.columns)
        np.testing.assert_allclose(
            plausibilities.loc[2, "connection_angle"], expected_angle_plaus
        )
        np.testing.assert_allclose(
            plausibilities.loc[2, "connection_distance"],
            expected_distance_plaus,
        )
        np.testing.assert_allclose(
            plausibilities.loc[2, "total"],
            expected_angle_plaus * expected_distance_plaus,
        )
