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
import pathlib
import os
import json

from scipy.sparse import dok_array, issparse
from openquake.hazardlib.geo import Line, Point

from openquake.fnm.rupture_connections import get_proximal_rup_angles
from openquake.fnm.rupture_filtering import (
    get_rupture_plausibilities,
    compact_cosine_sigmoid,
    _compute_distance_plausibility,
    _compute_angle_plausibility,
    _compute_slip_az_plausibility,
    _build_angle_matrix_from_pairs,
)
from openquake.fnm.all_together_now import build_fault_network

HERE = pathlib.Path(__file__).parent


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

        expected_angle_plaus = compact_cosine_sigmoid(angle_val, 90.0)
        decay = np.log(2.0) / 15.0
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


class TestHelperComputations(unittest.TestCase):
    def setUp(self):
        self.rupture_df = pd.DataFrame(
            {
                "ruptures": [[0], [1], [0, 1]],
                "slip_azimuth": [[0.0], [10.0], [0.0, 10.0]],
            }
        )

    def test_compute_distance_plausibility(self):
        distances = pd.Series(
            [np.array([5.0]), np.array([5.0]), np.array([5.0])],
            index=self.rupture_df.index,
        )
        out = _compute_distance_plausibility(
            self.rupture_df,
            distances,
            connection_distance_function="exponent",
            connection_distance_midpoint=5.0,
            no_connection_val=-1.0,
        )
        expected = np.exp(-np.log(2.0) / 5.0 * 5.0)
        np.testing.assert_allclose(out, np.array([expected] * 3))

        out_disabled = _compute_distance_plausibility(
            self.rupture_df,
            distances,
            connection_distance_function="exponent",
            connection_distance_midpoint=None,
            no_connection_val=-1.0,
        )
        np.testing.assert_allclose(out_disabled, np.ones(3))

    def test_compute_angle_plausibility(self):
        angle_series = pd.Series(
            [np.array([10.0]), np.array([20.0]), np.array([30.0])],
            index=self.rupture_df.index,
        )
        out = _compute_angle_plausibility(
            self.rupture_df,
            angle_series,
            angle_matrix=None,
            bin_adj_mat=None,
            single_rup_df=None,
            subfaults=None,
            connection_angle_function="cosine",
            connection_angle_midpoint=90.0,
        )
        expected_vals = compact_cosine_sigmoid(angle_series.iloc[2], 90.0)
        self.assertAlmostEqual(out[2], np.prod(expected_vals))

        out_disabled = _compute_angle_plausibility(
            self.rupture_df,
            angle_series,
            angle_matrix=None,
            bin_adj_mat=None,
            single_rup_df=None,
            subfaults=None,
            connection_angle_function="cosine",
            connection_angle_midpoint=None,
        )
        np.testing.assert_allclose(out_disabled, np.ones(3))

    def test_build_angle_matrix_from_pairs(self):
        # two perpendicular traces should give ~90 degree angle
        subfaults = [
            [
                {
                    "fid": "f1",
                    "fault_position": (0, 0),
                    "trace": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                }
            ],
            [
                {
                    "fid": "f2",
                    "fault_position": (0, 0),
                    "trace": [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                }
            ],
        ]
        single_rup_df = pd.DataFrame(
            {"patches": [[0], [0]], "fault": ["f1", "f2"]}
        )
        bin_mat = dok_array((2, 2), dtype=np.int8)
        bin_mat[0, 1] = 1
        bin_mat[1, 0] = 1

        ang_mat = _build_angle_matrix_from_pairs(
            single_rup_df, subfaults, bin_mat
        )
        # angle lookup is symmetric
        angle_val = float(ang_mat[0, 1])
        self.assertAlmostEqual(angle_val, 90.0, delta=1.0)

    def test_compute_slip_plausibility(self):
        out = _compute_slip_az_plausibility(
            self.rupture_df,
            slip_azimuth_function="cosine",
            slip_azimuth_midpoint=90.0,
        )
        expected = compact_cosine_sigmoid(np.array([10.0]), 90.0)
        np.testing.assert_allclose(out.iloc[2], np.prod(expected))

        out_disabled = _compute_slip_az_plausibility(
            self.rupture_df,
            slip_azimuth_function="cosine",
            slip_azimuth_midpoint=None,
        )
        self.assertEqual(out_disabled, 1.0)


class TestFaultNetworkPlausibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_dir = HERE / "data"
        fgj_name = os.path.join(test_data_dir, "motagua_3_faults.geojson")
        settings = {
            "subsection_size": [10.0, 10.0],
            "max_jump_distance": 15.0,
            "rupture_filtering_connection_distance_midpoint": 15.0,
            "rupture_filtering_connection_angle_midpoint": 90.0,
            "rupture_filtering_slip_azimuth_midpoint": 90.0,
            "parallel_subfault_build": False,
            "parallel_multifault_search": False,
            "filter_by_plausibility": False,
            "filter_by_overlap": False,
            "filter_by_angle": False,
            "sparse_distance_matrix": True,
        }
        cls.settings = settings
        cls.fault_network = build_fault_network(
            fault_geojson=fgj_name, settings=settings
        )

    def test_angle_matrix_build(self):
        fn = self.fault_network
        bin_adj = fn["bin_dist_mat"]
        self.assertTrue(issparse(bin_adj))

        ang_mat = _build_angle_matrix_from_pairs(
            fn["single_rup_df"], fn["subfaults"], bin_adj
        )
        self.assertIsInstance(ang_mat, dok_array)
        self.assertGreater(len(ang_mat.keys()), 0)

    def test_plausibility_dataframe(self):
        fn = self.fault_network
        plaus_df = get_rupture_plausibilities(
            fn["rupture_df"],
            distance_matrix=fn["dist_mat"],
            bin_adj_mat=fn["bin_dist_mat"],
            single_rup_df=fn["single_rup_df"],
            subfaults=fn["subfaults"],
            connection_distance_midpoint=self.settings[
                "rupture_filtering_connection_distance_midpoint"
            ],
            connection_angle_midpoint=self.settings[
                "rupture_filtering_connection_angle_midpoint"
            ],
            slip_azimuth_midpoint=self.settings[
                "rupture_filtering_slip_azimuth_midpoint"
            ],
        )
        self.assertGreater(len(plaus_df), 0)
        for col in ["connection_angle", "connection_distance", "slip_azimuth"]:
            self.assertTrue(np.all(plaus_df[col] > 0))
        self.assertTrue(np.all(plaus_df["total"] > 0))
