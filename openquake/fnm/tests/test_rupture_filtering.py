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

from openquake.fnm.rupture_connections import (
    get_multifault_connection_pairs,
    get_multifault_rupture_distances,
    get_proximal_rup_angles,
)
from openquake.fnm.rupture_filtering import (
    connection_angle_plausibility,
    get_rupture_plausibilities,
    get_single_rupture_plausibilities,
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


class TestPlausibilityRegressions(unittest.TestCase):
    def test_one_degree_connection_is_not_missing(self):
        expected = 0.5 * (1.0 + np.cos(np.deg2rad(1.0)))
        np.testing.assert_allclose(
            connection_angle_plausibility([1.0]), expected, rtol=1e-12
        )

    def test_distance_plausibility_counts_every_jump(self):
        # path 2 - 9 - 5 with two 8 km jumps; the rupture list order puts
        # the unconnected ruptures 2 and 5 next to each other
        dist = dok_array((10, 10), dtype=np.float32)
        for a, b in [(2, 9), (9, 5)]:
            dist[a, b] = dist[b, a] = 8.0
        rup_df = pd.DataFrame(
            {"ruptures": [[2, 5, 9], [2, 9]], "slip_azimuth": [[0.0] * 3, [0.0] * 2]}
        )
        plaus = get_rupture_plausibilities(
            rup_df,
            distance_matrix=dist,
            connection_distance_midpoint=5.0,
            connection_angle_midpoint=None,
            slip_azimuth_midpoint=None,
        )
        one_jump = np.exp(-np.log(2.0) / 5.0 * 8.0)
        np.testing.assert_allclose(
            plaus["connection_distance"].values, [one_jump**2, one_jump]
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
            "max_jump_distance": 30.0,
            "rupture_filtering_connection_distance_midpoint": 15.0,
            "rupture_filtering_connection_angle_midpoint": 90.0,
            "rupture_filtering_slip_azimuth_midpoint": 90.0,
            "parallel_subfault_build": False,
            "parallel_multifault_search": False,
            "filter_by_plausibility": False,
            "filter_by_overlap": False,
            "filter_by_angle": False,
            "sparse_distance_matrix": True,
            "always_return_full_rup": False,
        }
        cls.settings = settings
        cls.fault_network = build_fault_network(
            fault_geojson=fgj_name, settings=settings
        )

    def test_single_rupture_angle_plausibility_matches_batch(self):
        fn = self.fault_network
        pairs = get_multifault_connection_pairs(fn["rupture_df"], fn["dist_mat"])
        angle_matrix = _build_angle_matrix_from_pairs(
            fn["single_rup_df"], fn["subfaults"], fn["bin_dist_mat"]
        )
        batch = get_rupture_plausibilities(
            fn["rupture_df"],
            distance_matrix=fn["dist_mat"],
            angle_matrix=angle_matrix,
            connection_distance_midpoint=None,
            connection_angle_midpoint=90.0,
            slip_azimuth_midpoint=None,
        )
        # Exercise both real connections and single-fault ruptures, using
        # the module's -1 convention to represent no connection in the
        # single-rupture API. The batch API already treats that case as 1.
        self.assertTrue(any(len(p) == 0 for p in pairs))
        self.assertTrue(any(len(p) > 0 for p in pairs))
        for index, rup_pairs in pairs.items():
            with self.subTest(rupture=index):
                angles = [float(angle_matrix[i, j]) for i, j in rup_pairs]
                single = get_single_rupture_plausibilities(
                    {"connection_angles": angles if angles else [-1.0]},
                    connection_distance_midpoint=None,
                    connection_angle_midpoint=90.0,
                    slip_azimuth_midpoint=None,
                )
                np.testing.assert_allclose(
                    [single["connection_angle"], single["total"]],
                    batch.loc[index, ["connection_angle", "total"]].to_numpy(),
                    rtol=1e-12,
                )

    def test_missing_angle_does_not_penalize_real_connection(self):
        fn = self.fault_network
        angle_matrix = _build_angle_matrix_from_pairs(
            fn["single_rup_df"], fn["subfaults"], fn["bin_dist_mat"]
        )
        pairs = get_multifault_connection_pairs(fn["rupture_df"], fn["dist_mat"])
        connected_pairs = [p for p in pairs if p]
        self.assertEqual(len(connected_pairs), 1)
        angles = [float(angle_matrix[i, j]) for i, j in connected_pairs[0]]
        expected = np.prod(0.5 * (1.0 + np.cos(np.deg2rad(angles))))
        # A missing-angle marker must contribute a factor of exactly one,
        # while retaining the penalty for the GeoJSON-derived connection.
        single = get_single_rupture_plausibilities(
            {"connection_angles": angles + [-1.0]},
            connection_distance_midpoint=None,
            connection_angle_midpoint=90.0,
            slip_azimuth_midpoint=None,
        )
        np.testing.assert_allclose(
            [single["connection_angle"], single["total"]],
            [expected, expected],
            rtol=1e-12,
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
        _ang_mat = {(4, 20): 63.44241277654345, (20, 4): 63.44241277654345}
        assert ang_mat.keys() == _ang_mat.keys()
        for k, v in ang_mat.items():
            np.testing.assert_almost_equal(v, _ang_mat[k], decimal=2)

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

        _plaus_df = pd.DataFrame(
            {
                'connection_angle': {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                    13: 1.0,
                    14: 1.0,
                    15: 1.0,
                    16: 1.0,
                    17: 1.0,
                    18: 1.0,
                    19: 1.0,
                    20: 1.0,
                    21: 1.0,
                    22: 1.0,
                    23: 1.0,
                    24: 1.0,
                    25: 1.0,
                    26: 1.0,
                    27: 0.7235485370846245,
                },
                'connection_distance': {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                    13: 1.0,
                    14: 1.0,
                    15: 1.0,
                    16: 1.0,
                    17: 1.0,
                    18: 1.0,
                    19: 1.0,
                    20: 1.0,
                    21: 1.0,
                    22: 1.0,
                    23: 1.0,
                    24: 1.0,
                    25: 1.0,
                    26: 1.0,
                    27: 0.40738886535354024,
                },
                'slip_azimuth': {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                    13: 1.0,
                    14: 1.0,
                    15: 1.0,
                    16: 1.0,
                    17: 1.0,
                    18: 1.0,
                    19: 1.0,
                    20: 1.0,
                    21: 1.0,
                    22: 1.0,
                    23: 1.0,
                    24: 1.0,
                    25: 1.0,
                    26: 1.0,
                    27: 0.6961685583017807,
                },
                'total': {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                    13: 1.0,
                    14: 1.0,
                    15: 1.0,
                    16: 1.0,
                    17: 1.0,
                    18: 1.0,
                    19: 1.0,
                    20: 1.0,
                    21: 1.0,
                    22: 1.0,
                    23: 1.0,
                    24: 1.0,
                    25: 1.0,
                    26: 1.0,
                    27: 0.20520655500749668,
                },
            }
        )

        pd.testing.assert_frame_equal(
            plaus_df, _plaus_df, rtol=1e-2, check_dtype=True
        )


class TestFaultNetworkPlausibilityFullRup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_dir = HERE / "data"
        fgj_name = os.path.join(test_data_dir, "motagua_3_faults.geojson")
        settings = {
            "subsection_size": [10.0, 10.0],
            "max_jump_distance": 30.0,
            "rupture_filtering_connection_distance_midpoint": 15.0,
            "rupture_filtering_connection_angle_midpoint": 90.0,
            "rupture_filtering_slip_azimuth_midpoint": 90.0,
            "parallel_subfault_build": False,
            "parallel_multifault_search": False,
            "filter_by_plausibility": False,
            "filter_by_overlap": False,
            "filter_by_angle": False,
            "sparse_distance_matrix": True,
            "always_return_full_rup": True,
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
        # With always_return_full_rup=True, we have 32 ruptures instead of 24
        # So the angle matrix will have different keys
        # Just check that it's properly constructed
        for (i, j), angle in ang_mat.items():
            self.assertIsInstance(angle, (float, np.floating))
            self.assertGreaterEqual(angle, 0.0)
            self.assertLessEqual(angle, 180.0)

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
        # With always_return_full_rup=True, we expect 32 ruptures
        # (28 single-fault + 4 multifault)
        self.assertEqual(len(plaus_df), 32)
        for col in ["connection_angle", "connection_distance", "slip_azimuth"]:
            self.assertTrue(np.all(plaus_df[col] > 0))
        self.assertTrue(np.all(plaus_df["total"] > 0))

        # The three-fault rupture forms a cycle. Its shortest connections
        # are 4--23 (3.1238 km) and 23--20 (1.4302 km), rather than the
        # consecutive-list connection 4--20 (19.4328 km).
        three_fault = fn["rupture_df"][
            fn["rupture_df"].ruptures.apply(lambda r: set(r) == {4, 20, 23})
        ]
        self.assertEqual(len(three_fault), 1)
        np.testing.assert_allclose(
            plaus_df.loc[
                three_fault.index[0],
                ["connection_distance", "connection_angle", "total"],
            ].to_numpy(dtype=float),
            [0.8102295084, 0.0265688313, 0.0047834483],
            rtol=1e-4,
        )


class TestFaultNetworkConnectionPlausibility(unittest.TestCase):
    def test_generated_rupture_uses_actual_jumps(self):
        fn = build_fault_network(
            fault_geojson=HERE / "data" / "motagua_3_faults.geojson",
            settings={
                "subsection_size": [15.0, 15.0],
                "max_jump_distance": 10.0,
                "full_fault_only_mf_ruptures": False,
                "parallel_subfault_build": False,
                "parallel_multifault_search": False,
                "filter_by_overlap": False,
                "filter_by_plausibility": True,
                "rupture_filtering_connection_distance_midpoint": 5.0,
                "rupture_filtering_connection_angle_midpoint": 90.0,
                "rupture_filtering_slip_azimuth_midpoint": None,
            },
        )
        three_fault = fn["rupture_df"][
            fn["rupture_df"].ruptures.apply(lambda r: set(r) == {0, 6, 7})
        ]
        self.assertEqual(len(three_fault), 1)
        # Use the generated list unchanged: consecutive entries 0 and 6
        # are unconnected, while the actual path is 0--7--6.
        self.assertEqual(three_fault.iloc[0].ruptures, [0, 6, 7])
        self.assertEqual(fn["bin_dist_mat"][0, 6], 0)
        self.assertEqual(fn["dist_mat"][0, 6], 0.0)
        self.assertEqual(fn["bin_dist_mat"][0, 7], 1)
        self.assertEqual(fn["bin_dist_mat"][7, 6], 1)

        pairs = get_multifault_connection_pairs(three_fault, fn["dist_mat"])
        self.assertEqual(
            {frozenset(p) for p in pairs.iloc[0]},
            {frozenset({0, 7}), frozenset({7, 6})},
        )
        distances = get_multifault_rupture_distances(
            three_fault, fn["dist_mat"]
        )
        # Expected geometry values are fixed independently of connection
        # selection, so a skipped jump cannot also alter the expectation.
        expected_distances = np.array([3.1237750, 1.4301100])
        expected_angles = np.array([162.9023733, 89.9132993])
        np.testing.assert_allclose(
            np.sort(distances.iloc[0]), np.sort(expected_distances), rtol=1e-4
        )
        expected_distance = np.exp(-np.log(2.0) * expected_distances.sum() / 5.0)
        expected_angle = np.prod(
            0.5 * (1.0 + np.cos(np.deg2rad(expected_angles)))
        )
        np.testing.assert_allclose(
            fn["plausibility"].loc[
                three_fault.index[0],
                ["connection_distance", "connection_angle", "slip_azimuth", "total"],
            ].to_numpy(dtype=float),
            [
                expected_distance,
                expected_angle,
                1.0,
                expected_distance * expected_angle,
            ],
            rtol=1e-4,
        )
