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

import os
import json
import pathlib
import unittest
import numpy as np
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface,
    get_profiles_from_simple_fault_data,
)
from openquake.aft.rupture_distances import RupDistType

from openquake.fnm.once_more_with_feeling import (
    simple_fault_from_feature,
    get_subsections_from_fault,
)
from openquake.fnm.ships_in_the_night import (
    get_bb_from_surface,
    get_bounding_box_distances,
    get_close_faults,
    get_rupture_patches_from_single_fault,
    get_all_contiguous_subfaults,
    subfaults_are_adjacent,
    get_single_fault_rupture_coordinates,
    get_single_fault_rups,
    get_all_single_fault_rups,
    get_rupture_adjacency_matrix,
    make_binary_distance_matrix,
    get_proximal_rup_angles,
    filter_bin_adj_matrix_by_rupture_angle,
    get_multifault_ruptures,
    rdist_to_dist_matrix,
    get_mf_distances_from_adj_matrix,
)

HERE = pathlib.Path(__file__).parent.absolute()


class TestDistanceMatrix(unittest.TestCase):
    def setUp(self):
        mesh_spacing = 2.5
        profile_sd = 2.5
        edge_sd = 5.0

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 80.0
        fault_trace = Line([Point(10.0, 45.0), Point(10.3, 45.0)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf1 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 60.0
        fault_trace = Line([Point(9.9, 45.2), Point(10.2, 45.3)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf2 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 90.0
        fault_trace = Line([Point(10.3, 45.4), Point(10.2, 45.7)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf3 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

        # Create the Kite Fault Surface
        usd = 0
        lsd = 12.0
        dip = 90.0
        fault_trace = Line([Point(10.1, 45.25), Point(10.3, 45.25)])
        profiles = get_profiles_from_simple_fault_data(
            fault_trace, usd, lsd, dip, mesh_spacing
        )
        self.surf4 = KiteSurface.from_profiles(profiles, profile_sd, edge_sd)

    def test_get_bb_from_kite_surface(self):
        """ """
        bb = get_bb_from_surface(self.surf1)
        expected = np.array(
            [[45.0, 10.254], [45.0, 10.0], [44.98, 10.0], [44.98, 10.254]]
        )
        np.testing.assert_almost_equal(expected, bb, decimal=3)

    def test_get_dist_matrix(self):
        """ """
        dmtx = get_bounding_box_distances(
            [
                self.surf1,
                self.surf2,
                self.surf3,
                self.surf4,
            ]
        )

        # Testing the distance matrix - Using the service available at NOAA
        # https://www.nhc.noaa.gov/gccalc.shtml the distance between (45, 10)
        # and (45.2, 10) is 22 km. This corresponds (with some minor
        # differences) to the distance between bbox 0 and bbox 1. The distance
        # between bbox 1 and 3 is 0, as expected since they intersect.
        expected_old = np.array(
            [
                [0.0, 16.5839325, 44.4673135, 27.7674462],
                [0.0, 0.00000000, 11.5560295, 0.00000000],
                [0.0, 0.00000000, 0.00000000, 16.6747524],
                [0.0, 0.00000000, 0.00000000, 0.00000000],
            ]
        )

        # This is the new array based on the new distance calculation
        # using the aftershock distance calculation functions. The numbers
        # are quite similar to the old but the format is different.
        expected = np.array(
            [
                (0, 0, 0.0),
                (0, 1, 16.699274),
                (0, 2, 44.057846),
                (0, 3, 27.632025),
                (1, 0, 16.699274),
                (1, 1, 0.0),
                (1, 2, 11.687302),
                (1, 3, 9.169806),
                (2, 0, 44.057846),
                (2, 1, 11.687302),
                (2, 2, 0.0),
                (2, 3, 16.439398),
                (3, 0, 27.632025),
                (3, 1, 9.169806),
                (3, 2, 16.439398),
                (3, 3, 0.0),
            ],
            dtype=RupDistType,
        )

        for i, row in enumerate(dmtx):
            self.assertEqual(row['r1'], expected[i][0])
            self.assertEqual(row['r2'], expected[i][1])
            self.assertAlmostEqual(row['d'], expected[i][2], places=3)


class Test3Faults(unittest.TestCase):
    def setUp(self):
        test_data_dir = HERE / 'data'
        fgj_name = os.path.join(test_data_dir, "motagua_3_faults.geojson")

        with open(fgj_name) as f:
            fgj = json.load(f)
        self.features = fgj['features']

        self.faults = [
            simple_fault_from_feature(
                feature, edge_sd=2.0, lsd_default=20.0, usd_default=0.0
            )
            for feature in self.features
        ]

        self.fault_surfaces = [fault['surface'] for fault in self.faults]

    def test_get_bb_from_simple_fault_surface(self):
        bb = np.array(
            [
                [14.90267166, -89.5385221],
                [14.90267166, -89.64507513],
                [14.82076928, -89.64507513],
                [14.82076928, -89.5385221],
            ]
        )

        bb_from_fault = get_bb_from_surface(self.faults[1]['surface'])
        np.testing.assert_array_almost_equal(bb_from_fault, bb)

    def test_get_bounding_box_distances_no_max_dist(self):
        bb_dists = get_bounding_box_distances(
            self.fault_surfaces, max_dist=None
        )

        bb_dists_ = np.zeros(9, dtype=RupDistType)
        bb_dists_['r1'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        bb_dists_['r2'] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        bb_dists_['d'] = np.array(
            [
                0.0,
                0.68141705,
                7.1189256,
                0.68141705,
                0.0,
                5.3932385,
                7.1189256,
                5.3932385,
                0.0,
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(bb_dists['r1'], bb_dists_['r1'])
        np.testing.assert_array_almost_equal(bb_dists['r2'], bb_dists_['r2'])
        np.testing.assert_array_almost_equal(bb_dists['d'], bb_dists_['d'])

    def test_get_bounding_box_distances_with_max_dist(self):
        bb_dists = get_bounding_box_distances(
            self.fault_surfaces, max_dist=3.0
        )

        bb_dists_ = np.zeros(5, dtype=RupDistType)
        bb_dists_['r1'] = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        bb_dists_['r2'] = np.array([0, 1, 0, 1, 2], dtype=np.int32)
        bb_dists_['d'] = np.array(
            [0.0, 0.68141705, 0.68141705, 0.0, 0.0],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(bb_dists['r1'], bb_dists_['r1'])
        np.testing.assert_array_almost_equal(bb_dists['r2'], bb_dists_['r2'])
        np.testing.assert_array_almost_equal(bb_dists['d'], bb_dists_['d'])

    def test_get_close_faults_no_max_dist(self):
        close_fault_dists = get_close_faults(self.faults, max_dist=None)

        close_faults = np.zeros(9, dtype=RupDistType)
        close_faults['r1'] = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32
        )
        close_faults['r2'] = np.array(
            [0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32
        )
        close_faults['d'] = np.array(
            [
                0.0,
                0.68141705,
                7.1189256,
                0.68141705,
                0.0,
                5.3932385,
                7.1189256,
                5.3932385,
                0.0,
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(
            close_fault_dists['r1'], close_faults['r1']
        )
        np.testing.assert_array_almost_equal(
            close_fault_dists['r2'], close_faults['r2']
        )

    def test_get_close_faults_with_max_dist(self):
        close_fault_dists = get_close_faults(self.faults, max_dist=3.0)
        close_fault_dists_ = np.zeros(5, dtype=RupDistType)
        close_fault_dists_['r1'] = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        close_fault_dists_['r2'] = np.array([0, 1, 0, 1, 2], dtype=np.int32)
        close_fault_dists_['d'] = np.array(
            [0.0, 0.68141705, 0.68141705, 0.0, 0.0],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(
            close_fault_dists['r1'], close_fault_dists_['r1']
        )
        np.testing.assert_array_almost_equal(
            close_fault_dists['r2'], close_fault_dists_['r2']
        )
        np.testing.assert_array_almost_equal(
            close_fault_dists['d'], close_fault_dists_['d']
        )

    def test_get_rupture_patches_from_single_fault_1_row(self):
        rup_patches = get_rupture_patches_from_single_fault(
            get_subsections_from_fault(
                self.faults[0],
                subsection_size=[15.0, 15.0],
                edge_sd=2.0,
                surface=self.faults[0]['surface'],
            )
        )

        assert rup_patches == {
            'ccaf121': [[0], [0, 1], [0, 1, 2], [1], [1, 2], [2]]
        }

    def test_get_rupture_patches_from_single_fault_2_rows(self):
        rup_patches = get_rupture_patches_from_single_fault(
            get_subsections_from_fault(
                self.faults[0],
                subsection_size=10.0,
                edge_sd=2.0,
                surface=self.faults[0]['surface'],
            )
        )

        assert rup_patches == {
            'ccaf121': [
                [0],
                [0, 1],
                [0, 1, 4, 5],
                [0, 1, 2, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1],
                [1, 2],
                [1, 2, 5, 6],
                [1, 2, 3, 5, 6, 7],
                [2],
                [2, 3],
                [2, 3, 6, 7],
                [3],
                [4],
                [4, 5],
                [5],
                [5, 6],
                [6],
                [6, 7],
                [7],
            ]
        }

    def test_get_all_contiguous_subfaults_1_row(self):
        subs = get_all_contiguous_subfaults(3, 1, s_length=10.0, d_length=10.0)
        assert subs == [
            [(0, 0)],
            [(0, 0), (0, 1)],
            [(0, 0), (0, 1), (0, 2)],
            [(0, 1)],
            [(0, 1), (0, 2)],
            [(0, 2)],
        ]

    def test_get_all_contiguous_subfaults_1_col(self):
        subs = get_all_contiguous_subfaults(1, 3, s_length=10.0, d_length=10.0)
        assert subs == [[(0, 0)], [(1, 0)], [(2, 0)]]

    def test_get_all_contiguous_subfaults_rectangle_aspect_filter(self):
        subs = get_all_contiguous_subfaults(3, 3, min_aspect_ratio=1.0)
        assert subs == [
            [(0, 0)],
            [(0, 0), (0, 1)],
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ],
            [(0, 1)],
            [(0, 1), (0, 2)],
            [(0, 1), (0, 2), (1, 1), (1, 2)],
            [(0, 2)],
            [(1, 0)],
            [(1, 0), (1, 1)],
            [(1, 0), (1, 1), (1, 2)],
            [(1, 0), (1, 1), (2, 0), (2, 1)],
            [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            [(1, 1)],
            [(1, 1), (1, 2)],
            [(1, 1), (1, 2), (2, 1), (2, 2)],
            [(1, 2)],
            [(2, 0)],
            [(2, 0), (2, 1)],
            [(2, 0), (2, 1), (2, 2)],
            [(2, 1)],
            [(2, 1), (2, 2)],
            [(2, 2)],
        ]

    def test_get_all_contiguous_subfaults_rectangle_no_aspect_filter(self):
        subs = get_all_contiguous_subfaults(3, 3, min_aspect_ratio=0.1)
        assert subs == [
            [(0, 0)],
            [(0, 0), (0, 1)],
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ],
            [(0, 1)],
            [(0, 1), (0, 2)],
            [(0, 1), (1, 1)],
            [(0, 1), (0, 2), (1, 1), (1, 2)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
            [(0, 2)],
            [(0, 2), (1, 2)],
            [(0, 2), (1, 2), (2, 2)],
            [(1, 0)],
            [(1, 0), (1, 1)],
            [(1, 0), (1, 1), (1, 2)],
            [(1, 0), (2, 0)],
            [(1, 0), (1, 1), (2, 0), (2, 1)],
            [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            [(1, 1)],
            [(1, 1), (1, 2)],
            [(1, 1), (2, 1)],
            [(1, 1), (1, 2), (2, 1), (2, 2)],
            [(1, 2)],
            [(1, 2), (2, 2)],
            [(2, 0)],
            [(2, 0), (2, 1)],
            [(2, 0), (2, 1), (2, 2)],
            [(2, 1)],
            [(2, 1), (2, 2)],
            [(2, 2)],
        ]

    def test_subfaults_are_adjacent(self):
        subs = get_subsections_from_fault(
            self.faults[0],
            subsection_size=10.0,
            edge_sd=2.0,
            surface=self.faults[0]['surface'],
        )
