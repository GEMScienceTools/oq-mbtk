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
import pandas as pd

from scipy.sparse import dok_array

from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.kite_fault import (
    KiteSurface,
    get_profiles_from_simple_fault_data,
)
from openquake.aft.rupture_distances import RupDistType

from openquake.fnm.all_together_now import (
    build_fault_network,
)

from openquake.fnm.fault_modeler import (
    simple_fault_from_feature,
    get_subsections_from_fault,
    group_subfaults_by_fault,
    make_sf_rupture_meshes,
)
from openquake.fnm.rupture_connections import (
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
    make_binary_adjacency_matrix,
    get_proximal_rup_angles,
    filter_bin_adj_matrix_by_rupture_angle,
    get_multifault_ruptures,
    rdist_to_dist_matrix,
    get_mf_distances_from_adj_matrix,
    make_binary_adjacency_matrix_sparse,
    _latlon_to_xyz,
    _xyz_to_latlon,
    _geog_vec_to_xyz,
    intersection_pt,
    find_intersection_angle,
    angle_difference,
    get_dists_and_azimuths_from_pt,
    adjust_distances_based_on_azimuth,
    _check_overlap,
    calc_rupture_overlap,
    _is_strike_slip,
    filter_bin_adj_matrix_by_rupture_overlap,
    sparse_to_adj_dict,
    sparse_to_adjlist,
    find_connected_subsets_dfs,
    find_connected_components,
    find_connected_subgraphs,
    get_multifault_rupture_distances,
    subgraphs_from_connected_components,
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
            [[44.98, 10.0], [44.98, 10.254], [45.0, 10.254], [45.0, 10.0]]
        )

        np.testing.assert_almost_equal(expected, bb, decimal=3)

    def test_get_bounding_box_distances(self):
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
                (0, 1, 16.696556),
                (0, 2, 44.050705),
                (0, 3, 27.633398),
                (1, 0, 16.696556),
                (1, 1, 0.0),
                (1, 2, 11.687619),
                (1, 3, 9.183205),
                (2, 0, 44.050705),
                (2, 1, 11.687619),
                (2, 2, 0.0),
                (2, 3, 16.434488),
                (3, 0, 27.633398),
                (3, 1, 9.183205),
                (3, 2, 16.434488),
                (3, 3, 0.0),
            ],
            dtype=[('r1', '<i4'), ('r2', '<i4'), ('d', '<f4')],
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
                [14.82076928, -89.64507513],
                [14.82076928, -89.5385221],
                [14.90267166, -89.5385221],
                [14.90267166, -89.64507513],
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
        close_fault_dists_ = {
            (0, 1): 18.880685516364007,
            (1, 0): 18.880685516364007,
            (0, 2): 0.4039877204355394,
            (2, 0): 0.4039877204355394,
            (1, 2): 0.4347642767151649,
            (2, 1): 0.4347642767151649,
        }

        assert set(close_fault_dists.keys()) == set(close_fault_dists_.keys())
        for k, v in close_fault_dists.items():
            assert np.isclose(v, close_fault_dists_[k])

    def test_get_close_faults_with_max_dist(self):
        close_fault_dists = get_close_faults(self.faults, max_dist=3.0)
        close_fault_dists_ = {
            (0, 2): 0.4039877204355394,
            (2, 0): 0.4039877204355394,
            (1, 2): 0.4347642767151649,
            (2, 1): 0.4347642767151649,
        }

        assert set(close_fault_dists.keys()) == set(close_fault_dists_.keys())
        for k, v in close_fault_dists.items():
            assert np.isclose(v, close_fault_dists_[k])

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

        assert subfaults_are_adjacent(subs[1], subs[2]) is True
        assert subfaults_are_adjacent(subs[0], subs[2]) is False
        assert subfaults_are_adjacent(subs[1], subs[0]) is True
        assert subfaults_are_adjacent(subs[5], subs[1]) is True
        assert subfaults_are_adjacent(subs[4], subs[1]) is False

    def test_get_single_fault_rupture_coordinates_2_subfaults(self):
        subs = get_subsections_from_fault(
            self.faults[0],
            subsection_size=10.0,
            edge_sd=2.0,
            surface=self.faults[0]['surface'],
        )

        rup_coords = get_single_fault_rupture_coordinates([0, 1], subs)
        rup_coords_ = np.array(
            [
                [43.07803508, -6163.69348811, 1611.41796799],
                [43.35673269, -6164.21878892, 1609.39986208],
                [43.2671511, -6164.74962421, 1607.36773154],
                [43.00987219, -6165.26823062, 1605.38431393],
                [43.68919786, -6165.76544028, 1603.45526581],
                [41.82652961, -6161.96029058, 1610.71943867],
                [42.10514829, -6162.48541864, 1608.70190308],
                [42.01559133, -6163.01600501, 1606.67034686],
                [41.75838421, -6163.53433419, 1604.68748983],
                [42.4375183, -6164.03146311, 1602.75898695],
                [40.57574682, -6160.22681162, 1610.02097925],
                [40.85428656, -6160.75176691, 1608.00401409],
                [40.76475422, -6161.28210439, 1605.97303228],
                [40.50761892, -6161.8001564, 1603.99073591],
                [41.18656139, -6162.29720453, 1602.06277835],
                [39.32568676, -6158.49305144, 1609.32258981],
                [39.60414754, -6159.01783395, 1607.30619515],
                [39.51463983, -6159.54792258, 1605.27578784],
                [39.25757636, -6160.06569747, 1603.29405224],
                [39.93632718, -6160.56266474, 1601.36664008],
                [44.03258065, -6166.27443761, 1601.4873374],
                [43.63846065, -6166.79659692, 1599.48629783],
                [43.25907373, -6167.31866181, 1597.48244945],
                [43.2151794, -6167.83749969, 1595.47924641],
                [43.46313516, -6168.35670404, 1593.4640026],
                [42.78080396, -6164.54030729, 1600.79161479],
                [42.38679424, -6165.06216019, 1598.79114086],
                [42.00751347, -6165.58392156, 1596.78785892],
                [41.96363093, -6166.1025243, 1594.78522218],
                [42.21151638, -6166.62155158, 1592.77054809],
                [41.52974991, -6162.80589554, 1600.09596253],
                [41.13585049, -6163.3274421, 1598.09605432],
                [40.75667587, -6163.84890002, 1596.09333893],
                [40.71280513, -6164.36726765, 1594.09126857],
                [40.96062025, -6164.88611785, 1592.0771643],
                [40.27941854, -6161.07120256, 1599.40038068],
                [39.88562943, -6161.59244285, 1597.40103829],
                [39.506561, -6162.1135974, 1595.39888953],
                [39.46270205, -6162.63172994, 1593.39738565],
                [39.71044684, -6163.15040305, 1591.38385128],
            ]
        )
        np.testing.assert_array_almost_equal(rup_coords, rup_coords_)

    def test_get_single_fault_rupture_coordinates_1_subfault(self):
        subs = get_subsections_from_fault(
            self.faults[0],
            subsection_size=10.0,
            edge_sd=2.0,
            surface=self.faults[0]['surface'],
        )

        rup_coords = get_single_fault_rupture_coordinates([0], subs)
        rup_coords_ = np.array(
            [
                [43.07803508, -6163.69348811, 1611.41796799],
                [43.35673269, -6164.21878892, 1609.39986208],
                [43.2671511, -6164.74962421, 1607.36773154],
                [43.00987219, -6165.26823062, 1605.38431393],
                [43.68919786, -6165.76544028, 1603.45526581],
                [41.82652961, -6161.96029058, 1610.71943867],
                [42.10514829, -6162.48541864, 1608.70190308],
                [42.01559133, -6163.01600501, 1606.67034686],
                [41.75838421, -6163.53433419, 1604.68748983],
                [42.4375183, -6164.03146311, 1602.75898695],
                [40.57574682, -6160.22681162, 1610.02097925],
                [40.85428656, -6160.75176691, 1608.00401409],
                [40.76475422, -6161.28210439, 1605.97303228],
                [40.50761892, -6161.8001564, 1603.99073591],
                [41.18656139, -6162.29720453, 1602.06277835],
                [39.32568676, -6158.49305144, 1609.32258981],
                [39.60414754, -6159.01783395, 1607.30619515],
                [39.51463983, -6159.54792258, 1605.27578784],
                [39.25757636, -6160.06569747, 1603.29405224],
                [39.93632718, -6160.56266474, 1601.36664008],
            ]
        )
        np.testing.assert_array_almost_equal(rup_coords, rup_coords_)

    def test_get_single_fault_rups(self):
        subs = get_subsections_from_fault(
            self.faults[2],
            subsection_size=10.0,
            edge_sd=2.0,
            surface=self.faults[2]['surface'],
        )

        rup_df = get_single_fault_rups(subs, subfault_index_start=10)

        rup_df_ = pd.DataFrame(
            {
                'fault_rup': [0, 1, 2, 3, 4, 5, 6],
                'patches': [[0], [0, 1], [0, 1, 2, 3], [1], [2], [2, 3], [3]],
                'subfaults': [
                    [10],
                    [10, 11],
                    [10, 11, 12, 13],
                    [11],
                    [12],
                    [12, 13],
                    [13],
                ],
                'fault': [
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                ],
                'full_fault_rupture': [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
            },
            index=np.arange(7) + 10,
        )

        pd.testing.assert_frame_equal(rup_df, rup_df_)

    def test_get_all_single_fault_rups(self):
        all_subs = [
            get_subsections_from_fault(
                fault, subsection_size=[10.0, 10.0], surface=fault['surface']
            )
            for fault in self.faults
        ]

        single_fault_rups, rup_df = get_all_single_fault_rups(all_subs)

        single_fault_rups_ = [
            [
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
            ],
            [[0], [1]],
            [[0], [0, 1], [0, 1, 2, 3], [1], [2], [2, 3], [3]],
        ]

        rup_df_ = pd.DataFrame(
            {
                'fault_rup': [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    0,
                    1,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ],
                'patches': [
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
                    [0],
                    [1],
                    [0],
                    [0, 1],
                    [0, 1, 2, 3],
                    [1],
                    [2],
                    [2, 3],
                    [3],
                ],
                'subfaults': [
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
                    [8],
                    [9],
                    [10],
                    [10, 11],
                    [10, 11, 12, 13],
                    [11],
                    [12],
                    [12, 13],
                    [13],
                ],
                'fault': [
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf121',
                    'ccaf134',
                    'ccaf134',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                    'ccaf148',
                ],
                'full_fault_rupture': [
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
            },
            index=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
            ],
        )

        assert single_fault_rups == single_fault_rups_
        pd.testing.assert_frame_equal(rup_df, rup_df_)

    def test_get_rupture_adjacency_matrix_specific_values(self):
        all_subs = [
            get_subsections_from_fault(
                fault, subsection_size=[10.0, 10.0], surface=fault['surface']
            )
            for fault in self.faults
        ]

        rup_df, dist_adj_matrix = get_rupture_adjacency_matrix(
            self.faults,
            all_subfaults=all_subs,
            multifaults_on_same_fault=False,
            max_dist=20.0,
            full_fault_only_mf_ruptures=False,
        )

        close_faults = get_close_faults(self.faults, max_dist=20.0 * 1.5)
        # close_faults = {(i, j): d for i, j, d in close_faults}
        fault_lookup = {fault['fid']: i for i, fault in enumerate(self.faults)}

        sub_groups = group_subfaults_by_fault(all_subs)

        for i, rup_i in rup_df.iterrows():
            coords_i = get_single_fault_rupture_coordinates(
                rup_i['patches'], sub_groups[rup_i['fault']]
            )
            for j, rup_j in rup_df.iterrows():
                coords_j = get_single_fault_rupture_coordinates(
                    rup_j['patches'], sub_groups[rup_j['fault']]
                )

                if i == j:
                    assert dist_adj_matrix[i, j] == 0.0

                elif (
                    fault_lookup[rup_i['fault']],
                    fault_lookup[rup_j['fault']],
                ) not in close_faults:  # faults too far apart
                    assert dist_adj_matrix[i, j] == 0.0

                elif rup_i['fault'] == rup_j['fault']:
                    assert dist_adj_matrix[i, j] == 0.0

                elif i < j:
                    dists = np.linalg.norm(
                        coords_i[:, None, :] - coords_j[None, :, :], axis=-1
                    )

                    closest_dist = np.min(dists)

                    if closest_dist <= 20.0:
                        np.testing.assert_almost_equal(
                            closest_dist, dist_adj_matrix[i, j], decimal=3
                        )
                    else:
                        assert dist_adj_matrix[i, j] == 0.0
                    # assert dist_adj_matrix[i, j] == np.inf
            # subfaaaaaults

    def test_get_rupture_adjacency_matrix_default(self):
        all_subs = [
            get_subsections_from_fault(
                fault, subsection_size=[10.0, 10.0], surface=fault['surface']
            )
            for fault in self.faults
        ]

        rup_df, dist_adj_matrix = get_rupture_adjacency_matrix(
            self.faults,
            all_subfaults=all_subs,
            multifaults_on_same_fault=False,
            max_dist=20.0,
        )

        rup_df_ = pd.DataFrame(
            {
                # fmt: off
                'fault_rup': [
                    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    16,17,18,19,0,1,0,1,2,3,4,5,6,
                ],
                # fmt: on
                'patches': [
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
                    [0],
                    [1],
                    [0],
                    [0, 1],
                    [0, 1, 2, 3],
                    [1],
                    [2],
                    [2, 3],
                    [3],
                ],
                'subfaults': [
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
                    [8],
                    [9],
                    [10],
                    [10, 11],
                    [10, 11, 12, 13],
                    [11],
                    [12],
                    [12, 13],
                    [13],
                ],
                # fmt: off
                'fault': [
                    'ccaf121','ccaf121','ccaf121','ccaf121','ccaf121',
                    'ccaf121','ccaf121','ccaf121','ccaf121','ccaf121',
                    'ccaf121','ccaf121','ccaf121','ccaf121','ccaf121',
                    'ccaf121','ccaf121','ccaf121','ccaf121','ccaf121',
                    'ccaf134','ccaf134','ccaf148','ccaf148','ccaf148',
                    'ccaf148','ccaf148','ccaf148','ccaf148',
                ],
                'full_fault_rupture': [
                    False, False, False, False,  True, False, False, False, 
                    False, False, False, False, False, False, False, False, 
                    False, False, False, False, False, False, False, False,  
                    True, False, False, False, False
                ]
                # fmt: on
            },
            # fmt: off
            index=[
                0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                15,16,17,18,19,20,21,22,23,24,25,26,27,28,
            ],
            # fmt: on
        )
        # fmt: off
        dist_adj_matrix_ = dok_array(np.zeros((29, 29), dtype=np.float32))
        dist_adj_matrix_[(4, 24)] = 3.1237957
        dist_adj_matrix_[(24, 4)] = 3.1237957

        # fmt: on
        pd.testing.assert_frame_equal(rup_df, rup_df_)
        np.testing.assert_allclose(
            dist_adj_matrix.todense(),
            dist_adj_matrix_.todense(),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_make_binary_adjacency_matrix(self):
        # fmt: off
        dist_mat = np.array([
            [0.0, 0.0, 23.5, 0.0], 
            [1.0, 2.0, 3.0, 4.0]])

        bin_adj_mat = make_binary_adjacency_matrix(dist_mat, 10.0)

        bin_adj_mat_ = np.array([
            [0, 0, 0, 0], 
            [1, 1, 1, 1]], dtype=np.int32)
        # fmt: on

        np.testing.assert_array_equal(bin_adj_mat, bin_adj_mat_)

    @unittest.skip("Deprecated function")
    def test_get_proximal_rup_angles(self):
        all_subs = [
            get_subsections_from_fault(
                fault, subsection_size=[15.0, 15.0], surface=fault['surface']
            )
            for fault in self.faults
        ]

        single_rup_df, dist_adj_matrix = get_rupture_adjacency_matrix(
            self.faults,
            all_subfaults=all_subs,
            multifaults_on_same_fault=False,
            max_dist=20.0,
        )

        binary_adjacence_matrix = make_binary_adjacency_matrix(
            dist_adj_matrix, max_dist=10.0
        )

        sf_meshes = make_sf_rupture_meshes(
            single_rup_df['patches'], single_rup_df['fault'], all_subs
        )
        rup_angles = get_proximal_rup_angles(
            sf_meshes, binary_adjacence_matrix
        )

        # I don't know if these are being calculated correctly
        # but the test is for consistency with existing code
        rup_angles_ = {
            (0, 7): (70.61841415750447, 161.4449292745863),
            (1, 7): (70.00011153265001, 161.47178019533004),
            (2, 7): (73.25858241559682, 153.90944998942285),
            (6, 7): (77.33953084171088, 90.18657231023437),
        }

        for subs, angles in rup_angles.items():
            for i in [0, 1]:
                np.testing.assert_almost_equal(
                    rup_angles_[subs][i], angles[i], decimal=3
                )

    @unittest.skip("Deprecated function")
    def test_filter_bin_adj_matrix_by_rupture_angle(self):
        all_subs = [
            get_subsections_from_fault(
                fault, subsection_size=[15.0, 15.0], surface=fault['surface']
            )
            for fault in self.faults
        ]

        single_rup_df, dist_adj_matrix = get_rupture_adjacency_matrix(
            self.faults,
            all_subfaults=all_subs,
            multifaults_on_same_fault=False,
            max_dist=20.0,
        )

        # fmt: off
        binary_adjacency_matrix = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

        # fmt: on
        filtered_bin_adj_matrix = filter_bin_adj_matrix_by_rupture_angle(
            single_rup_df,
            all_subs,
            binary_adjacency_matrix,
            threshold_angle=160.0,
        )

        # fmt: off
        filtered_bin_adj_matrix_ = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        # fmt: on
        np.testing.assert_array_equal(
            filtered_bin_adj_matrix, filtered_bin_adj_matrix_
        )

    def test_rdist_to_dist_matrix_infer_shape(self):
        rdist = np.zeros(9, dtype=RupDistType)
        rdist['r1'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        rdist['r2'] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        rdist['d'] = np.array(
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

        dist_mat = rdist_to_dist_matrix(rdist)

        dist_mat_ = np.array(
            [
                [0.0, 0.68141705, 7.1189256],
                [0.68141705, 0.0, 5.3932385],
                [7.1189256, 5.3932385, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(dist_mat, dist_mat_)

    def test_rdist_to_dist_matrix_dont_infer_shape(self):
        rdist = np.zeros(9, dtype=RupDistType)
        rdist['r1'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        rdist['r2'] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        rdist['d'] = np.array(
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

        dist_mat = rdist_to_dist_matrix(rdist, nrows=3, ncols=4)

        dist_mat_ = np.array(
            [
                [0.0, 0.68141705, 7.1189256, 0.0],
                [0.68141705, 0.0, 5.3932385, 0.0],
                [7.1189256, 5.3932385, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(dist_mat, dist_mat_)

    def test_get_multifault_ruptures_1(self):
        # simple test with out long rups, because this was the dist matrix
        # that I had on hand
        dist_adj_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.357016, 3.1237752],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.357016, 3.1237752],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.357016, 3.1237752],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.93904],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.93904],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    19.357016,
                    19.357016,
                    19.357016,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.4397656,
                ],
                [
                    3.1237752,
                    3.1237752,
                    3.1237752,
                    15.93904,
                    15.93904,
                    0.0,
                    0.4397656,
                    0.0,
                ],
            ],
            dtype=np.float32,
        )

        multifault_ruptures = get_multifault_ruptures(
            dist_adj_matrix, max_dist=10.0
        )

        multifault_ruptures_ = [
            [0, 2, 6, 7],
            [0, 1, 7],
            [0, 2, 3, 6, 7],
            [2, 6],
            [1, 3, 6, 7],
            [1, 2, 6, 7],
            [0, 2, 4, 6, 7],
            [0, 4, 7],
            [1, 4, 7],
            [0, 6],
            [1, 7],
            [0, 2, 6],
            [1, 2, 7],
            [1, 3, 7],
            [1, 6, 7],
            [0, 1, 6, 7],
            [0, 7],
            [1, 6],
            [1, 2, 6],
            [0, 2, 7],
            [4, 7],
            [1, 2, 4, 6, 7],
            [3, 4, 7],
            [3, 7],
            [2, 6, 7],
            [2, 3, 7],
            [0, 3, 6, 7],
            [0, 1, 6],
            [2, 7],
            [2, 4, 7],
            [0, 1, 2, 6, 7],
            [0, 3, 7],
            [0, 1, 4, 6, 7],
            [0, 6, 7],
            [0, 1, 3, 6, 7],
            [1, 2, 3, 6, 7],
            [2, 3, 6, 7],
            [1, 4, 6, 7],
            [2, 4, 6, 7],
            [0, 4, 6, 7],
        ]

        assert multifault_ruptures == multifault_ruptures_

    def test_get_mf_distances_from_adj_matrix_1(self):
        dist_adj_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.357016, 3.1237752],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.357016, 3.1237752],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.357016, 3.1237752],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.93904],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.93904],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    19.357016,
                    19.357016,
                    19.357016,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.4397656,
                ],
                [
                    3.1237752,
                    3.1237752,
                    3.1237752,
                    15.93904,
                    15.93904,
                    0.0,
                    0.4397656,
                    0.0,
                ],
            ],
            dtype=np.float32,
        )

        mf_dist = get_mf_distances_from_adj_matrix([0, 7], dist_adj_matrix)

        np.testing.assert_array_almost_equal(mf_dist, np.array([3.1237752]))


class TestRuptureAngle(unittest.TestCase):

    def setUp(self):
        test_data_dir = HERE / 'data'
        fgj_name = os.path.join(
            test_data_dir, "rupture_angle_test_faults.geojson"
        )

        settings = {
            "max_jump_distance": 15.0,
        }

        self.fnn = build_fault_network(
            fault_geojson=fgj_name, settings=settings
        )

        self.ff = {
            f['fid']: Line([Point(*coords) for coords in f['trace']])
            for f in self.fnn['faults']
        }

    def test_find_intersection_angle(self):
        pt1, ang1 = find_intersection_angle(self.ff['0'], self.ff['1'])
        np.testing.assert_almost_equal(ang1, 23.43694989335944, decimal=3)
        np.testing.assert_almost_equal(pt1[0], -94.08663435482701, decimal=3)
        np.testing.assert_almost_equal(pt1[1], 36.16509951930527, decimal=3)

        pt_1_1, ang_1_1 = find_intersection_angle(self.ff['1'], self.ff['0'])
        assert pt1 == pt_1_1
        assert ang1 == ang_1_1

        pt2, ang2 = find_intersection_angle(self.ff['2'], self.ff['0'])
        np.testing.assert_almost_equal(ang2, 172.39985367835723, decimal=3)
        np.testing.assert_almost_equal(pt2[0], -94.23742852059331, decimal=3)
        np.testing.assert_almost_equal(pt2[1], 35.96270244712509, decimal=3)

        pt3, ang3 = find_intersection_angle(self.ff['3'], self.ff['4'])
        np.testing.assert_almost_equal(ang3, 172.08426757800999, decimal=3)
        np.testing.assert_almost_equal(pt3[0], -93.39321640386562, decimal=3)
        np.testing.assert_almost_equal(pt3[1], 36.40119252015622, decimal=3)

        pt4, ang4 = find_intersection_angle(self.ff['6'], self.ff['7'])
        np.testing.assert_almost_equal(ang4, 170.12816378838227, decimal=3)
        np.testing.assert_almost_equal(pt4[0], -92.95468876195562, decimal=3)
        np.testing.assert_almost_equal(pt4[1], 35.43799949715642, decimal=3)

        pt5, ang5 = find_intersection_angle(self.ff['5'], self.ff['7'])
        np.testing.assert_almost_equal(ang5, 168.31057889585173, decimal=3)
        np.testing.assert_almost_equal(pt5[0], -92.31044905834094, decimal=3)
        np.testing.assert_almost_equal(pt5[1], 35.542955886288354, decimal=3)

        pt6, ang6 = find_intersection_angle(self.ff['5'], self.ff['6'])
        np.testing.assert_almost_equal(ang6, 1.8673365279969403, decimal=3)
        np.testing.assert_almost_equal(pt6[0], -89.36589556390788, decimal=3)
        np.testing.assert_almost_equal(pt6[1], 36.566661543292504, decimal=3)

    def test_calc_rupture_overlap(self):
        ov1 = calc_rupture_overlap(self.ff['6'], self.ff['7'])
        np.testing.assert_almost_equal(ov1[1], 7.619135308500439, decimal=3)
        assert ov1[0] == 'partial'

        ov2 = calc_rupture_overlap(self.ff['6'], self.ff['5'])
        np.testing.assert_almost_equal(ov2[1], 52.09827524556749, decimal=3)
        assert ov2[0] == 'partial'

        ov3 = calc_rupture_overlap(self.ff['3'], self.ff['4'])
        assert ov3 == ('none', 0.0)

        ov4 = calc_rupture_overlap(self.ff['0'], self.ff['8'])
        assert ov4[0] == 'full'
        np.testing.assert_almost_equal(ov4[1], 38.54035180548081, decimal=3)

        ov5 = calc_rupture_overlap(self.ff['0'], self.ff['0'])
        assert ov5[0] == 'equality'
        np.testing.assert_almost_equal(ov5[1], 21.302702515507917, decimal=3)

    def test_filter_bin_adj_matrix_by_rupture_overlap(self):
        filtered_bin_adj_mat, rup_angles = (
            filter_bin_adj_matrix_by_rupture_overlap(
                self.fnn['single_rup_df'],
                self.fnn['subfaults'],
                self.fnn['bin_dist_mat'],
                strike_slip_only=True,
                threshold_overlap=20.0,
                threshold_angle=45.0,
            )
        )

        _rup_angles = {
            (3, 121): (
                (-94.23745270604606, 35.96267798197013),
                172.46587766701364,
            ),
            (15, 85): (
                (-92.95494494712885, 35.43752901940287),
                170.0622416474218,
            ),
            (26, 39): (
                (-93.38581598681843, 36.40534133795235),
                172.3923990528838,
            ),
            (39, 26): (
                (-93.38581598681843, 36.40534133795235),
                172.3923990528838,
            ),
            (85, 15): (
                (-92.95494494712885, 35.43752901940287),
                170.0622416474218,
            ),
            (121, 3): (
                (-94.23745270604606, 35.96267798197013),
                172.46587766701364,
            ),
        }
        for rup, angles in rup_angles.items():
            for i in [0, 1]:
                np.testing.assert_almost_equal(
                    _rup_angles[rup][i], angles[i], decimal=3
                )
        assert set(rup_angles.keys()) == set(_rup_angles.keys())

        filt_bin_mat_adj_dict = {k: v for k, v in filtered_bin_adj_mat.items()}
        _filt_bin_mat_adj_dict = {
            (3, 121): 1,
            (15, 85): 1,
            (26, 39): 1,
            (121, 3): 1,
            (85, 15): 1,
            (39, 26): 1,
        }

        assert filt_bin_mat_adj_dict == _filt_bin_mat_adj_dict
