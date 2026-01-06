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
    get_multifault_ruptures_fast,
    rdist_to_dist_matrix,
    get_mf_distances_from_adj_matrix,
    make_binary_adjacency_matrix_sparse,
    _lonlat_to_xyz,
    _xyz_to_lonlat,
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
    get_rupture_grouping,
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
        assert subs == [[(0, 0)], [(1, 0)], [(2, 0)], [(0, 0), (1, 0), (2, 0)]]

    def test_get_all_contiguous_subfaults_always_has_full_grid(self):
        subs = get_all_contiguous_subfaults(2, 5, min_aspect_ratio=1.5)
        full_grid = [(r, c) for r in range(5) for c in range(2)]
        assert full_grid in subs

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
                [37.90312048, -6158.25665837, 1602.88541233],
                [38.04250859, -6160.69070464, 1593.5012304],
                [38.07634948, -6156.75901024, 1608.6242704],
                [38.30784961, -6157.76128678, 1604.77780568],
                [38.35733478, -6157.25706696, 1606.7101444],
                [38.38238677, -6160.19549917, 1595.40639535],
                [38.44819196, -6161.17789487, 1591.60674987],
                [38.47522112, -6161.67703934, 1589.67263363],
                [38.56901147, -6158.72696385, 1601.06151599],
                [38.75324327, -6159.70091151, 1597.30592836],
                [39.12857749, -6159.20590492, 1599.20447008],
                [39.15241056, -6159.99138688, 1603.58210723],
                [39.29184055, -6162.42646769, 1594.19526843],
                [39.32568676, -6158.49305144, 1609.32258981],
                [39.55725317, -6159.49571832, 1605.47503628],
                [39.60675174, -6158.99126886, 1607.40792199],
                [39.63181396, -6161.93097789, 1596.10097283],
                [39.69763875, -6162.91378968, 1592.30025146],
                [39.72467606, -6163.41314698, 1590.36558755],
                [39.81848964, -6160.4617642, 1601.75769454],
                [40.00277439, -6161.43610004, 1598.00104366],
                [40.3782138, -6160.94080226, 1599.90012289],
                [40.40242343, -6161.72583447, 1604.27887242],
                [40.54189529, -6164.16194992, 1594.88937718],
                [40.57574682, -6160.22681162, 1610.02097925],
                [40.8073795, -6161.22986885, 1606.1723371],
                [40.85689146, -6160.72518973, 1608.10576971],
                [40.8819639, -6163.6661757, 1596.79562093],
                [40.9478083, -6164.64940363, 1592.99382386],
                [40.97485376, -6165.14897378, 1591.05861235],
                [41.06869057, -6162.19628357, 1602.45394348],
                [41.25302826, -6163.1710076, 1598.69622949],
                [41.62857284, -6162.67541857, 1600.59584615],
                [41.65315904, -6163.46000092, 1604.97570784],
                [41.79267276, -6165.89715109, 1595.58355658],
                [41.82652961, -6161.96029058, 1610.71943867],
                [42.05822855, -6162.96373818, 1606.86970805],
                [42.1077539, -6162.45882936, 1608.80368749],
                [42.13283657, -6165.40109242, 1597.49033959],
                [42.19870056, -6166.38473651, 1593.68746699],
                [42.22575418, -6166.88451953, 1591.75170797],
                [42.3196142, -6163.93052174, 1603.15026272],
                [42.50400483, -6164.905634, 1599.39148581],
                [42.87965457, -6164.40975365, 1601.29163982],
                [42.90461733, -6165.19388603, 1605.67261342],
                [43.04417291, -6167.63207102, 1596.27780656],
                [43.07803508, -6163.69348811, 1611.41796799],
                [43.30980026, -6164.6973261, 1607.56714909],
                [43.359339, -6164.19218755, 1609.50167525],
                [43.38443189, -6167.13572782, 1598.18512875],
                [43.45031548, -6168.11978811, 1594.38118079],
                [43.47737726, -6168.61978402, 1592.44487435],
                [43.57126048, -6165.6644785, 1603.8466522],
                [43.75570405, -6166.63997901, 1600.08681254],
                [44.13145892, -6166.14380727, 1601.98750381],
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
                [43.359339, -6164.19218755, 1609.50167525],
                [43.30980026, -6164.6973261, 1607.56714909],
                [42.90461733, -6165.19388603, 1605.67261342],
                [43.57126048, -6165.6644785, 1603.8466522],
                [44.13145892, -6166.14380727, 1601.98750381],
                [41.82652961, -6161.96029058, 1610.71943867],
                [42.1077539, -6162.45882936, 1608.80368749],
                [42.05822855, -6162.96373818, 1606.86970805],
                [41.65315904, -6163.46000092, 1604.97570784],
                [42.3196142, -6163.93052174, 1603.15026272],
                [42.87965457, -6164.40975365, 1601.29163982],
                [40.57574682, -6160.22681162, 1610.02097925],
                [40.85689146, -6160.72518973, 1608.10576971],
                [40.8073795, -6161.22986885, 1606.1723371],
                [40.40242343, -6161.72583447, 1604.27887242],
                [41.06869057, -6162.19628357, 1602.45394348],
                [41.62857284, -6162.67541857, 1600.59584615],
                [39.32568676, -6158.49305144, 1609.32258981],
                [39.60675174, -6158.99126886, 1607.40792199],
                [39.55725317, -6159.49571832, 1605.47503628],
                [39.15241056, -6159.99138688, 1603.58210723],
                [39.81848964, -6160.4617642, 1601.75769454],
                [40.3782138, -6160.94080226, 1599.90012289],
                [38.07634948, -6156.75901024, 1608.6242704],
                [38.35733478, -6157.25706696, 1606.7101444],
                [38.30784961, -6157.76128678, 1604.77780568],
                [37.90312048, -6158.25665837, 1602.88541233],
                [38.56901147, -6158.72696385, 1601.06151599],
                [39.12857749, -6159.20590492, 1599.20447008],
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
                'fault_group': [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
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
                ],
                'fault_group': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                2, 2, 2, 2, 2, 2, 2]
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

    def test_make_binary_adjacency_matrix_sparse(self):
        dist_mat = dok_array(
            np.array([[0.0, 0.0, 23.5, 0.0], [1.0, 2.0, 3.0, 4.0]])
        )

        bin_adj_mat = make_binary_adjacency_matrix_sparse(dist_mat, 10.0)

        bin_adj_mat_ = {
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }

        assert dict(bin_adj_mat) == bin_adj_mat_

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
        all_subs = [
            get_subsections_from_fault(
                fault, subsection_size=[15, 15], surface=fault['surface']
            )
            for fault in self.faults
        ]

        single_rup_df, dist_mat = get_rupture_adjacency_matrix(
            self.faults,
            all_subfaults=all_subs,
            max_dist=10.0,
            full_fault_only_mf_ruptures=False,
        )

        bm = make_binary_adjacency_matrix_sparse(dist_mat, max_dist=10.0)
        rup_groups = get_rupture_grouping(self.faults, single_rup_df)

        mf_rups = get_multifault_ruptures_fast(
            bm,
            rup_groups=rup_groups,
            max_sf_rups_per_mf_rup=7,
        )

        multifault_ruptures = [
            [1, 7],
            [2, 6, 7],
            [0, 6, 7],
            [6, 7],
            [1, 6, 7],
            [0, 7],
            [2, 7],
        ]

        assert mf_rups == multifault_ruptures

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

    def test_filter_bin_adj_matrix_by_rupture_overlap_strike_slip(self):
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
            (3, 111): (
                (-94.09127222911088, 36.15837942558471),
                23.72541541759415,
            ),
            (3, 123): (
                (-94.23744951464562, 35.96268215874638),
                172.46575156350337,
            ),
            (16, 86): (
                (-92.9550071372123, 35.43749999300131),
                170.06010702131718,
            ),
            (27, 40): (
                (-93.3857279349451, 36.40540484288613),
                172.39265427405473,
            ),
            (111, 123): (  # not ss
                (-93.98590224542299, 36.2177880383831),
                16.229116005034143,
            ),
            (111, 136): (  # not ss
                (-95.47421682875286, 35.361624710523586),
                10.688976730137199,
            ),
            (111, 3): (  # not both ss
                (-94.09127222911088, 36.15837942558471),
                23.72541541759415,
            ),
            (123, 3): (
                (-94.23744951464562, 35.96268215874638),
                172.46575156350337,
            ),
            (86, 16): (
                (-92.9550071372123, 35.43749999300131),
                170.06010702131718,
            ),
            (40, 27): (
                (-93.3857279349451, 36.40540484288613),
                172.39265427405473,
            ),
            (123, 111): (
                (-93.98590224542299, 36.2177880383831),
                16.229116005034143,
            ),
            (136, 111): (
                (-95.47421682875286, 35.361624710523586),
                10.688976730137199,
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
            (3, 111): 1,
            (3, 123): 1,
            (16, 86): 1,
            (27, 40): 1,
            (111, 123): 1,
            (111, 136): 1,
            (111, 3): 1,
            (123, 3): 1,
            (86, 16): 1,
            (40, 27): 1,
            (123, 111): 1,
            (136, 111): 1,
        }

        assert filt_bin_mat_adj_dict == _filt_bin_mat_adj_dict

    def test_filter_bin_adj_matrix_by_rupture_overlap_all(self):
        filtered_bin_adj_mat, rup_angles = (
            filter_bin_adj_matrix_by_rupture_overlap(
                self.fnn['single_rup_df'],
                self.fnn['subfaults'],
                self.fnn['bin_dist_mat'],
                strike_slip_only=False,
                threshold_overlap=20.0,
                threshold_angle=45.0,
            )
        )

        _rup_angles = {
            (3, 111): (
                (-94.09127222911088, 36.15837942558471),
                23.72541541759415,
            ),
            (3, 123): (
                (-94.23744951464562, 35.96268215874638),
                172.46575156350337,
            ),
            (16, 86): (
                (-92.9550071372123, 35.43749999300131),
                170.06010702131718,
            ),
            (27, 40): (
                (-93.3857279349451, 36.40540484288613),
                172.39265427405473,
            ),
            (111, 123): (
                (-93.98590224542299, 36.2177880383831),
                16.229116005034143,
            ),
            (111, 136): (
                (-95.47421682875286, 35.361624710523586),
                10.688976730137199,
            ),
            (111, 3): (
                (-94.09127222911088, 36.15837942558471),
                23.72541541759415,
            ),
            (123, 3): (
                (-94.23744951464562, 35.96268215874638),
                172.46575156350337,
            ),
            (86, 16): (
                (-92.9550071372123, 35.43749999300131),
                170.06010702131718,
            ),
            (40, 27): (
                (-93.3857279349451, 36.40540484288613),
                172.39265427405473,
            ),
            (123, 111): (
                (-93.98590224542299, 36.2177880383831),
                16.229116005034143,
            ),
            (136, 111): (
                (-95.47421682875286, 35.361624710523586),
                10.688976730137199,
            ),
        }

        for rup, angles in rup_angles.items():
            for i in [0, 1]:
                np.testing.assert_almost_equal(
                    _rup_angles[rup][i], angles[i], decimal=3
                )
        assert set(rup_angles.keys()) == set(_rup_angles.keys())

        _filt_bin_mat_adj_dict = {
            (3, 123): 1,
            (16, 86): 1,
            (27, 40): 1,
            (111, 123): 1,
            (123, 3): 1,
            (86, 16): 1,
            (40, 27): 1,
            (123, 111): 1,
        }

        assert dict(filtered_bin_adj_mat) == _filt_bin_mat_adj_dict


def test__lonlat_to_xyz():
    xyz = _lonlat_to_xyz(-122.606, 45.49)

    np.testing.assert_array_almost_equal(
        xyz, np.array([-0.37775834, -0.590548, 0.71312811])
    )


def test__lonlat_to_xyz_roundtrip():
    lon = -122.606
    lat = 45.49
    xyz = _lonlat_to_xyz(lon, lat)
    lon_, lat_ = _xyz_to_lonlat(xyz)
    np.testing.assert_approx_equal(lon, lon_)
    np.testing.assert_approx_equal(lat, lat_)


def test__geog_vec_to_xyz():
    pass
