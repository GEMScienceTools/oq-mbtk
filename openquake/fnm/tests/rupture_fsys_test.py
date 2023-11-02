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

from openquake.fnm.fault_system import get_rups_fsys, _get_area_fraction
from openquake.fnm.tests.connection_test import _get_surfs
from openquake.fnm.rupture import (
    _check_rupture_has_connections,
    check_rup_exists,
)

HERE = pathlib.Path(__file__).parent


def _get_array_from_list(dat):
    mlen = np.max([len(i) for i in dat])
    out = np.ones((len(dat), mlen)) * -1
    for i, tmp in enumerate(dat):
        out[i, 0:len(tmp)] = tmp
    return out


class TestRuptureFraction(unittest.TestCase):
    def test_compute_area_fraction(self):
        """Check the fraction of the total rupture area on each section"""
        rups = [[1, 2], [3], [1, 3, 4]]
        areas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        computed = _get_area_fraction(rups, areas)
        expected = [[0.4, 0.6], [1.0], [0.182, 0.364, 0.454]]
        # Get arrays
        expected = _get_array_from_list(expected)
        computed = _get_array_from_list(computed)
        np.testing.assert_array_almost_equal(computed, expected)


class TestRuptureExists(unittest.TestCase):
    def test_rupture_exists(self):
        """Check if a rupture exists in itself"""
        rups = np.array(
            [
                np.array([0.0, 0.0, 24.0, 12.0, 15.0, 3.0, 1.0]),
                np.array([0.0, 0.0, 12.0, 12.0, 15.0, 3.0, 0.0]),
                np.array([0.0, 12.0, 12.0, 12.0, 15.0, 3.0, 2.0]),
            ]
        )
        rups = rups.astype(int)
        res = check_rup_exists(rups, rups)
        assert res


class TestCheckRuptureHasConnection(unittest.TestCase):
    def test_rupture_has_connection(self):
        """Check if rupture has connection"""
        conns = np.array(
            [
                [0, 1, 0, 0, 12, 12, 0, 12, 12, 12],
                [1, 2, 0, 0, 12, 12, 0, 12, 12, 12],
            ]
        )
        rup = np.array([0.0, 0.0, 24.0, 12.0, 7.0, 1.0, 2.0])
        computed = _check_rupture_has_connections(conns, rup)
        # Expected:
        # 1 - Since we found connection
        # 1 - Since the index of the other section is 1
        # 1 - Since the other section is the first one in the connection
        # 1 - Since the incremental index for the connection found is 1 - The
        #     first one is filtered out upfront
        expected = np.array([[1, 1, 1, 1]])
        np.testing.assert_array_equal(computed, expected)


class TestCreateRupturesSystem(unittest.TestCase):
    def test_generate_ruptures_fsys_2sections(self):
        # In this test we find only one connection between the eastmost
        # subsection in the first section and the westernmost subsection in the
        # second subsection
        surfs = _get_surfs()
        surfs = surfs[0:2]

        # Criteria for building the fault system
        threshold = 20.0

        settings = {}
        criteria = {}
        key = 'min_distance_between_subsections'
        criteria[key] = {'threshold_distance': threshold}
        settings['connections'] = criteria
        settings['ruptures'] = {}
        settings['ruptures']['aspect_ratios'] = [0, 100]
        settings['ruptures']['magnitude_scaling_rel'] = 'generic'
        settings['general'] = {}
        settings['general']['subsection_size'] = [-0.5, -1]

        results = get_rups_fsys(surfs, settings)
        rups = results['ruptures_single_section_indexes']
        sec_idxs = results['ruptures_indexes_of_sections_involved']

        # The expected number if ruptures is 36 because we have 20 ruptures
        # occurring on each individual section + 16 ruptures occurring jointly
        # on the two sections. The only connection is between subsection 0 on
        # section 0 and subsection 3 on section 1.
        #
        #                  |---|---|---|---|    - SECTION 1
        #                  24  18  12  6   0
        #  |---|---|---|---|                    - SECTION 0
        #  24  18  12  6   0
        #
        #  Example of composite rupture:
        #
        #                  |xxxx|---|---|---|    - SECTION 1
        #                  24  18  12  6   0
        #  |---|---|---|xxx|                    - SECTION 0
        #  24  18  12  6   0

        # Check the number of ruptures
        expected = 36
        np.testing.assert_equal(len(rups), expected)

        # Check the index of the sections composing each rupture
        expected_idxs = [
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
        expected_idxs = _get_array_from_list(expected_idxs)
        computed_idxs = _get_array_from_list(sec_idxs)
        np.testing.assert_array_equal(computed_idxs, expected_idxs)

    def test_generate_ruptures_fsys_3sections(self):
        # The expected number if ruptures is 78 because we have 30 ruptures
        # occurring on each individual section + 32 ruptures occurring jointly
        # on two sections and 16 ruptures occurring on 3 sections. The only
        # connections are between subsection 0 on section 0 and subsection 3 on
        # section 1 and between subsection 0 on section 1 and subsection 3 on
        # section 2.
        #
        #                                   |---|---|---|---|    - SECTION 2
        #                                   24  18  12  6   0
        #                  |---|---|---|---|                     - SECTION 1
        #                  24  18  12  6   0
        #  |---|---|---|---|                                     - SECTION 0
        #  24  18  12  6   0
        #
        #  Example of composite rupture (level 3):
        #
        #                                   |xxx|---|---|---|    - SECTION 2
        #                                   24  18  12  6   0
        #                  |xxxx|xxx|xxx|xxx|                    - SECTION 1
        #                  24  18  12  6   0
        #  |---|---|---|xxx|                                     - SECTION 0
        #  24  18  12  6   0

        surfs = _get_surfs()
        threshold = 20.0

        settings = {}
        criteria = {}
        key = "min_distance_between_subsections"
        criteria[key] = {"threshold_distance": threshold}
        settings["connections"] = criteria
        settings["ruptures"] = {}
        settings["ruptures"]["aspect_ratios"] = np.array([0, 100], dtype=int)
        settings["ruptures"]["magnitude_scaling_rel"] = "generic"
        settings["general"] = {}
        settings["general"]["subsection_size"] = [-0.5, -1]

        # Get the ruptures from the fault system
        results = get_rups_fsys(surfs, settings)
        rups = results['ruptures_single_section_indexes']
        nsections = np.array([len(r) for r in rups])

        # Ruptures level 1
        computed = np.sum(nsections == 1)
        expected = 30
        np.testing.assert_equal(computed, expected)

        # Ruptures level 2
        computed = np.sum(nsections == 2)
        expected = 32
        np.testing.assert_equal(computed, expected)

        # Ruptures level 3
        computed = np.sum(nsections == 3)
        expected = 16
        np.testing.assert_equal(computed, expected)
