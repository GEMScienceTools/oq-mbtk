# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
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
from openquake.cat.completeness.generate import _get_completenesses
from openquake.cat.completeness.plot import plot_completeness

PLOT = False


class GenerateTest(unittest.TestCase):
    """
    Test the generation of sets of completeness windows
    """

    def test_generate_simple(self):

        mags = [4.0, 5.0, 6.0]
        years =  [1980, 1990, 2000]
        num_steps = 0
        min_mag_compl = 4.0
        apriori_conditions_out = {}
        apriori_conditions_in = {}
        step = 6
        flexible = False

        # Generate the set of completeness windows
        perms, mags, years = _get_completenesses(
            mags, years, folder_out=None, num_steps=num_steps,
            min_mag_compl=min_mag_compl,
            apriori_conditions_out=apriori_conditions_out,
            apriori_conditions_in=apriori_conditions_in,
            step=step, flexible=flexible)

        expected = np.array([[0, 2, 2],
                             [0, 1, 2],
                             [0, 1, 1],
                             [0, 0, 2],
                             [0, 0, 1],
                             [0, 0, 0]])

        # Test
        np.testing.assert_equal(perms, expected)

        if PLOT:
            plot_completeness(perms, mags, years, 1980, 2005, 3.9, 6.1, '')

    def test_generate_flexible(self):
        """ test flexible is True """
        # As the previous test but with flexible = True

        mags = [4.0, 5.0, 6.0]
        years =  [1980, 1990, 2000]
        num_steps = 0
        min_mag_compl = 4.0
        apriori_conditions_out = {}
        apriori_conditions_in = {}
        step = 6
        flexible = True

        # Generate the set of completeness windows. The `perms` matrix in this
        # case has a length of 10 rather than 6 as in the previous case.
        perms, mags, years = _get_completenesses(
            mags, years, folder_out=None, num_steps=num_steps,
            min_mag_compl=min_mag_compl,
            apriori_conditions_out=apriori_conditions_out,
            apriori_conditions_in=apriori_conditions_in,
            step=step, flexible=flexible)

        expected = np.array([[ 0, -1, -1],
                             [ 0,  2, -1],
                             [ 0,  2,  2],
                             [ 0,  1, -1],
                             [ 0,  1,  2],
                             [ 0,  1,  1],
                             [ 0,  0, -1],
                             [ 0,  0,  2],
                             [ 0,  0,  1],
                             [ 0,  0,  0]])

        # Test
        np.testing.assert_equal(perms, expected)

        if PLOT:
            plot_completeness(perms, mags, years, 1980, 2005, 3.9, 6.1, '')


    def test_generate_apriori(self):
        """ test use of apriori conditions """
        # As the previous test but with two apriori conditions

        mags = [4.0, 5.0, 6.0]
        years =  [1980, 1990, 2000, 2010]
        num_steps = 0
        min_mag_compl = 4.0
        apriori_conditions_out = {1985: 5.5, 1995: 4.5}
        apriori_conditions_in = {}
        step = 6
        flexible = True

        # Generate the set of completeness windows. The `perms` matrix in this
        # case has a length of 10 rather than 6 as in the previous case.
        perms, mags, years = _get_completenesses(
            mags, years, folder_out=None, num_steps=num_steps,
            min_mag_compl=min_mag_compl,
            apriori_conditions_out=apriori_conditions_out,
            apriori_conditions_in=apriori_conditions_in,
            step=step, flexible=flexible)

        expected = np.array([[0, 2, 2, 2],
                             [0, 1, 2, 2],
                             [0, 1, 1, 2],
                             [0, 0, 2, 2],
                             [0, 0, 1, 2]])

        if PLOT:
            plot_completeness(perms, mags, years, 1980, 2005, 3.9, 6.1, '',
                              apriori_conditions_out)

        # Test
        np.testing.assert_equal(perms, expected)

    def test_generate_apriori_in_out(self):
        """ test use of apriori conditions in/out"""
        # As the previous test but also with apriori conditions in

        mags = [4.0, 5.0, 6.0]
        years =  [1980, 1990, 2000, 2010]
        num_steps = 0
        min_mag_compl = 4.0
        apriori_conditions_out = {1985: 5.5, 1995: 4.5}
        apriori_conditions_in = {1995: 5.5}
        step = 6
        flexible = True

        # Generate the set of completeness windows. The `perms` matrix in this
        # case has a length of 10 rather than 6 as in the previous case.
        perms, mags, years = _get_completenesses(
            mags, years, folder_out=None, num_steps=num_steps,
            min_mag_compl=min_mag_compl,
            apriori_conditions_out=apriori_conditions_out,
            apriori_conditions_in=apriori_conditions_in,
            step=step, flexible=flexible)

        expected = np.array([[0, 1, 1, 2], [0, 0, 1, 2]])

        if PLOT:
            plot_completeness(perms, mags, years, 1980, 2005, 3.9, 6.1, '',
                              apriori_conditions_out, apriori_conditions_in)

        # Test
        np.testing.assert_equal(perms, expected)
