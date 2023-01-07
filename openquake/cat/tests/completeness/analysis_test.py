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

import sys
import os
import unittest
import numpy as np
from openquake.cat.completeness.analysis import (
    clean_completeness, get_earliest_year_with_n_occurrences)
from openquake.cat.completeness.generate import _get_completenesses
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue

DATAFOLDER = os.path.join(os.path.dirname(__file__), 'data')


class TestGetYear(unittest.TestCase):

    def setUp(self):
        fname = os.path.join(DATAFOLDER, 'cat_00.csv')
        self.cat02 = _load_catalogue(fname)
        fname = os.path.join(DATAFOLDER, 'cat_05.csv')
        self.cat05 = _load_catalogue(fname)

    def test_min_year_01(self):
        nocc = 3
        ctab = np.array([[1960., 4.6], [1900., 5.]])
        fun = get_earliest_year_with_n_occurrences
        eyea = fun(ctab, self.cat02, nocc)
        self.assertEqual(2, len(eyea))

    def test_min_year_02(self):
        nocc = 2
        ctab = np.array([[2000., 4.6]])
        fun = get_earliest_year_with_n_occurrences
        eyea = fun(ctab, self.cat05, nocc)
        np.testing.assert_equal([1966], eyea)

class TestCleanCompleteness(unittest.TestCase):

    def test01(self):
        compl = np.array([[1930., 4.], [1900., 4.]])
        computed = clean_completeness(compl)
        expected = np.array([[1900., 4.]])
        np.testing.assert_array_equal(computed, expected)

    def test02(self):
        compl = np.array([[1990, 5.], [1960, 7.], [1900, 7.]])
        computed = clean_completeness(compl)
        expected = np.array([[1990., 5.], [1900., 7.]])
        np.testing.assert_array_equal(computed, expected)


class TestCompletenessGeneration(unittest.TestCase):

    def setUp(self):
        self.expect01 = np.array([[2, 2, 2, 2, 2, 2],
                                  [1, 2, 2, 2, 2, 2],
                                  [1, 1, 2, 2, 2, 2],
                                  [1, 1, 1, 2, 2, 2],
                                  [1, 1, 1, 1, 2, 2],
                                  [1, 1, 1, 1, 1, 2],
                                  [1, 1, 1, 1, 1, 1],
                                  [0, 2, 2, 2, 2, 2],
                                  [0, 1, 2, 2, 2, 2],
                                  [0, 1, 1, 2, 2, 2],
                                  [0, 1, 1, 1, 2, 2],
                                  [0, 1, 1, 1, 1, 2],
                                  [0, 1, 1, 1, 1, 1],
                                  [0, 0, 2, 2, 2, 2],
                                  [0, 0, 1, 2, 2, 2],
                                  [0, 0, 1, 1, 2, 2],
                                  [0, 0, 1, 1, 1, 2],
                                  [0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 2, 2, 2],
                                  [0, 0, 0, 1, 2, 2],
                                  [0, 0, 0, 1, 1, 2],
                                  [0, 0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 2, 2],
                                  [0, 0, 0, 0, 1, 2],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 2],
                                  [0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0]])

        self.expect02 = np.array([[0, 2, 2, 2, 2, 2],
                                  [0, 1, 2, 2, 2, 2],
                                  [0, 1, 1, 2, 2, 2],
                                  [0, 1, 1, 1, 2, 2],
                                  [0, 1, 1, 1, 1, 2],
                                  [0, 1, 1, 1, 1, 1],
                                  [0, 0, 2, 2, 2, 2],
                                  [0, 0, 1, 2, 2, 2],
                                  [0, 0, 1, 1, 2, 2],
                                  [0, 0, 1, 1, 1, 2],
                                  [0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 2, 2, 2],
                                  [0, 0, 0, 1, 2, 2],
                                  [0, 0, 0, 1, 1, 2],
                                  [0, 0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 2, 2],
                                  [0, 0, 0, 0, 1, 2],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 2],
                                  [0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0]])

        self.expect03 = np.array([[0, 1, 1, 2, 2, 2],
                                  [0, 1, 1, 1, 2, 2],
                                  [0, 1, 1, 1, 1, 2],
                                  [0, 1, 1, 1, 1, 1],
                                  [0, 0, 1, 2, 2, 2],
                                  [0, 0, 1, 1, 2, 2],
                                  [0, 0, 1, 1, 1, 2],
                                  [0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 2, 2, 2],
                                  [0, 0, 0, 1, 2, 2],
                                  [0, 0, 0, 1, 1, 2],
                                  [0, 0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 2, 2],
                                  [0, 0, 0, 0, 1, 2],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 2],
                                  [0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0]])

        self.expect05 = np.array([[0, 1, 1, 1, 1, 2],
                                  [0, 1, 1, 1, 1, 1],
                                  [0, 0, 1, 1, 1, 2],
                                  [0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 2],
                                  [0, 0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 2],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 2],
                                  [0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0]])

    def test01(self):
        years = np.array([1900, 1930, 1960, 1970, 1980, 1990])
        mags = np.array([5.0, 6.0, 7.0])
        disps, mags, years = _get_completenesses(mags=mags, years=years,
                                                step=3, min_mag_compl=7.0)
        np.testing.assert_array_almost_equal(self.expect01, disps)

    def test02(self):
        years = np.array([1900, 1930, 1960, 1970, 1980, 1990])
        mags = np.array([5.0, 6.0, 7.0])
        disps, mags, years = _get_completenesses(mags=mags, years=years,
                                                min_mag_compl=5.0, step=3)
        np.testing.assert_array_almost_equal(self.expect02, disps)

    def test03(self):
        """ Testing a-priori conditions """
        years = np.array([1900, 1930, 1960, 1970, 1980, 1990])
        mags = np.array([5.0, 6.0, 7.0])
        # This implies that the column for the year just after 1965 (column 3)
        # has an index for magnitude lower or equal to 6.2 (i.e. 0 or 1)
        conds = {1965: 6.2}
        disps, mags, years = _get_completenesses(mags=mags, years=years,
                                                min_mag_compl=5.0, step=3,
                                                apriori_conditions=conds)
        np.testing.assert_array_almost_equal(self.expect03, disps)

    def test04(self):
        """ Testing a-priori conditions """
        years = np.array([1900, 1930, 1960, 1970, 1980, 1990])
        mags = np.array([5.0, 6.0, 7.0])
        # Conditions
        conds = {1965: 7.0}
        disps, mags, years = _get_completenesses(mags=mags, years=years,
                                                min_mag_compl=5.0, step=3,
                                                apriori_conditions=conds)
        np.testing.assert_array_almost_equal(self.expect02, disps)

    def test05(self):
        """ As 03 but now with more selective condition """
        years = np.array([1900, 1930, 1960, 1970, 1980, 1990])
        mags = np.array([5.0, 6.0, 7.0])
        # In this case the column for 1930 (the fifth one in disps must have
        # index for 0 or 1
        conds = {1920: 6.2}
        disps, mags, years = _get_completenesses(mags=mags, years=years,
                                                min_mag_compl=5.0, step=3,
                                                apriori_conditions=conds)
        np.testing.assert_array_almost_equal(self.expect05, disps)


class TestCompletenessGenerationWithOptional(unittest.TestCase):

    def setUp(self):
        self.expect01 = np.array([[0, 2, 2, 2, 2, 2],
                                  [0, 1, 2, 2, 2, 2],
                                  [0, 1, 1, 2, 2, 2],
                                  [0, 1, 1, 1, 2, 2],
                                  [0, 1, 1, 1, 1, 2],
                                  [0, 1, 1, 1, 1, 1],
                                  [0, 0, 2, 2, 2, 2],
                                  [0, 0, 1, 2, 2, 2],
                                  [0, 0, 1, 1, 2, 2],
                                  [0, 0, 1, 1, 1, 2],
                                  [0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 2, 2, 2],
                                  [0, 0, 0, 1, 2, 2],
                                  [0, 0, 0, 1, 1, 2],
                                  [0, 0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 2, 2],
                                  [0, 0, 0, 0, 1, 2],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 2],
                                  [0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0]])
        fname = 'res_01_flexible.gz'
        self.expect01_flx = np.loadtxt(os.path.join(DATAFOLDER, fname))

    def test_flexible_01(self):
        """ First test for the flexible option """
        years = np.array([1900, 1930, 1960, 1970, 1980, 1990])
        mags = np.array([5.0, 6.0, 7.0])
        disps, mags, years = _get_completenesses(mags=mags, years=years,
                                                min_mag_compl=5.0, step=3,
                                                flexible=True)

        # Check that if we remove the dispositions with the dummy variables
        # we get the same dispositions generated with flexible=False
        tmp = disps
        tmp = tmp[np.all(tmp >= 0, axis=1), :]
        np.testing.assert_array_almost_equal(self.expect01, tmp)

        # Now check that the result is as expected
        np.testing.assert_array_almost_equal(self.expect01_flx, disps)

