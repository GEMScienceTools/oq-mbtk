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

"""
Module create_map_functions_test
"""

import os
import numpy
import pandas
import unittest
import geopandas as gpd

from openquake.ghm.create_homogenised_curves import get_hcurves_geodataframe
from openquake.ghm.create_homogenised_curves import recompute_probabilities

DATA = os.path.join(os.path.dirname(__file__), 'data', 'hazardcurve')


class ReadHazardCurveTestCase(unittest.TestCase):
    """ Check information read from hazard curve file """

    def test_read_file01(self):
        """ Read hazard curve file 01"""
        fname = os.path.join(DATA, 'hazard_curve-mean-PGA_18150.csv')
        gdf, header = get_hcurves_geodataframe(fname)
        self.assertTrue(header[0] == 'mean', 'Header not matching')
        self.assertTrue(header[1] == 1.0, 'Header not matching')
        self.assertTrue(header[2] == 'PGA', 'Header not matching')
        msg = 'The output type is not a GeoDataFrame'
        self.assertTrue(isinstance(gdf, gpd.GeoDataFrame), msg)
        expected = numpy.array([6.762546E-02])
        lab = 'poe-0.005'
        numpy.testing.assert_almost_equal(expected, gdf[lab].values)

    def test_read_file02(self):
        """ Read hazard curve file 02"""
        fname = os.path.join(DATA, 'hazard_curve-mean-PGA_22087.csv')
        gdf, header = get_hcurves_geodataframe(fname)
        self.assertTrue(header[0] == 'mean', 'Header not matching')
        self.assertTrue(header[1] == 50.0, 'Header not matching')
        self.assertTrue(header[2] == 'PGA', 'Header not matching')
        prbs = numpy.array([0.99999999, 0.99999999, 0.99999999])
        rate = -numpy.log(1. - prbs) / 50.
        expected = 1. - numpy.exp(-rate)
        lab = 'poe-0.005'
        numpy.testing.assert_almost_equal(expected, gdf[lab].values)


class HomogeniseHazardCurvesDataFrameTest(unittest.TestCase):
    """ Check the conversion of the probabilities of exceedance """

    def test01(self):
        """ Recompute the proabiility of exceedance """
        fname = os.path.join(DATA, 'hazard_curve-mean-PGA_22087.csv')
        daf = pandas.read_csv(fname, skiprows=1)

        lab = 'poe-0.005'
        prbs = numpy.array([0.99999999, 0.99999999, 0.99999999])
        rate = -numpy.log(1. - prbs) / 50.
        expected = 1. - numpy.exp(-rate)
        new_df = recompute_probabilities(daf, 50.0, 1.0)
        computed = new_df[lab].values
        numpy.testing.assert_almost_equal(expected, computed)
