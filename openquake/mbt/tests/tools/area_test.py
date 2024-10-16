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
:module:`openquake.mbt.tests.tools.area_test`
"""

import os
import unittest
import matplotlib.pyplot as plt

from openquake.mbt.tools.tr.catalogue import get_catalogue
from openquake.mbt.tools.area import (
    create_catalogue, load_geometry_from_shapefile)
from openquake.mbt.oqt_project import OQtModel

BASE_DATA_PATH = os.path.dirname(__file__)

PLOT = False


class SelectEqksWithinAreaTestCase(unittest.TestCase):
    """
    This class tests the selection of earthquakes within a polygon
    """

    def setUp(self):
        self.catalogue_fname = None

    def testcase01(self):
        """
        Simple area source
        """
        datafold = '../data/tools/area/case01/'
        datafold = os.path.join(BASE_DATA_PATH, datafold)
        #
        # create the source and set the geometry
        model = OQtModel('0', 'test')
        #
        # read geometries
        shapefile = os.path.join(datafold, 'polygon.shp')
        srcs = load_geometry_from_shapefile(shapefile)
        model.sources = srcs
        #
        # read catalogue
        self.catalogue_fname = os.path.join(datafold, 'catalogue.csv')
        cat = get_catalogue(self.catalogue_fname)
        #
        # select earthquakes within the polygon
        scat = create_catalogue(model, cat, ['1'])

        # cleaning
        os.remove(os.path.join(datafold, 'catalogue.pkl'))

        # check
        self.assertEqual(len(scat.data['longitude']), 5)

    def testcase02(self):
        """
        Area source straddling the IDL
        """

        datafold = '../data/tools/area/case02/'
        datafold = os.path.join(BASE_DATA_PATH, datafold)

        # create the source and set the geometry
        model = OQtModel('0', 'test')

        # read geometries
        shapefile = os.path.join(datafold, 'area_16.shp')
        srcs = load_geometry_from_shapefile(shapefile)
        model.sources = srcs

        # read catalogue
        self.catalogue_fname = os.path.join(datafold, 'catalogue.csv')
        cat = get_catalogue(self.catalogue_fname)

        # Select earthquakes within the polygon
        scat = create_catalogue(model, cat, ['16'])

        # Clean
        os.remove(os.path.join(datafold, 'catalogue.pkl'))

        # Plot
        if PLOT:

            fig, axs = plt.subplots(1, 1)
            fig.set_size_inches(8, 6)

            xp = srcs['16'].polygon.lons
            yp = srcs['16'].polygon.lats
            xp[xp < 0] = 360 + xp[xp < 0]
            plt.plot(xp, yp)

            xe = cat.data['longitude']
            ye = cat.data['latitude']
            xe[xe < 0] = 360 + xe[xe < 0]
            plt.plot(xe, ye, 'o')

            xi = scat.data['longitude']
            yi = scat.data['latitude']
            xi[xi < 0] = 360 + xi[xi < 0]
            plt.plot(xi, yi, 'x')

            plt.show()

        # check
        self.assertEqual(len(scat.data['longitude']), 4)
