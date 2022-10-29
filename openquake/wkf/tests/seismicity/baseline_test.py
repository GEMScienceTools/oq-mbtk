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

import os
import pandas as pd
import tempfile
import unittest
from openquake.wkf.seismicity.baseline import add_baseline_seismicity

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'baseline')


class AddBaselineTestCase(unittest.TestCase):
    """ Adds baseline to a polygon source """

    def test_add_baseline(self):
        """ Test adding a baseline seismicity """
        # The input polygon contains three cells given an h3 resolution equal
        # to 5. The area of the three cells is:
        # - geohash 85754e2bfffffff: 188.65961426 km2
        # - geohash 85754e77fffffff: 188.09231588 km2
        # - geohash 85754e3bfffffff: 188.79162082 km2
        # Let's consider the first cell. Given a GR a-value of 0.863 per km2
        # [i.e. 7.2946 events/(yr*km2) with magnitude larger than 0] this will
        # generate 188.65961426 * 7.2946 eqks/(yr*km2) = 1376.1917 eqks/yr.
        # This corresponds to a GR a-value of 3.1387
        folder_name = None
        folder_name_out = tempfile.mkdtemp()
        fname_config = os.path.join(DATA_PATH, 'config.toml')
        fname_poly = os.path.join(DATA_PATH, 'polygon.shp')
        add_baseline_seismicity(folder_name, folder_name_out, fname_config,
                                fname_poly)
        tdf = pd.read_csv(os.path.join(folder_name_out, '0.csv'))
        res = tdf[(abs(tdf.lat - 0.05652207) < 1e-4) &
                  (abs(tdf.lon - 0.10945153) < 1e-4)]
        msg = 'Wrong value of agr'
        self.assertAlmostEqual(res.agr.values[0], 3.1387, places=4,
                               message=msg)
