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
import numpy
import pandas as pd
import filecmp
import unittest
import tempfile
import matplotlib.pyplot as plt
from openquake.sub.cross_sections import Trench
from openquake.hazardlib.geo.geodetic import point_at
from openquake.sub.create_multiple_cross_sections import get_cs

PLOT = False
BASE_PATH = os.path.dirname(__file__)


class CrossSectionIntersectionTest(unittest.TestCase):

    def setUp(self):
        trench_fname = os.path.join(BASE_PATH, 'data', 'traces', 'trench.xyz')
        data = numpy.loadtxt(trench_fname)
        self.trench = Trench(data)

    def test01(self):

        tmp_fname = 'trash'
        _, tmp_fname = tempfile.mkstemp()

        cs_length = 400
        cs_depth = 100
        intd = 100
        handle, tmp_fname = tempfile.mkstemp()
        get_cs(self.trench, 'tmp.txt', cs_length, cs_depth, intd, 0, tmp_fname)

        if PLOT:
            _ = plt.figure()
            columns = ['lon', 'lat', 'dep', 'len', 'azim', 'id', 'fname']
            df = pd.read_csv(tmp_fname, names=columns, delimiter=' ')
            for i, row in df.iterrows():
                ex, ey = point_at(row.lon, row.lat, row.azim, row.len)
                plt.plot([row.lon], [row.lat], 'o')
                plt.text(row.lon, row.lat, '{:d}'.format(row.id))
                plt.plot([row.lon, ex], [row.lat, ey], '-')
            plt.show()

        expected = os.path.join(BASE_PATH, 'data', 'traces', 'expected.txt')
        msg = 'The two files do not match'

        df_computed = pd.read_csv(tmp_fname, delimiter=',')
        df_expected = pd.read_csv(expected, delimiter=',')

        pd.testing.assert_frame_equal(df_computed, df_expected)
        # self.assertTrue(filecmp.cmp(tmp_fname, expected), msg)
