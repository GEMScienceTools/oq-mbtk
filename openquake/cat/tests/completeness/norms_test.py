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
import unittest
import numpy as np
from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.cat.completeness.norms import get_norm_optimize_b, get_norm_optimize_c

DATA = os.path.join(os.path.dirname(__file__), 'data')

class NormBTest(unittest.TestCase):

    def setUp(self):
        dat = [[1900, 6.0],
               [1980, 6.0],
               [1970, 5.0],
               [1980, 5.0],
               [1980, 5.7],
               [1990, 5.0]]
        dat = np.array(dat)
        cat = Catalogue()
        cat.load_from_array(['year', 'magnitude'], dat)
        cat = _add_defaults(cat)
        cat.data["dtime"] = cat.get_decimal_time()
        self.cat = cat
        self.compl = np.array([[1980, 5.0], [1950, 5.9]])

    def test_case01(self):
        mbinw = 0.5
        ybinw = 10.0
        aval = 2.0
        bval = 1.0
        cmag, t_per, n_obs = get_completeness_counts(self.cat, self.compl,
                                                     mbinw)
        norm = get_norm_optimize_b(aval, bval, self.compl, self.cat, mbinw,
                                   ybinw)

    def test_case02(self):

        mbinw = 0.1
        ybinw = 10.0
        aval = 4.6
        bval = 1.0

        aval = 3.8897
        bval = 0.8332

        tmp = np.loadtxt(os.path.join(DATA, 'cat_norm_02.csv'), skiprows=1,
                         delimiter=',')

        cat = Catalogue()
        cat.load_from_array(['year', 'magnitude'], tmp)
        cat = _add_defaults(cat)
        cat.data["dtime"] = cat.get_decimal_time()

        compl = np.array([[2000, 4.4], [1990, 5.0], [1980, 5.8]])
        #compl = np.array([[1995, 4.4], [1985, 5.4]])
        #compl = np.array([[2000, 4.4]])

        compl = np.array([[2000, 4.4], [1990, 5.5]])

        compl = np.array([[2000, 4.4], [1985, 5.8]]);
        aval = 3.8004918570326267; bval = 0.8114202323942403; rmag = 4.55; rmagu = 10.15

        cmag, t_per, n_obs = get_completeness_counts(cat, compl, mbinw)
        #norm = get_norm_optimize_b(aval, bval, compl, cat, mbinw, ybinw=ybinw,
        #                           mmin=rmag, mmax=rmagu)
        norm = get_norm_optimize_c(cat, aval, bval, compl, 2022)
        print(f'{norm:.5e}')
