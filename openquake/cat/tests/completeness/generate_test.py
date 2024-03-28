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
from openquake.cat.completeness.norms import get_completeness_matrix


class CompletenessMatrixTest(unittest.TestCase):

    def setUp(self):
        dat = [[1900, 6.0],
               [1980, 6.0],
               [1970, 5.0],
               [1980, 5.0],
               [1990, 5.0]]
        dat = np.array(dat)
        cat = Catalogue()
        cat.load_from_array(['year', 'magnitude'], dat)
        cat = _add_defaults(cat)
        cat.data["dtime"] = cat.get_decimal_time()
        self.cat = cat
        self.compl = np.array([[1980, 5.0], [1950, 5.9]])

    def test_case01(self):
        binw = 0.5
        oin, out, cmags, cyeas = get_completeness_matrix(self.cat, self.compl,
                                                         0.5, 10.0)
        oin_expected = np.array([[-1., -1., -1., -1., -1., -1., -1., -1.,  1.,  1.],
                                 [-1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.],
                                 [-1., -1., -1., -1., -1.,  0.,  0.,  0.,  1.,  0.]])
        out_expected = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1., -1.],
                                 [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1.],
                                 [ 1.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1.]])
        np.testing.assert_array_equal(oin, oin_expected)
        np.testing.assert_array_equal(out, out_expected)

        # Check the consistency of results with the ones provided by the
        # completeness count
        cmag, t_per, n_obs = get_completeness_counts(self.cat, self.compl,
                                                     binw)
        oin[oin < 0] = 0.0
        np.testing.assert_array_equal(np.sum(oin, axis=1), n_obs)
