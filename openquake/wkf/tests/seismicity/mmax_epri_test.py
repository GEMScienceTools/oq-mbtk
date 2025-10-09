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
import tempfile
import numpy as np
import pandas as pd
from openquake.wkf.seismicity.mmax_epri import (
    get_mmax_pmf, get_composite_likelihood)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class MmaxEPRITest(unittest.TestCase):
    """ Tests the calculation of the EPRI mmax distribution """

    def test_01(self):
        """
        Tests the PMF described in Figure 5.2.1-1 of the CEUS-SSC report. See
        page 432. Since I did not manage to find it in the report, I assume a
        value of bGR = 1.0.
        """
        n_gt_n0 = 2.0
        mag0 = 4.5
        mmaxobs = 5.3
        pri_mean = 6.4
        pri_std = 0.85
        bgr = 1.0
        wdt = 0.5
        bins = np.arange(5.25, 9.5, wdt)
        # Expected results manually digitized
        fname = os.path.join(DATA_PATH, 'ceus_fig_5.2.1-1.csv')
        expected = np.loadtxt(fname, delimiter=',')

        wei, mag = get_mmax_pmf(
            pri_mean, pri_std, bins, mmaxobs=mmaxobs, mag0=mag0,
            n_gt_n0=n_gt_n0, bgr=bgr)
        np.testing.assert_almost_equal(wei, expected[:, 1], decimal=2)


class CompositeLikelihoodTest(unittest.TestCase):
    """ Tests the calculation of a composite likelihood"""

    def test_clikl(self):
        fname_cat = os.path.join(DATA_PATH, 'ctlg_composite_prior.csv')
        dfc = pd.read_csv(fname_cat)
        ccomp = [[2000.0, 3.9], [1950, 5.5]]
        bgr = 1.0
        wdt = 0.5
        mag0 = np.ceil(np.min(dfc.magnitude)/0.1)*0.1
        bins = np.arange(mag0, 9.5, wdt)
        mupp, lkl = get_composite_likelihood(dfc, ccomp, bgr)
        pri_mean = 6.4
        pri_std = 0.85
        out_folder = tempfile.mkdtemp()
        fig_name = os.path.join(out_folder, 'mmax.png')
        wei, mag = get_mmax_pmf(pri_mean, pri_std, bins, mupp=mupp,
                                likelihood=lkl, fig_name=fig_name)
