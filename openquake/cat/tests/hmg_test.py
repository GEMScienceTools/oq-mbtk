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
import toml

from openquake.cat.hmg.hmg import process_dfs, process_magnitude

aeq = np.testing.assert_equal
aaeq = np.testing.assert_almost_equal


BASE_PATH = os.path.dirname(__file__)

SETTINGS_HOMOG = """
[magnitude.NEIC.mb] 
#from weatherill
low_mags = [5.0, 6.0]
conv_eqs = ["0.8 * m + 0.2", "m"]
sigma = [0.283, 0.283]
"""



class HomogeniseNEICmbTestCase(unittest.TestCase):

    def setUp(self):

        self.data_path = os.path.join(BASE_PATH, 'data', 'test_hmg')

    def test_case01(self):
        """
        tests that magnitudes are converted correctly for an example
        with two "low_mags" values, neither of which is 0
        """
        td = toml.loads(SETTINGS_HOMOG)

        # Reading otab and mtab
        fname_otab = os.path.join(self.data_path, "test_hmg_otab.h5")
        odf = pd.read_hdf(fname_otab)
        fname_mtab = os.path.join(self.data_path, "test_hmg_mtab_clip.h5")
        mdf = pd.read_hdf(fname_mtab)
        work = pd.merge(odf, mdf, on=["eventID", "originID"])
        save = pd.DataFrame(columns=work.columns)

        rules = toml.loads(SETTINGS_HOMOG)
        #breakpoint()
        save, work = process_magnitude(work, rules['magnitude'])

        results = save.magMw.values
        expected = [4.36, 4.76, 6.8 ]
        aaeq(results, expected, decimal=6)
