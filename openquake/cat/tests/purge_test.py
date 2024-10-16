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
import filecmp
import numpy as np
import pandas as pd
import toml

from openquake.cat.hmg.purge import purge 

# -----------------------------------------------------------------------------

BASE_DATA_PATH = os.path.dirname(__file__)

# -----------------------------------------------------------------------------


class PurgeTestCase(unittest.TestCase):

    def setUp(self):

        self.data_path = os.path.join(BASE_DATA_PATH, 'data', 'test_purge')

    def test_case01(self):
        """
        tests that two events are purged from a catlogue 
        """
        # Reading h5 
        fname_cat = os.path.join(self.data_path, "test_h5_in.h5")
        test_file = os.path.join(self.data_path, "test_h5_purged.h5")
        base_file = os.path.join(self.data_path, "test_h5_purged_expected.h5")
        dups = os.path.join(self.data_path, "duplicates.csv")
        purge(fname_cat, test_file, dups) 
        os.remove(fname_cat+'.bak')

        # compare files
        cat1 = pd.read_hdf(test_file)
        cat2 = pd.read_hdf(base_file)
        self.assertTrue(cat1.equals(cat2))
#        self.assertTrue(filecmp.cmp(base_file, test_file))
        os.remove(test_file)
