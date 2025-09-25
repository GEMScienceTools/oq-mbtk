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
from openquake.cat.isf_catalogue import ISFCatalogue
from openquake.cat.parsers.isf_catalogue_reader import ISFReader

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

SETTINGS_ISF1 = """
[magnitude.TIR.Ml]
low_mags = [0.0]
conv_eqs = ["0.96 * m + 0.23"]
sigma = [0.1]

"""

SETTINGS_ISF2 = """
[magnitude.AFAD.MW]
low_mags = [0.0]
conv_eqs = ["m"]
sigma = [0.1]

[magnitude.ATH.ML]
low_mags = [0.0]
conv_eqs = ["1.1 * m -0.2"]
sigma = [0.1]

[magnitude.TIR.Ml]
low_mags = [0.0]
conv_eqs = ["0.96 * m + 0.23"]
sigma = [0.1]

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
        
class HomogeniseIsfTestCase(unittest.TestCase):
     
    def setUp(self):
        self.fname_isf6 = os.path.join(BASE_PATH, 'data', 'cat06.isf')
         
    def test_case01(self):
        """
        tests homogenisation of an isf file with multiple agencies
        only one magnitude conversion
        """
        #parser = ISFReader(self.fname_isf2)
        cat = ISFCatalogue(self.fname_isf6, name = "isf")
        parser = ISFReader(self.fname_isf6)
        catisf = parser.read_file("tisf", "Test isf")

        otab, mtab = catisf.get_origin_mag_tables()
        work = pd.merge(pd.DataFrame(otab), pd.DataFrame(mtab), on=["eventID", "originID"])
         
        rules = toml.loads(SETTINGS_ISF1)
        save, work = process_magnitude(work, rules['magnitude'])
        
        results = save.magMw.values
        expected = [3.878, 3.686, 3.686, 2.822, 3.302, 3.494, 4.07 ]
        aaeq(results, expected, decimal=6)
         
         
    def test_case02(self):
    
        """
        tests homogenisation of an isf file with multiple agencies
        use multiple agencies
        """
        #parser = ISFReader(self.fname_isf2)
        cat = ISFCatalogue(self.fname_isf6, name = "isf")
        parser = ISFReader(self.fname_isf6)
        catisf = parser.read_file("tisf", "Test isf")

        otab, mtab = catisf.get_origin_mag_tables()
        work = pd.merge(pd.DataFrame(otab), pd.DataFrame(mtab), on=["eventID", "originID"])
         
        rules = toml.loads(SETTINGS_ISF2)
        save, work = process_magnitude(work, rules['magnitude'])
       
        results = save.magMw.values
        # Note magnitudes will be in the order selected!
        expected = [4.0, 3.43, 3.32, 3.43, 2.66, 3.302, 3.494 ]
        aaeq(results, expected, decimal=6)         
