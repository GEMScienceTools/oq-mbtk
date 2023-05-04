# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Tests the GM database parsing and selection
"""
import os
import unittest

from openquake.hazardlib import valid
from openquake.smt.residuals import gmpe_residuals as res
from openquake.hazardlib.gsim.mgmpe import modifiable_gmpe as mgmpe
from openquake.smt.comparison.utils_gmpes import al_atik_sigma_check


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__),'data')

class utils_gmpes_test(unittest.TestCase):
    '''
    Test utils_gmpes functions
    '''
    
    def test_al_atik_sigma_check(self):
        """
        Check that sigma for GMPEs is checked correctly, and that Al-Atik (2015)
        sigma model is correctly implemented if required
        """
        # Only YA15 should be flagged and Al-Atik 2015 added as sigma model
        filename = os.path.join(BASE_DATA_PATH, 'utils_gmpes_test.toml')
        residuals = res.Residuals.from_toml(filename)
        
        imts = ['PGA']
        for idx, inputted_gmpe in enumerate(residuals.gmpe_list):
            gmpe_outputted, gmpe_sigma_flag = al_atik_sigma_check(inputted_gmpe,
                                                                  imts[0], task
                                                                  = 'residual')
            if idx == 0:
                self.assertTrue(gmpe_sigma_flag == True) # Check flagged
                # Get expected modified GMPE
                tmp_gmm = valid.gsim(str(inputted_gmpe).split('_toml=')[
                    1].replace(')',''))
                tmp_gmpe = str(tmp_gmm).split(']')[0].replace('[','')
                kwargs = {'gmpe': {tmp_gmpe: {'sigma_model_alatik2015': {}}},
                          'sigma_model_alatik2015': {}}
                inputted_gmpe = mgmpe.ModifiableGMPE(**kwargs)
                self.assertTrue(gmpe_outputted == mgmpe.ModifiableGMPE(**kwargs)) # Check is modified GMPE
            if idx == 1:
                self.assertTrue(gmpe_sigma_flag == False) # Check not flagged
                self.assertTrue(gmpe_outputted == valid.gsim(
                    inputted_gmpe.split('(')[0])) # Check not modified GMPE
                    
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
