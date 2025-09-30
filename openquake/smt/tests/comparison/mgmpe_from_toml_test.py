#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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
Tests for execution and expected values from mgmpe features when specified in
an SMT format toml input file
"""
import os
import numpy as np
import shutil
import pickle
import unittest

from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_compare_gmpes import compute_matrix_gmpes


# Base path
base = os.path.join(os.path.dirname(__file__), "data")

# Absolute tolerance
ATOL = 1E-6


class ModifyGroundMotionsTestCase(unittest.TestCase):
    """
    Test case for the execution and expected values from mgmpe features when
    specified within an SMT format TOML input file
    """
    @classmethod 
    def setUpClass(self):
        self.input_file = os.path.join(base, "mgmpe_inputs.toml")
        self.output_directory = os.path.join(base, 'mgmpe_test')
        self.exp_mgmpe = os.path.join(base, "exp_mgmpe.pkl")
        # Set the output
        if not os.path.exists(self.output_directory): os.makedirs(
                self.output_directory)
    
    def test_mgmpe_from_toml(self):
        """
        Check GMPEs modified using mgmpe features specified within the toml
        are executed correctly and the expected values are returned
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        # Get matrices of predicted ground-motions per GMM
        obs_matrix = compute_matrix_gmpes(config, mtxs_type='median')
        
        # Remove the gmpe_list key
        del obs_matrix['gmpe_list']

        # Load the matrices of expected ground-motions per GMM         
        if not os.path.exists(self.exp_mgmpe):
            with open(self.exp_mgmpe, 'wb') as f: # Write if doesn't exist
                pickle.dump(obs_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.exp_mgmpe, 'rb') as f:
                exp_matrix = pickle.load(f)
                
        # Now check matrix per imt (key)
        assert obs_matrix.keys() == exp_matrix.keys()
        for key in obs_matrix:
            np.testing.assert_allclose(
                 obs_matrix[key], exp_matrix[key], atol=ATOL)

    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.output_directory)