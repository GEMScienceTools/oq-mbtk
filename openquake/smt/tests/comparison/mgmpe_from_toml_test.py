#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2024 GEM Foundation
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
import pandas as pd
import shutil
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
        # Set the output
        if not os.path.exists(self.output_directory): os.makedirs(
                self.output_directory)
    
    def test_mgmpe_executions(self):
        """
        Check GMPEs modified using mgmpe features specified within the toml
        are executed correctly and the expected values are returned
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        # Get matrix of predicted ground-motions per GMM
        mtxs_medians = compute_matrix_gmpes(config, mtxs_type='median')
        
        # Get observed values and target values
        observ_mtxs = pd.DataFrame(mtxs_medians[0])
        target_mtxs = pd.read_csv(
            os.path.join(base, 'target_medians_matrix.csv'))

        # Check equal   
        np.testing.assert_allclose(
            np.array(observ_mtxs), np.array(target_mtxs), atol=ATOL)
        
    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.output_directory)