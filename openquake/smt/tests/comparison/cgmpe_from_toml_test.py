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
Tests for execution and correctness of values from a conditional GMPE
when specified in an SMT format toml input file
"""
import os
import pandas as pd
import numpy as np
import shutil
import unittest

from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_compare_gmpes import compute_matrix_gmpes


# Base path
BASE = os.path.join(os.path.dirname(__file__), "data")


def matrix_to_df(matrix):
    """
    Convert matrix of ground-motions to dataframe.
    """
    store = {}
    for imt in matrix.keys():
        store[str(imt)] = np.array(matrix[imt]).flatten()

    return pd.DataFrame(store)


class ConditionGroundMotionsTestCase(unittest.TestCase):
    """
    Test case for the execution and correctness of values from a conditional
    gmpe specified within an SMT format TOML input file
    """
    @classmethod 
    def setUpClass(self):
        self.input_file = os.path.join(BASE, "cgmpe_test.toml")
        self.output_directory = os.path.join(BASE, 'cgmpe_test')
        self.exp_cgmpe = os.path.join(BASE, "exp_cgmpe.csv")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
    
    def test_cgmpe_from_toml(self):
        """
        Check GMPEs modified using cgmpe features specified within the toml
        are executed correctly and the expected values are returned
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        # Get matrices of predicted ground-motions per GMM
        obs_matrix = compute_matrix_gmpes(config, mtxs_type='median')
        del obs_matrix['gmpe_list']

        # Load the matrices of expected ground-motions per GMM         
        if not os.path.exists(self.exp_cgmpe):
            # Write if doesn't exist
            df = matrix_to_df(obs_matrix)
            df.to_csv(self.exp_cgmpe)
        exp_df = pd.read_csv(self.exp_cgmpe, index_col=0)

        # Load obs into dataframe
        obs_df = matrix_to_df(obs_matrix)

        # Now check matrix dfs
        pd.testing.assert_frame_equal(obs_df, exp_df, atol=1e-06)
        
        # Also, check the baseline ratio with mgmpe plotting works
        comp.plot_ratios(self.input_file, self.output_directory)
        
    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.output_directory)