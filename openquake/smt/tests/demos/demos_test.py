# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2014 GEM Foundation
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
Tests for execution of demos
"""
import os
import shutil
import unittest

from openquake.smt.demos.demo_comparison import main as comp_demo
from openquake.smt.demos.demo_residuals import main as res_demo
from openquake.smt.demos.demo_single_station_analysis import main as st_demo

# Base path
base = os.path.dirname(__file__)

# Input files
comparison_in = os.path.join(
    base, '..', '..', 'demos', 'demo_input_files',
    'demo_comparison_analysis_inputs.toml')
residuals_flatfile_in = os.path.join(
    base, '..', '..', 'demos', 'demo_input_files',
    'demo_flatfile.csv')
residuals_input_toml = os.path.join(
    base, '..', '..', 'demos', 'demo_input_files',
    'demo_residual_analysis_inputs.toml')
residuals_hrz_comp = "Geometric"

# Output paths
comparison_out = os.path.join(base, 'outputs_demo_comparison')
residuals_out = os.path.join(base, 'outputs_demo_residual_analysis')
stations_out = os.path.join(base, 'outputs_demo_station_analysis')


@unittest.skip # Check locally only (running on remote takes a while)
class DemosTestCase(unittest.TestCase):
    """
    Core test case for the SMT demos
    """
    @classmethod
    def setUpClass(self):
        # Demo scripts
        self.comparison_in = comparison_in
        self.residuals_flatfile_in = residuals_flatfile_in
        self.residuals_input_toml = residuals_input_toml
        self.residuals_hrz_comp = residuals_hrz_comp

        # Demo output locations
        self.comparison_out = comparison_out
        self.residuals_out = residuals_out
        self.stations_out = stations_out

    def test_comparison_demo(self):
        """
        Execute the comparison demo
        """
        comp_demo(
            input_toml=self.comparison_in, out_dir=self.comparison_out)

    def test_residuals_demo(self):
        """
        Execute the residual analysis demo
        """
        res_demo(flatfile=self.residuals_flatfile_in,
                 gmms_imts=self.residuals_input_toml,
                 comp=self.residuals_hrz_comp,
                 out_dir=self.residuals_out)
        
    def test_stations_demo(self):
        """
        Execute the single station residual analysis demo
        """
        st_demo(flatfile=self.residuals_flatfile_in,
                 gmms_imts=self.residuals_input_toml,
                 out_dir=self.stations_out,
                 threshold=45)
        
    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.comparison_out)
        shutil.rmtree(self.residuals_out)
        shutil.rmtree(self.stations_out)