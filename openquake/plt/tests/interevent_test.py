# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation and Électricité de France
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
#
# This script is produced within the scope of Work Package 5, named Simulation 
# platform, under SIGMA3 project. For more detailed information about 
# the project, please visit to https://sigma-programs.com/.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from scipy.stats import expon
from unittest.mock import patch

from openquake.plt.interevent import get_aic, analyze_interevent_times

"""
Testing methods and functions in the interevent.py
"""

class TestTimeIntervals(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and data for testing."""
        self.test_dir = tempfile.mkdtemp()
        
        self.cat_path = os.path.join(self.test_dir, "test_cat.csv")
        self.output_png = os.path.join(self.test_dir, "test_output.png")
        
        data = {
            'year': [2015, 2021, 2023, 2025, 2025],
            'month': [1, 7, 12, 9, 12],
            'day': [1, 15, 10, 20, 5],
            'magnitude': [4.5, 5.0, 3.8, 6.2, 7.7]
        }

        pd.DataFrame(data).to_csv(self.cat_path, index=False)

    def tearDown(self):
        """Clean up generated files if they exist."""
        shutil.rmtree(self.test_dir)

    def test_get_aic(self):
        """Test the AIC calculation helper function directly."""
        data = np.array([0.5, 1.2, 2.3, 0.8, 1.5])
        params = expon.fit(data)
        
        aic_score = get_aic(expon, params, data)
        self.assertIsInstance(aic_score, float)
        self.assertFalse(np.isnan(aic_score))

    @patch('openquake.plt.interevent.plt.savefig')
    @patch('openquake.plt.interevent.plt.show')
    def test_analyze_interevent_times_log(self, mock_show, mock_savefig):
        """Test the main plotting function running with logarithmic scale."""
        analyze_interevent_times(
            catalogue_path=self.cat_path,
            bin_scale="logarithmic",
            num_bins=5,
            output_png=self.output_png
        )
        mock_savefig.assert_called_once()

    @patch('openquake.plt.interevent.plt.savefig')
    @patch('openquake.plt.interevent.plt.show')
    def test_analyze_interevent_times_lin(self, mock_show, mock_savefig):
        """Test the main plotting function running with linear scale."""
        analyze_interevent_times(
            catalogue_path=self.cat_path,
            bin_scale="linear",
            num_bins=5,
            output_png=self.output_png
        )
        mock_savefig.assert_called_once()

    def test_invalid_file_path_raises_error(self):
        """Verify that a missing catalogue path correctly raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            analyze_interevent_times(catalogue_path="no_file.csv")

    def test_invalid_bin_scale_raises_error(self):
        """Verify that passing an unsupported scale option raises a ValueError."""
        with self.assertRaises(ValueError):
            analyze_interevent_times(
                catalogue_path=self.cat_path,
                bin_scale="invalid_one"
            )