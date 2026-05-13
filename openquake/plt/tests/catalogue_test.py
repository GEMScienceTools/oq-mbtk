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
# the project, please visit to <https://sigma-programs.com/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
Testing methods and functions in the catalogue.py
"""

import os
import unittest
import pandas as pd
import json
import tempfile
import shutil

from catalogue import prepare_geometry, split_catalogue_dynamic


class TestquakeTGMTMapping(unittest.TestCase):

    def setUp(self):
        # Create a temporary working directory
        self.test_dir = tempfile.mkdtemp()
        
        # Sample catalogue for test
        self.cat_path = os.path.join(self.test_dir, "test_cat.csv")
        data = {
            'longitude': [10.0, 11.0, 12.0, 13.0],
            'latitude': [40.0, 41.0, 42.0, 43.0],
            'magnitude': [4.0, 5.0, 6.0, 7.0]
        }
        pd.DataFrame(data).to_csv(self.cat_path, index=False)
        
        # Sample GeoJSON for test
        self.geojson_path = os.path.join(self.test_dir, "test_poly.geojson")
        poly = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 40], [15, 40], [15, 45], [10, 45], [10, 40]]]
                }
            }]
        }
        with open(self.geojson_path, "w") as f:
            json.dump(poly, f)
            
        # Mag bin samples for test
        self.custom_bins = [
            {"min_mag": 3.5, "max_mag": 4.5, "color": "blue", "size": "0.08c"},
            {"min_mag": 4.5, "max_mag": 5.5, "color": "green", "size": "0.12c"},
            {"min_mag": 5.5, "max_mag": 6.5, "color": "orange", "size": "0.18c"},
            {"min_mag": 6.5, "max_mag": None, "color": "red", "size": "0.25c"}
        ]

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        for i in range(1, len(self.custom_bins) + 1):
            temp_f = f"m{i}.xyz"
            if os.path.exists(temp_f):
                os.remove(temp_f)
        if os.path.exists("polygon.xy"):
            os.remove("polygon.xy")

    def test_prepare_geometry(self):
        """Test if GeoJSON is correctly converted to GMT XY format"""
        output = prepare_geometry(self.geojson_path, output_xy="polygon.xy")
        self.assertTrue(os.path.exists(output))
        with open(output, 'r') as f:
            lines = f.readlines()
            self.assertIn(">\n", lines)  # GMT segment header check

    def test_split_catalogue_dynamic(self):
        """Test if catalogue is correctly binned into dynamic XYZ files"""
        df = pd.read_csv(self.cat_path)
        temp_files = split_catalogue_dynamic(df, self.custom_bins)
        
        self.assertEqual(len(temp_files), 4)
        
        # # M7.0 should be in m4.xyz
        m4_path = temp_files[3] # 6.5 <= M case
        m4_df = pd.read_csv(m4_path, sep=" ", header=None)
        self.assertEqual(len(m4_df), 1)
