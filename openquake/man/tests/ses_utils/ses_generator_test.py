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
import json
import unittest
import tempfile
import shutil
import numpy as np

from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source.rupture import EBRupture
from openquake.man.ses_utils.ses_generator import ses_from_area_source

"""
Testing the function in the ses_generator.py
"""

class TestSESGenerator(unittest.TestCase):
    """
    Integration test for the ses_from_area_source function. Runs a simulation 
    by spinning up real OpenQuake objects and reading from a temporary 
    GeoJSON file.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.poly_file = os.path.join(self.test_dir, 'test_area_source.geojson')
        
        poly_json = {
            "type": "FeatureCollection",
            "name": "test_area",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [4.0, 44.0], 
                        [4.0, 46.0], 
                        [6.0, 46.0], 
                        [6.0, 44.0], 
                        [4.0, 44.0]
                    ]]
                },
                "properties": {}
            }]
        }
        
        with open(self.poly_file, 'w') as f:
            json.dump(poly_json, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ses_from_area_source_full_simulation(self):
        """ Testing the complete simulation pipeline executes successfully """
        mfd = TruncatedGRMFD(4.0, 7.0, 0.1, 4.5, 1.0)
        hdd = PMF([(0.3, 5.0), (0.7, 10.0)])
        
        ses = ses_from_area_source(self.poly_file, mfd, hdd)
        self.assertIsInstance(ses, list)
        self.assertGreater(len(ses), 0)
        
        # Verify that the objects inside the list are EBRupture instances
        first_event = ses[0]
        self.assertIsInstance(first_event, EBRupture)
        
        # Data integrity check: simulated magnitudes stay within the MFD bounds [4.0, 7.0]
        magnitudes = [e.mag for e in ses]
        self.assertTrue(all(4.0 <= m <= 7.0 for m in magnitudes))
