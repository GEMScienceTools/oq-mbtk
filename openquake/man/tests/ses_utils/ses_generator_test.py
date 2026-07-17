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

import numpy as np
import json
import os
import shutil
import tempfile
import unittest

from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source.rupture import EBRupture
from openquake.man.ses_utils.ses_generator import prepare_source_for_sampling, ses_from_area_source
from openquake.man.ses_utils.ses_source import get_area_source


"""
Tests for the functions in ses_generator.py.
"""

class TestSESGenerator(unittest.TestCase):
    """
    Integration test for the ses_from_area_source function. Runs a simulation 
    by spinning up real OpenQuake objects and reading from a temporary 
    GeoJSON file.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.poly_file = os.path.join(self.test_dir, "test_area_source.geojson")

        polygon_geojson = {
            "type": "FeatureCollection",
            "name": "test_area",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [10.0, 40.0],
                                [10.0, 45.0],
                                [15.0, 45.0],
                                [15.0, 40.0],
                                [10.0, 40.0],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        }

        with open(self.poly_file, "w", encoding="utf-8") as file:
            json.dump(polygon_geojson, file)

        self.mfd = TruncatedGRMFD(min_mag=4.0, max_mag=6.5, bin_width=0.1, a_val=4.5, b_val=1.0)
        self.hdd = PMF([(0.3, 5.0), (0.7, 10.0)])

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_prepare_source_for_sampling(self):
        """ Tests source objects receive metadata required by BaseSeismicSource.sample_ruptures(). """
        src = get_area_source(self.mfd, polygon_fname=self.poly_file, hdd=self.hdd)
        self.assertIsNone(src.sampling)
        prepared_src = prepare_source_for_sampling(src)
        self.assertIs(prepared_src, src)

        self.assertEqual(src.id, 0)
        self.assertEqual(src.offset, 0)
        self.assertEqual(src.smweight, 1.0)

        self.assertIsInstance(src.sampling, dict)
        self.assertIn("samples", src.sampling)
        self.assertIn("trt_smr", src.sampling)

        np.testing.assert_array_equal(src.sampling["samples"], np.array([1], dtype=np.uint32))
        np.testing.assert_array_equal(src.sampling["trt_smr"], np.array([0], dtype=np.uint32))

        self.assertEqual(src.sampling["samples"].dtype, np.dtype(np.uint32))
        self.assertEqual(src.sampling["trt_smr"].dtype, np.dtype(np.uint32))

    def test_ses_from_area_source_full_simulation(self):
        """ Test that the complete SES simulation pipeline executes successfully. """
        ses = ses_from_area_source(self.poly_file, self.mfd, self.hdd)

        self.assertIsInstance(ses, list)
        self.assertGreater(len(ses), 0, "ses_from_area_source returned an empty SES.")

        for event in ses:
            self.assertIsInstance(event, EBRupture)

        mags = [event.mag for event in ses]

        self.assertTrue(all(4.0 <= mag <= 6.5 for mag in mags), "At least one simulated magnitude is outside the MFD bounds.")
