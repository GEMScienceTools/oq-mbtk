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
Tests parsing of the GEM globally homogenised flatfile using the parser
"""
import os
import shutil
import unittest
import pickle

from openquake.smt.residuals import gmpe_residuals as res
from openquake.smt.residuals.parsers.gem_flatfile_parser import GEMFlatfileParser


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Defines the record IDs for the target data set
TARGET_IDS = [
"EQ_EMSC_20161026_0000077_3A_MZ01_ESM",
"EQ_HelenaMontana_01_USGS_CarrollCollege_NGAWest2",
"EQ_32_MARN_0_NGASUB",
"EQ_198000000000_TK_2001_Turkiye_SMD",
"EQ_2017_12_31_071100_kiknet_OITH11_kiknet"]


class GEMFlatfileParserTestCase(unittest.TestCase):
    """
    Tests the parsing of the GEM global flatfile
    """
    @classmethod
    def setUpClass(cls):
        cls.GEM_flatfile_directory = os.path.join(
            BASE_DATA_PATH, "GEM_flatfile_test.csv")
        cls.db_file = os.path.join(
            BASE_DATA_PATH, "GEM_conversion_test_metadata")       
        cls.gmpe_list = ["AkkarEtAlRjb2014", "ChiouYoungs2014"]
        cls.imts = ["PGA", "SA(1.0)"]
        cls.metadata_pth = os.path.join(cls.db_file, "metadatafile.pkl")

    def test_gem_flatfile_parser(self):
        parser = GEMFlatfileParser.autobuild("000", "GEM_conversion_test",
                                             self.db_file,
                                             self.GEM_flatfile_directory)
        with open(self.metadata_pth, "rb") as f:
            db = pickle.load(f)
        
        # Should contain 5 records
        self.assertEqual(len(db), 5)
        
        # Record IDs should be equal to the specified target IDs
        self.assertListEqual([rec.id for rec in db], TARGET_IDS)

        # Also run an arbitrary residual analysis to check
        # the constructed db is functioning correctly
        residuals = res.Residuals(self.gmpe_list, self.imts)
        residuals.compute_residuals(db, component="rotD50")

        del parser

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.db_file)