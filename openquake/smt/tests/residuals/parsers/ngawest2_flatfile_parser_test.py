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
Tests parsing of the NGAWest2 flatfile format in SMT
"""
import os
import shutil
import unittest
import pickle

from openquake.smt.residuals import gmpe_residuals as res
from openquake.smt.residuals.parsers.ngawest2_flatfile_parser import NGAWest2FlatfileParser


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Defines the record IDs for the target data set
TARGET_IDS = [
'Earthquake_HelenaMontana_01_NetworkCode_USGS_StationName_CarrollCollege',
'Earthquake_HelenaMontana_02_NetworkCode_USGS_StationName_HelenaFedBldg',
'Earthquake_HumboltBay_NetworkCode_USGS_StationName_FerndaleCityHall',
'Earthquake_ImperialValley_01_NetworkCode_USGS_StationName_ElCentroArray#9',
'Earthquake_NorthwestCalif_01_NetworkCode_USGS_StationName_FerndaleCityHall',
'Earthquake_ImperialValley_02_NetworkCode_USGS_StationName_ElCentroArray#9',
'Earthquake_NorthwestCalif_02_NetworkCode_USGS_StationName_FerndaleCityHall',
'Earthquake_NorthernCalif_01_NetworkCode_USGS_StationName_FerndaleCityHall',
'Earthquake_Borrego_NetworkCode_USGS_StationName_ElCentroArray#9',
'Earthquake_ImperialValley_03_NetworkCode_USGS_StationName_ElCentroArray#9']


class NGAWest2FlatfileParserTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.NGAWest2_flatfile_directory = os.path.join(
            BASE_DATA_PATH, "NGAWest2_test.csv")
        cls.NGAWest2_vertical_flatfile_directory = os.path.join(
            BASE_DATA_PATH,"NGAWest2_vertical_test.csv")
        cls.db_file = os.path.join(BASE_DATA_PATH, "NGAWest2_test_metadata")       
        cls.gmpe_list = ["AkkarEtAlRjb2014", "ChiouYoungs2014"]
        cls.imts = ["PGA", "SA(1.0)"]

    def test_NGAWest2_flatfile_parser(self):
        parser = NGAWest2FlatfileParser.autobuild("000", "NGAWest2_test",
                                             self.db_file,
                                             self.NGAWest2_flatfile_directory,
                                             self.NGAWest2_vertical_flatfile_directory)
        with open(os.path.join(self.db_file, "metadatafile.pkl"), "rb") as f:
            db = pickle.load(f)

        # Should contain 10 records
        self.assertEqual(len(db), 10)
        
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