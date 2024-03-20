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
Tests parsing of the GEM globally homogenised flatfile using the parser
"""
import os
import sys
import shutil
import unittest
from openquake.smt.parsers.gem_flatfile_parser import GEMFlatfileParser

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

# Defines the record IDs for the target data set
TARGET_IDS = [
"EQ_EMSC_20161026_0000077_3A_MZ01_ESM_",
"EQ_HelenaMontana_01_USGS_CarrollCollege_NGAWest2_",
"EQ_32_MARN_0_NGASUB_",
"EQ_1976_08_19_01_12_39_TK_2001_Turkiye_SMD_",
"EQ_2017_12_31_071100_kiknet_OITH11_kiknet_"]

#Specify base directory
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

class GEMFlatfileParserTestCase(unittest.TestCase):
    """
    Tests the parsing of the GEM global flatfile
    """
    @classmethod
    def setUpClass(cls):
        cls.GEM_flatfile_directory = os.path.join(BASE_DATA_PATH,
                                                  "GEM_flatfile_test.csv")
        cls.db_file = os.path.join(BASE_DATA_PATH,
                                   "ESM_conversion_test_metadata")       

    def test_gem_flatfile_parser(self):
        """
        Tests the parsing of the GEM flatfile. 
        
        Checks the proxy will give the KiKNet record the geometric mean of the
        horizontal components as a proxy for the missing RotD50 acc values beyond
        5 s + the removal option will then not discard this record as RotD50 is
        now 'complete' for all required spectral periods
        """
        parser = GEMFlatfileParser.autobuild("000", "GEM_conversion_test",
                                             self.db_file, self.GEM_flatfile_directory,
                                             removal=True, proxy=True)
        with open(os.path.join(self.db_file, "metadatafile.pkl"), "rb") as f:
            db = pickle.load(f)
        # Should contain 5 records
        self.assertEqual(len(db), 5)
        # Record IDs should be equal to the specified target IDs
        for rec in db:
            print(rec.id)
        self.assertListEqual([rec.id for rec in db], TARGET_IDS)
        del parser

    @classmethod
    def tearDownClass(cls):
        """
        Remove the database
        """
        shutil.rmtree(cls.db_file)