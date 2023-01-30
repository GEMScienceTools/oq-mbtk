# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
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
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
Testing methods and functions in the ISF catalogue module
"""

import os
import unittest
import datetime as dt
import numpy as np

from openquake.cat.parsers.converters import GenericCataloguetoISFParser
from openquake.cat.parsers.isf_catalogue_reader import ISFReader
from openquake.cat.isf_catalogue import get_threshold_matrices

BASE_DATA_PATH = os.path.dirname(__file__)


class MergeGenericCatalogueTest(unittest.TestCase):

    def setUp(self):
        self.fname_isf = os.path.join(BASE_DATA_PATH, 'data', 'cat01.isf')
        self.fname_isf2 = os.path.join(BASE_DATA_PATH, 'data', 'cat02.isf')

        self.fname_csv = os.path.join(BASE_DATA_PATH, 'data', 'cat01.csv')
        self.fname_csv2 = os.path.join(BASE_DATA_PATH, 'data', 'cat02.csv')
        self.fname_csv3 = os.path.join(BASE_DATA_PATH, 'data', 'cat03.csv')

        self.fname_idf4 = os.path.join(BASE_DATA_PATH, 'data', 'cat04.isf')
        self.fname_csv4 = os.path.join(BASE_DATA_PATH, 'data', 'cat04.csv')

    def test_case01(self):
        """Merging .csv formatted catalogue"""

        # Read the ISF formatted file
        parser = GenericCataloguetoISFParser(self.fname_csv)
        _ = parser.parse("tcsv", "Test CSV")
        catcsv = parser.export("tcsv", "Test CSV")
        parser = ISFReader(self.fname_isf)
        catisf = parser.read_file("tisf", "Test CSV")

        # Read the ISF formatted file
        catisf._create_spatial_index()
        delta = 10

        # Merging the catalogue
        tz = dt.timezone(dt.timedelta(hours=8))
        out, doubts = catisf.add_external_idf_formatted_catalogue(
                catcsv, ll_deltas=0.05, delta_t=delta, utc_time_zone=tz)

        # Testing output
        msg = 'The number of colocated events is wrong'
        self.assertEqual(1, len(out), msg)
        msg = 'The number of events in the catalogue is wrong'
        self.assertEqual(3, len(catisf.events), msg)

    def test_case02(self):
        """Merging .csv formatted catalogue with time dependent thresholds"""
        # We expect the following:
        # - The 1st event in the .csv catalogue will be removed (duplicate)
        # - The 2nd event in the .csv catalogue will be kept (distance)
        # - The 3rd event in the .csv catalogue will be kept (time)
        # In total the merged catalogue must contain 6 events

        # Read the CSV formatted file
        parser = GenericCataloguetoISFParser(self.fname_csv2)
        _ = parser.parse("tcsv", "Test CSV")
        catcsv = parser.export("tcsv", "Test CSV")

        # Read the ISF formatted file
        parser = ISFReader(self.fname_isf2)
        catisf = parser.read_file("tisf", "Test CSV")

        # Set the deltas
        catisf._create_spatial_index()
        delta1 = 10.0
        delta2 = 5.0

        # Merging the catalogue
        tz = dt.timezone(dt.timedelta(hours=0))
        ll_delta = np.array([[1899, 0.1], [1950, 0.05]])
        delta = [[1899, delta1], [1950, delta2]]
        out, doubtss = catisf.add_external_idf_formatted_catalogue(
                catcsv, ll_deltas=ll_delta, delta_t=delta, utc_time_zone=tz)

        # Testing output
        msg = 'The number of colocated events is wrong'
        self.assertEqual(2, len(out), msg)
        msg = 'The number of events in the catalogue is wrong'
        self.assertEqual(6, len(catisf.events), msg)

    def test_case03(self):
        """Testing the identification of doubtful events"""
        # In this test the first event in the .csv file is a duplicate of
        # the 2015 earthquake and it is therefore excluded.
        # The second and third events in the .csv catalogue are outside of the
        # selection windows hence are added to the cataloue as new events.
        # These two events are also within the selection buffer hence they are
        # signalled as the doubtful events.

        # Read the CSV formatted file
        parser = GenericCataloguetoISFParser(self.fname_csv3)
        _ = parser.parse("tcsv", "Test CSV")
        catcsv = parser.export("tcsv", "Test CSV")

        # Read the ISF formatted file
        parser = ISFReader(self.fname_isf2)
        catisf = parser.read_file("tisf", "Test CSV")

        # Create the spatial index
        catisf._create_spatial_index()
        buff_t = 2.0

        # Merg the .csv catalogue into the isf one
        tz = dt.timezone(dt.timedelta(hours=0))
        ll_delta = np.array([[1899, 0.1], [1950, 0.05]])
        delta = [[1899, 10.0], [1950, 5.0]]
        out, doubts = catisf.add_external_idf_formatted_catalogue(
                catcsv, ll_deltas=ll_delta, delta_t=delta, utc_time_zone=tz,
                buff_t=buff_t, buff_ll=0.02)

        # Testing output
        msg = 'The number of colocated events is wrong'
        self.assertEqual(1, len(out), msg)
        msg = 'The number of events in the catalogue is wrong'
        self.assertEqual(5, len(catisf.events), msg)

        # Check doubtful earthquakes
        msg = 'The information about doubtful earthquakes is wrong'
        self.assertEqual([1, 2], doubts[2], msg)
        self.assertEqual(1, len(doubts), msg)

    def test_case04(self):
        """Merge ISC-GEM not identified through search"""

        # Create an ins
        parser = ISFReader(self.fname_idf4)
        cat = parser.read_file("ISC_DB1", "Test ISF")

        parser = GenericCataloguetoISFParser(self.fname_csv4)
        cat_iscgem = parser.parse("ISCGEM", "ISC-GEM")

        delta = 30.0
        timezone = dt.timezone(dt.timedelta(hours=0))

        cat._create_spatial_index()
        with self.assertWarns(UserWarning) as cm:
            _ = cat.add_external_idf_formatted_catalogue(
                cat_iscgem, ll_deltas=0.40, delta_t=delta,
                utc_time_zone=timezone, buff_t=dt.timedelta(0), buff_ll=0,
                use_ids=True, logfle=None)
        self.assertIn('isf_catalogue.py', cm.filename)
        self.assertEqual(835, cm.lineno)


class GetThresholdMatricesTest(unittest.TestCase):

    def test_gmtx01(self):
        """ Simple case with scalars """

        delta_t = 30
        delta_ll = 0.2
        mage, timee, time_d, ll_d = get_threshold_matrices(delta_t, delta_ll)

        computed = time_d[0, 0].total_seconds()
        np.testing.assert_almost_equal(computed, delta_t)
        np.testing.assert_almost_equal(ll_d[0, 0], delta_ll)

    def test_gmtx02(self):
        """ Case with list of scalars """

        delta_t = [[1900, 30.0], [1960, 20.0], [1980, 20.0]]
        delta_ll = [[1900, 0.3], [1960, 0.2], [1980, 0.2]]
        mage, timee, time_d, ll_d = get_threshold_matrices(delta_t, delta_ll)

        expected = np.array([t[1] for t in delta_t])
        computed = np.array([t[0].total_seconds() for t in time_d])
        np.testing.assert_almost_equal(computed, expected)

        expected = np.ones((3, 40))
        expected[0, :] = 0.3
        expected[1:, :] = 0.2
        computed = ll_d
        np.testing.assert_almost_equal(computed, expected)

    def test_gmtx03(self):
        """ Case with a list of functions """

        delta_t = [[1900, '5*m'], [1960, '2.5*m']]
        delta_ll = [[1900, '0.1*m'], [1960, '0.05*m']]
        mage, timee, time_d, ll_d = get_threshold_matrices(delta_t, delta_ll)

        mags = np.arange(1.0, 9.0, 0.2)
        computed = np.array([t.total_seconds() for t in time_d[0]])
        expected = 5.0 * mags
        np.testing.assert_almost_equal(computed, expected)

        computed = np.array([t.total_seconds() for t in time_d[1]])
        expected = 2.5 * mags
        np.testing.assert_almost_equal(computed, expected)

        expected = 0.1 * mags
        np.testing.assert_almost_equal(ll_d[0], expected)

        expected = 0.05 * mags
        np.testing.assert_almost_equal(ll_d[1], expected)
