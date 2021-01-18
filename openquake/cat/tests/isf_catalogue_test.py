import os
import unittest
import numpy as np
import datetime as dt

from openquake.cat.parsers.converters import GenericCataloguetoISFParser
from openquake.cat.parsers.isf_catalogue_reader import ISFReader

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
        #
        # Read the ISF formatted file
        parser = GenericCataloguetoISFParser(self.fname_csv)
        _ = parser.parse("tcsv", "Test CSV")
        catcsv = parser.export("tcsv", "Test CSV")
        parser = ISFReader(self.fname_isf)
        catisf = parser.read_file("tisf", "Test CSV")
        #
        # Read the ISF formatted file
        catisf._create_spatial_index()
        delta = dt.timedelta(seconds=10)
        #
        # Merging the catalogue
        tz = dt.timezone(dt.timedelta(hours=8))
        out, doubts = catisf.add_external_idf_formatted_catalogue(
                catcsv, ll_deltas=0.05, delta_t=delta, utc_time_zone=tz)
        #
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

        #
        # Read the ISF formatted file
        parser = GenericCataloguetoISFParser(self.fname_csv2)
        _ = parser.parse("tcsv", "Test CSV")
        catcsv = parser.export("tcsv", "Test CSV")
        parser = ISFReader(self.fname_isf2)
        catisf = parser.read_file("tisf", "Test CSV")
        #
        # Read the ISF formatted file
        catisf._create_spatial_index()
        delta1 = dt.timedelta(seconds=10.)
        delta2 = dt.timedelta(seconds=5.)
        #
        # Merging the catalogue
        tz = dt.timezone(dt.timedelta(hours=0))
        ll_delta = np.array([[1899, 0.1], [1950, 0.05]])
        delta = [[1899, delta1], [1950, delta2]]
        out, doubtss = catisf.add_external_idf_formatted_catalogue(
                catcsv, ll_deltas=ll_delta, delta_t=delta, utc_time_zone=tz)
        #
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

        #
        # Read the ISF formatted file
        parser = GenericCataloguetoISFParser(self.fname_csv3)
        _ = parser.parse("tcsv", "Test CSV")
        catcsv = parser.export("tcsv", "Test CSV")
        parser = ISFReader(self.fname_isf2)
        catisf = parser.read_file("tisf", "Test CSV")
        #
        # Read the ISF formatted file
        catisf._create_spatial_index()
        delta1 = dt.timedelta(seconds=10.)
        delta2 = dt.timedelta(seconds=5.)
        buff_t = dt.timedelta(seconds=2.)
        #
        # Merging the catalogue
        tz = dt.timezone(dt.timedelta(hours=0))
        ll_delta = np.array([[1899, 0.1], [1950, 0.05]])
        delta = [[1899, delta1], [1950, delta2]]
        out, doubts = catisf.add_external_idf_formatted_catalogue(
                catcsv, ll_deltas=ll_delta, delta_t=delta, utc_time_zone=tz,
                buff_t=buff_t, buff_ll=0.02)
        #
        # Testing output
        msg = 'The number of colocated events is wrong'
        self.assertEqual(1, len(out), msg)
        msg = 'The number of events in the catalogue is wrong'
        self.assertEqual(5, len(catisf.events), msg)
        #
        # Check doubtful earthquakes
        msg = 'The information about doubtful earthquakes is wrong'
        self.assertEqual([1, 2], doubts[2], msg)
        self.assertEqual(1, len(doubts), msg)

    def test_case04(self):
        """Merge ISC-GEM not identified through search"""

        parser = ISFReader(self.fname_idf4)
        cat = parser.read_file("ISC_DB1", "Test ISF")

        parser = GenericCataloguetoISFParser(self.fname_csv4)
        cat_iscgem = parser.parse("ISCGEM", "ISC-GEM")

        delta = dt.timedelta(seconds=30)
        timezone = dt.timezone(dt.timedelta(hours=0))

        cat._create_spatial_index()
        with self.assertWarns(UserWarning) as cm:
            _ = cat.add_external_idf_formatted_catalogue(cat_iscgem,
                        ll_deltas=0.40, delta_t=delta, utc_time_zone=timezone,
                        buff_t=dt.timedelta(0), buff_ll=0, use_ids=True,
                        logfle=None)
        self.assertIn('isf_catalogue.py', cm.filename)
        self.assertEqual(821, cm.lineno)
