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
        self.fname_csv = os.path.join(BASE_DATA_PATH, 'data', 'cat01.csv')

        self.fname_isf2 = os.path.join(BASE_DATA_PATH, 'data', 'cat02.isf')
        self.fname_csv2 = os.path.join(BASE_DATA_PATH, 'data', 'cat02.csv')

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
        out = catisf.add_external_idf_formatted_catalogue(catcsv,
                                                          ll_deltas=0.05,
                                                          time_delta=delta,
                                                          utc_time_zone=tz)
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
        out = catisf.add_external_idf_formatted_catalogue(catcsv,
                                                          ll_deltas=ll_delta,
                                                          time_delta=delta,
                                                          utc_time_zone=tz)
        #
        # Testing output
        msg = 'The number of colocated events is wrong'
        self.assertEqual(2, len(out), msg)
        msg = 'The number of events in the catalogue is wrong'
        self.assertEqual(6, len(catisf.events), msg)
