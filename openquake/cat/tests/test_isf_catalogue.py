import os
import unittest
import datetime as dt

from openquake.cat.parsers.converters import GenericCataloguetoISFParser
from openquake.cat.parsers.isf_catalogue_reader import ISFReader

BASE_DATA_PATH = os.path.dirname(__file__)


class MergeGenericCatalogueTest(unittest.TestCase):

    def setUp(self):
        self.fname_isf = os.path.join(BASE_DATA_PATH, 'data', 'cat01.isf')
        self.fname_csv = os.path.join(BASE_DATA_PATH, 'data', 'cat01.csv')

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
                                                          ll_delta=0.05,
                                                          time_delta=delta,
                                                          utc_time_zone=tz)
        #
        # Testing output
        msg = 'The number of colocated events is wrong'
        self.assertEqual(1, len(out), msg)
        msg = 'The number of events in the catalogue is wrong'
        self.assertEqual(3, len(catisf.events), msg)
