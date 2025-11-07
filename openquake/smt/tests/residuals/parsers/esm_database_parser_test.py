import os
import unittest

from openquake.smt.residuals.parsers.esm_database_parser import (ESMDatabaseParser,
                                                                 ESMTimeSeriesParser,
                                                                 ESMSpectraParser)


BASE = os.path.join(os.path.dirname(__file__), "data")

EXP_ACCELERATION_NSAMPLES = [21000, 21000, 17000, 21000, 21000] # Nsamples of x-component per record
EXP_NRECS = 5 # Number of 3-component time histories


class ESMDatabaseParserTest(unittest.TestCase):
    """
    Test that metadata is parsed correctly when using the
    ESM database parser.
    """
    @classmethod
    def setUpClass(cls):
        records = os.path.join(BASE, "esm_records")
        instance = ESMDatabaseParser(db_id='1', db_name='db', input_files=records)
        cls.database = instance.parse() # Parse the metadata of each record
        del instance

    def test_nrecords(self):
        self.assertEqual(len(self.database.records), EXP_NRECS)
        
    def test_time_series_parsing(self):
        for idx_rec, rec in enumerate(self.database.records):
            ts = ESMTimeSeriesParser(rec.time_series_file).parse_record()
            self.assertEqual(EXP_ACCELERATION_NSAMPLES[idx_rec],
                             ts["X"]["Original"]["Acceleration"].shape[0])
            
    #def test_spectra_parsing(self):
     #   for idx_rec, rec in enumerate(self.database.records):
      #      sp = ESMSpectraParser(rec.spectra_file).parse_spectra()