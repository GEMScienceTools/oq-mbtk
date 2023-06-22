### wkf adaptive smoothing tests

import tempfile
import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser


HERE = os.path.dirname(__file__)
#CWD = os.getcwd()
#DATA = os.path.relpath(os.path.join(HERE, 'data', 'rates_distribute'), CWD)

DATA_PATH = os.path.relpath(os.path.join(HERE, 'data'))

class AdaptiveSmoothingTestwkf(unittest.TestCase):

    def setUp(self):
        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        self.fname = fname
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        self.cat = cat
    
    def test_adap_smoothing_wkf(self):
        """ Test adaptive smoothing build """
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        fname_h3 = os.path.join(DATA_PATH, 'zones_h3', 'mapping_h2.csv')
        fname_out = os.path.join(DATA_PATH, 'smooth_adap.csv')

        # Run the code
        cmd = f"oqm wkf wkf_adaptive_smoothing {self.fname} {fname_h3} {config} {fname_out}"
        p = subprocess.run(cmd, shell=True)
        
        expected = np.array([1.76509281e-03, 2.24894579e-03, 1.37445308e-03, 9.81211408e-04,
       2.59893514e-03, 1.85124547e-03, 2.32545463e-03, 9.83247419e-04,
       8.80487564e-04, 4.30942107e-05, 9.73681917e-05, 3.91305325e-06,
       2.90803858e-03, 9.19557721e-04, 2.67332510e-03, 1.29838234e-03])

        # set up an expected
        computed = pd.read_csv(fname_out)
        np.testing.assert_almost_equal(expected, computed['nocc'], decimal=4)
