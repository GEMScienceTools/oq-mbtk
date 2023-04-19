### wkf adaptive smoothing tests

import tempfile
import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
import openquake.wkf.wkf_h3_zones_cat


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
   
    ## Actually see if we can just combine this with Marco's (faster) way so it can work for both smoothing options
    def test_cat_zones(self):
        """ test catalogue zone function """
        h3_level = 2
        #mor_cat = fname
        fname = self.fname
        fname_out = os.path.join(DATA_PATH, 'h3.csv')
        cmd = f"oqm wkf wkf_h3_zones_cat {h3_level} {fname} {fname_out}"
        p = subprocess.run(cmd, shell=True)
        
        h3_idx = pd.read_csv(fname_out)
        self.assertEqual(len(h3_idx), 8)
    
    def adap_smoothing_wkf(self):
        """ Test adaptive smoothing build """
        
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        fname_h3 = os.path.join(DATA_PATH, 'h3.csv')
        fname_out = os.path.join(HERE, 'smooth_adap.csv')

        # Run the code
        code = os.path.join(HERE, '..', 'wkf_adaptive_smoothing')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, self.cat, fname_h3, config, fname_out)
        #subprocess.call(cmd, shell=True)
        
        out = subprocess.call(cmd, shell=True)
        # Test results
        assert out == 1
        
        #expected = numpy.array([3.671158, 3.97219, 4.14828, 4.27322])
        # set up an expected....
        #computed = res.agr.to_numpy()
        #numpy.testing.assert_almost_equal(expected, computed, decimal=4)
