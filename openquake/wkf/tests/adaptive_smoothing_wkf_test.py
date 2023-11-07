### wkf adaptive smoothing tests

import tempfile
import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
from pathlib import Path

HERE = os.path.dirname(__file__)
#CWD = os.getcwd()
#DATA = os.path.relpath(os.path.join(HERE, 'data', 'rates_distribute'), CWD)

DATA_PATH = os.path.relpath(os.path.join(HERE, 'data', 'adap_smooth'))

class AdaptiveSmoothingTestwkf(unittest.TestCase):

    def setUp(self):
        print(DATA_PATH)
        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        self.fname = fname
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        self.cat = cat
    
    def test_adap_smoothing_wkf(self):
        """ Test adaptive smoothing build """

        
        tmpdir = Path(tempfile.gettempdir())
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        
        folder_out = tempfile.mkdtemp(suffix='adapsmooth', prefix=None, dir=tmpdir)
        fname_out = '{}/smooth_adap.csv'.format(folder_out)

        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        fname_h3 = os.path.join(DATA_PATH, 'mapping_h2.csv')
        
        # Run the code
        cmd = f"oqm wkf wkf_adaptive_smoothing {self.fname} {fname_h3} {config} {fname_out}"
        p = subprocess.run(cmd, shell=True)

        expected = np.array([0.000984, 0.000983, 0.001360, 0.001375,0.002249,
        0.002325, 0.000881, 0.001852, 0.001765, 0.002598, 0.002911, 0.000004,
        0.000044, 0.000926, 0.000099, 0.002677, 0.001304])

        # set up an expected
        computed = pd.read_csv(fname_out)
        np.testing.assert_almost_equal(expected, computed['nocc'], decimal=4)
