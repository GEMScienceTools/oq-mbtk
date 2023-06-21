### wkf smoothing tests

import tempfile
import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
#import openquake.wkf.wkf_h3_zones_cat


HERE = os.path.dirname(__file__)
CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#CWD = os.getcwd()
#DATA = os.path.relpath(os.path.join(HERE, 'data', 'rates_distribute'), CWD)

DATA_PATH = os.path.relpath(os.path.join(HERE, 'data'))

class test_smoothing_wkf(unittest.TestCase):

    def setUp(self):
        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        self.fname = fname
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        self.cat = cat
        print(HERE)
        
    def test_discretize_zones(self):
        """ test function for generating h3 cells per zone """
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        zones_h3_repr = os.path.join(DATA_PATH, 'zones_h3')
        polys = os.path.join(DATA_PATH, 'smooth_poly_test.geojson')
        h3_level = 2
        cmd = f"oqm wkf set_h3_to_zones {h3_level} {polys} {zones_h3_repr}"
        out = subprocess.call(cmd, shell=True)

        h3_idx = pd.read_csv(os.path.join(DATA_PATH, 'zones_h3', 'mapping_h2.csv'), header = None)
        self.assertEqual(len(h3_idx), 17)
        self.assertEqual(sum(h3_idx[1] == 1), 10)
        
    def test_box_counting(self):
        """ Run boxcounting on test data """
        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        h3_level = 2
        zones_h3_repr = os.path.join(DATA_PATH, 'zones_h3')
        fname_bcounting = os.path.join(DATA_PATH, 'box_counting')
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        #cmd = f"oqm wkf wkf_boxcounting_h3 {os.path.join(DATA_PATH, 'smooth_test.csv')} {zones_h3_repr} {config}"
        #cmd = f"{cmd} {h3_level} {fname_bcounting} -y 2018 -w \"one\""

        code = os.path.join(CODE, 'bin', 'wkf_boxcounting_h3.jl')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, fname, zones_h3_repr, config)
        #, h3_level, fname_bcounting, -y, 2018, -w,"one")
        cmd = f"{cmd} {h3_level} {fname_bcounting} -y 2018 -w \"one\""

        p = subprocess.run(cmd, shell=True)
        
        bc = pd.read_csv(os.path.join(DATA_PATH, 'box_counting', 'box_counting_h3_smooth_test.csv'))
        exp = np.array([1, 1, 1, 1, 1])
        np.testing.assert_equal(bc['count'], exp )
        #np.testing.assert_almost_equal(bc['h3idx'], (5.86504791023157E+017, 5.87132612162617E+017, 5.87008917104493E+017, 5.86991324918448E+017, 5.86991324918448E+017))
        #out = subprocess.call(cmd, shell=True)
        
    def test_smoothing_wkf(self):
        """ Test adaptive smoothing build """
        
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        fname_bcounting = os.path.join(DATA_PATH, 'box_counting', 'box_counting_h3_smooth_test.csv')
        fname_out = os.path.join(HERE, 'smooth.csv')
        
        # Run the code
        code = os.path.join(CODE, 'bin', 'wkf_smoothing.jl')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, fname_bcounting, config, fname_out)
        #subprocess.call(cmd, shell=True)
        
        out = subprocess.call(cmd, shell=True)
        # Test results
        
        expected = np.array([0.00400579, 0.00200289, 0.00400579, 0.00200289, 0.00200289,
       0.98998553, 0.00200289, 0.00400579, 0.00200289, 0.00200289,
       0.98998553, 0.00600868, 0.00200289, 0.00200289, 0.00200289,
       0.00200289, 0.00200289, 0.00200289, 0.98798263, 0.00200289,
       0.98998553, 0.00200289, 0.00200289, 0.00400579, 0.98998553])
        #expected = numpy.array([3.671158, 3.97219, 4.14828, 4.27322])
        # set up an expected....
        computed = pd.read_csv(os.path.join(HERE, 'smooth.csv'))
        np.testing.assert_almost_equal(expected, computed['nocc'], decimal=4)
