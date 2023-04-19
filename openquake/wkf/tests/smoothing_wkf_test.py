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
#CWD = os.getcwd()
#DATA = os.path.relpath(os.path.join(HERE, 'data', 'rates_distribute'), CWD)

DATA_PATH = os.path.relpath(os.path.join(HERE, 'data'))

class SmoothingTestwkf(unittest.TestCase):

    def setUp(self):
        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        self.fname = fname
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        self.cat = cat
        print(HERE)
        
    def discretize_zones(self):
        zones_h3_repr = os.path.join(DATA_PATH, 'zones_h3')
        #cmd = f"oqm wkf set_h3_to_zones {h3_level} {polygons} {zones_h3_repr}"
        code = os.path.join(HERE, 'wkf', 'set_h3_to_zones')
        print(code)
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, self.cat, zones_h3_repr, config, fname_bcounting, -y, 2018, -w, "one")
        out = subprocess.call(cmd, shell=True)
        
    def box_counting(self):
        """ Run boxcounting on test data """
        h3_level = 2
        zones_h3_repr = os.path.join(DATA_PATH, 'zones_h3')
        fname_bcounting = os.path.join(DATA_PATH, 'box_counting')
        #cmd = f"wkf_boxcounting_h3.jl {declustered_cat} {zones_h3_repr} {config}"
        #cmd = f"{cmd} {h3_level} {fld_box_counting} -y 2018 -w \"one\""
        #p = subprocess.run(cmd, shell=True)
        
        # Run the code
        code = os.path.join(HERE, '..', 'wkf_boxcounting_h3.jl')
        fmt = '{:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, self.cat, zones_h3_repr, config, fname_bcounting, -y, 2018, -w, "one")
        print(cmd)
        out = subprocess.call(cmd, shell=True)
        
    def smoothing_wkf(self):
        """ Test adaptive smoothing build """
        
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        fname_bcounting = os.path.join(DATA_PATH, 'box_counting', 'boxcount_h3.csv')
        fname_out = os.path.join(HERE, 'smooth.csv')
        
        # Run the code
        code = os.path.join(HERE, '..', 'wkf_smoothing.jl')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, fname_bcounting, config, fname_out)
        #subprocess.call(cmd, shell=True)
        
        out = subprocess.call(cmd, shell=True)
        # Test results
        assert out == 1
        
        #expected = numpy.array([3.671158, 3.97219, 4.14828, 4.27322])
        # set up an expected....
        #computed = res.agr.to_numpy()
        #numpy.testing.assert_almost_equal(expected, computed, decimal=4)
