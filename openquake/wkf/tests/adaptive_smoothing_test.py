import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
import openquake.wkf.wkf_h3_zones_cat
import subprocess


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class AdaptiveSmoothingTest(unittest.TestCase):
    
    def setUp(self):

        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        self.fname = fname
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        self.cat = cat
        

            
    def test_adaptive01(self):
        """Test for adaptive smoothing - test intensity at event locations, n_v = 3, Gaussian kernel"""
        cat = self.cat
        smooth = ak.AdaptiveSmoothing([cat.data['longitude'], cat.data['latitude']], grid = False, use_3d = False)
        ## Set up config
        config = {"kernel": "Gaussian", "n_v": 1, "d_i_min": 0.5}
        
        ## Apply adaptive smoothing
        adapt_mu = smooth.run_adaptive_smooth(cat, config )
        expect_mu = ((0.004907, 0.002478, 0.005005, 0.002618, 0.001087))
        obs_mu = adapt_mu['nocc'].values

        for i in range(len(obs_mu)): self.assertAlmostEqual(expect_mu[i], obs_mu[i], places = 6)
      
        
    def test_adaptive_fixed_loc(self):
        """Test for adaptive smoothing - test intensity at fixed locations, n_v = 1, Power Law kernel"""
        cat = self.cat
        ## Set up config
        config = {'kernel':"PowerLaw" , 'n_v': 1, 'd_i_min':0.5}
        
        ## Apply adaptive smoothing
        smooth = ak.AdaptiveSmoothing([[-46], [12]], grid = False, use_3d = False)
        adapt_mu = smooth.run_adaptive_smooth(cat, config )
        expect_mu = 0.000826
        self.assertAlmostEqual(adapt_mu['nocc'].values[0], expect_mu, places = 6)
        
    def test_infogain(self):
        """ Test information gain """
        cat = self.cat
        smooth = ak.AdaptiveSmoothing([cat.data['longitude'], cat.data['latitude']], grid = False, use_3d = False)
        config = {"kernel": "Gaussian", "n_v": 2, "d_i_min": 0.5}
        out = smooth.run_adaptive_smooth(cat, config )
        IG = smooth.information_gain(5, T = 1)
        self.assertEqual(IG, 1.0119860022288694)
        
    def test_cat_zones(self):
        """ test catalogue zone function """
        h3_level = 5
        #mor_cat = fname
        fname = self.fname
        fname_out = os.path.join(DATA_PATH, '/tmp/h3.csv')
        cmd = f"oqm wkf wkf_h3_zones_cat {h3_level} {fname} {fname_out}"
        p = subprocess.run(cmd, shell=True)
        
        h3_idx = pd.read_csv(fname_out)
        self.assertEqual(len(h3_idx), 2850)
