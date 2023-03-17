import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class AdaptiveSmoothingTest(unittest.TestCase):
    
    def setUp(self):

        fname = os.path.join(DATA_PATH, 'smooth_test.csv')
        
        parser = CsvCatalogueParser(fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        self.cat = cat

            
    def test_case01(self):
        """Test for adaptive smoothing - test intensity at event locations, n_v = 3, Gaussian kernel"""
                
        smooth = ak.AdaptiveSmoothing([self.cat.longitude, self.cat.latitude], grid = False, use_3d = False)
        ## Set up config
        config = {"kernel": "Gaussian", "n_v": 1, "d_i_min": 0.5}
        
        ## Apply adaptive smoothing
        adapt_mu = smooth.run_adaptive_smooth(cat, config )
        
    	expect_mu = ((0.004907, 0.002478, 0.005005, 0.002618, 0.001087))
        self.assertEqual(adapt_mu['nocc'], expect_mu)
        
    def test_case02(self):
        """Test for adaptive smoothing - test intensity at fixed locations, n_v = 1, Power Law kernel"""

        ## Set up config
        config = {'kernel':"PowerLaw" , 'n_v': 1, 'd_i_min':0.5}
        
        ## Apply adaptive smoothing
	smooth = ak.AdaptiveSmoothing([[-46], [12]], grid = False, use_3d = False)
    	adapt_mu = smooth.run_adaptive_smooth(cat, config )
    	# ((0.0001224756304465587, 0.00018776253602836626, 0.00033671052912002004, 0.0001140738487237348, 6.444434517520119e-05)
    	#expect_mu = 0.000825466889493881
    	expect_mu = 0.000826
    	# Helpfully the same as before!
        self.assertEqual(adapt_mu['nocc'], expect_mu)
        
    def test_case03(self):
        """ Test information gain """
        smooth = ak.AdaptiveSmoothing([cat.longitude, cat.latitude], grid = False, use_3d = False)
        config = {"kernel": "Gaussian", "n_v": 2, "d_i_min": 0.5}
        out = smooth.run_adaptive_smooth(cat, config )
        IG = smooth.information_gain(5, T = 1)
        self.assertEqual(IG, 1.0119860022288694)
