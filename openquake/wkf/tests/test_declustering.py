### wkf smoothing tests

import tempfile
import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser


HERE = os.path.dirname(__file__)
code_dir = os.path.normpath(os.path.join(*([HERE]+[".."]*2)))

#CWD = os.getcwd()
#DATA = os.path.relpath(os.path.join(HERE, 'data', 'rates_distribute'), CWD)

DATA_PATH = os.path.relpath(os.path.join(HERE, 'data'))

class test_decluster_catalogue(unittest.TestCase):

	def setUp(self):
		self.out_folder = tempfile.mkdtemp()

	def test_zaliapin_wkf(self):
		""" check that we can correctly call Zaliapin function and get expected output"""
        
		cat = os.path.join(DATA_PATH, 'zaliapin_test_catalogue.csv')
		config = os.path.join(DATA_PATH, 'config', 'wkf_testconfig_Zaliapin.toml')
		out_loc = self.out_folder
		
		cmd = f"oqm wkf decluster_catalogues {cat} {config} {out_loc}"
		p = subprocess.run(cmd, shell=True)
		
		res = pd.read_csv(os.path.join(self.out_folder, 'zaliapin_test_catalogue_dec_Zaliapin_undef.csv'))
		assert(len(res) == 12)

	def test_reasenberg_wkf(self):
		""" check that we can correctly call Zaliapin function and get expected output"""
        
		cat = os.path.join(DATA_PATH, 'zaliapin_test_catalogue.csv')
		config = os.path.join(DATA_PATH, 'config', 'wkf_testconfig_Reasenberg.toml')
		out_loc = self.out_folder
		
		cmd = f"oqm wkf decluster_catalogues {cat} {config} {out_loc}"
		p = subprocess.run(cmd, shell=True)
		
		res = pd.read_csv(os.path.join(self.out_folder, 'zaliapin_test_catalogue_dec_Reasenberg_undef.csv'))
		assert(len(res) == 12)

	def test_window_wkf(self):
		""" check that we can correctly call windowing function and get expected output"""
        
		cat = os.path.join(DATA_PATH, 'zaliapin_test_catalogue.csv')
		config = os.path.join(DATA_PATH, 'config', 'wkf_testconfig_GK.toml')
		out_loc = self.out_folder

		cmd = f"oqm wkf decluster_catalogues {cat} {config} {out_loc}"
		p = subprocess.run(cmd, shell=True)

		res = pd.read_csv(os.path.join(self.out_folder, 'zaliapin_test_catalogue_dec_GardnerKnopoffWindow_undef.csv'))
		assert(len(res) == 6)
