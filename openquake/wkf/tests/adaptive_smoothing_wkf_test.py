### wkf adaptive smoothing tests

import tempfile
import toml
import h3
import subprocess
import os
import unittest
import numpy as np
import pandas as pd
import openquake.mbt.tools.adaptive_smoothing as ak
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
from pathlib import Path

HERE = Path(__file__).parent
#HERE = os.path.dirname(__file__)
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
        

    def test_read_config(self):
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        config = toml.load(config)
        config = config['smoothing']
        conf = {"kernel": config["kernel"], "n_v": config['n_v'], "d_i_min": config['d_i_min'], "h3res": config['h3res'], "maxdist": config['maxdist']}
        assert config['n_v'] == 1

    def test_h3_indexing(self):

        h3_map = os.path.join(DATA_PATH, 'mapping_h2.csv')
        h3_idx = pd.read_csv(h3_map, names = ("h3", "id"))
       
        # Get lat/lon locations for each h3 cell, convert to seperate lat and
        # lon columns of dataframe
        h3_idx['latlon'] = h3_idx.loc[:,"h3"].apply(h3.h3_to_geo)
        locations = pd.DataFrame(h3_idx['latlon'].tolist())
        locations.columns = ["lat", "lon"]
        np.testing.assert_almost_equal(locations.lon[0], -46.025932, decimal = 6) 

    def test_adap_smooth(self):

        #conf as above
        config = os.path.join(DATA_PATH, 'smooth_config.toml')
        config = toml.load(config)
        config = config['smoothing']
        conf = {"kernel": config["kernel"], "n_v": config['n_v'], "d_i_min": config['d_i_min'], "h3res": config['h3res'], "maxdist": config['maxdist']}

        # locations as above
        h3_map = os.path.join(DATA_PATH, 'mapping_h2.csv')
        h3_idx = pd.read_csv(h3_map, names = ("h3", "id"))
        h3_idx['latlon'] = h3_idx.loc[:,"h3"].apply(h3.h3_to_geo)
        locations = pd.DataFrame(h3_idx['latlon'].tolist())
        locations.columns = ["lat", "lon"]
        #locations = self.locations
        smooth = ak.AdaptiveSmoothing([locations.lon, locations.lat], grid=False, use_3d=False, use_maxdist = True)
        out = smooth.run_adaptive_smooth(self.cat, conf)

        expected = np.array([0.000984, 0.000983, 0.001360, 0.001375,0.002249,
        0.002325, 0.000881, 0.001852, 0.001765, 0.002598, 0.002911, 0.000004,
        0.000044, 0.000926, 0.000099, 0.002677, 0.001304])
        out = pd.DataFrame(out)
        out.columns = ["lon", "lat", "nocc"]
        
        np.testing.assert_almost_equal(expected, out['nocc'], decimal=4)

        tmpdir = Path(tempfile.gettempdir())
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        
        folder_out = tempfile.mkdtemp(suffix='adapsmooth', prefix=None, dir=tmpdir)
        fname_out = '{}/smooth_adap.csv'.format(folder_out)
        print(fname_out)

        out.to_csv(fname_out, header=True)

        computed = pd.read_csv(fname_out)
        np.testing.assert_almost_equal(expected, computed['nocc'], decimal=4)
    
    def test_adap_smoothing_wkf(self):
        """ Test adaptive smoothing build """

        # set up tmp directory and tmp file 
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

        assert p.returncode == 0

        expected = np.array([0.000984, 0.000983, 0.001360, 0.001375,0.002249,
        0.002325, 0.000881, 0.001852, 0.001765, 0.002598, 0.002911, 0.000004,
        0.000044, 0.000926, 0.000099, 0.002677, 0.001304])

        # set up an expected
        computed = pd.read_csv(fname_out)
        np.testing.assert_almost_equal(expected, computed['nocc'], decimal=4)
