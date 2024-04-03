import os
import toml
import shutil
import unittest
import tempfile
import numpy as np
import pandas as pd

from openquake.mbt.tools.mfd_sample.make_mfds import (_create_catalogue_versions,
                                                      make_many_mfds)

BASE_PATH = os.path.dirname(__file__)

class TestGenCats(unittest.TestCase):

    def setUp(self):
        self.catfi = os.path.join(BASE_PATH, 'data', 'catalogue.csv')
        # Create the temporary folder                                           
        self.tmpd = next(tempfile._get_candidate_names())

    def test_generate_cats(self):
        """
        Test calculation of exceedance rate for magnitude equal to bin limit
        """
        _create_catalogue_versions(self.catfi, self.tmpd, 2, stype='random',
                               numstd=1, rseed=122)

        mags_exp_fi = os.path.join(BASE_PATH, 'expected', 'v_mags.csv')
        mags_out_fi = os.path.join(self.tmpd, 'v_mags.csv')
        expected = pd.read_csv(mags_exp_fi)
        output = pd.read_csv(mags_out_fi)
        assert expected.equals(output)

        shutil.rmtree(self.tmpd)

class TestWorkflow(unittest.TestCase):

    def test_full_wkflow(self):
        # reassign main configs
        config_fi = os.path.join(BASE_PATH, 'config', 'test.toml')
        config = toml.load(config_fi)
        cat_fi_name = config['main']['catalogue_filename']
        cat_fi_name_new = os.path.join(BASE_PATH, cat_fi_name)
        config['main']['catalogue_filename'] = cat_fi_name_new
        dec_toml = config['decluster']['decluster_settings']
        dec_toml_new = os.path.join(BASE_PATH, dec_toml)
        comp_toml = config['completeness']['completeness_settings']
        comp_toml_new = os.path.join(BASE_PATH, comp_toml)
        config['completeness']['completeness_settings'] = comp_toml_new

        # reassign decluster configs
        config_dec = toml.load(dec_toml_new)
        config_dec['main']['catalogue'] = cat_fi_name_new
        dec_tr = config_dec['main']['tr_file']
        dec_tr_new = os.path.join(BASE_PATH, dec_tr)
        config_dec['main']['tr_file'] = dec_tr_new
        config_dec_fi_new = os.path.join(BASE_PATH, 'test_dec_tmp.toml')
        config['decluster']['decluster_settings'] = config_dec_fi_new

        config_fi_new = os.path.join(BASE_PATH, 'test_tmp.toml')
        file=open(config_fi_new,"w")
        toml.dump(config, file)
        file.close()

        decfile=open(config_dec_fi_new,"w")
        toml.dump(config_dec, decfile)
        decfile.close()

        make_many_mfds(config_fi_new, BASE_PATH)

        expected_fi = os.path.join(BASE_PATH, 'expected2', 'mfd-results.csv')
        output_fi = os.path.join(BASE_PATH, 'test', 'mfd-results.csv')
        expected = pd.read_csv(expected_fi)
        output = pd.read_csv(output_fi)
        assert expected.equals(output)

        shutil.rmtree('test')
        os.remove('tmp-config-dcl.toml')
