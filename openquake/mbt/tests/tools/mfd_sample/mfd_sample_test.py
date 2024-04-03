import os
import unittest
import tempfile
import numpy as np
import pandas as pd

from openquake.mbt.tools.mfd_sample.make_mfds import _create_catalogue_versions

BASE_PATH = os.path.dirname(__file__)

class TestWorkflow(unittest.TestCase):

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
