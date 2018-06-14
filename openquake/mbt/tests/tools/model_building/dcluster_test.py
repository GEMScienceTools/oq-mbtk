import os
import h5py
import shutil
import numpy
import unittest

from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.mbt.tools.model_building.dclustering import decluster

BASE_PATH = os.path.dirname(__file__)


class TrTestCase(unittest.TestCase):
    """
    This class tests the construction of subcatalogues done by the
    dclusterer function.
    """

    def setUp(self):
        self.root_folder = os.path.join(BASE_PATH)
        self.tmp = os.path.join(BASE_PATH, '../../test/tmp/')
        #
        # creating temporary folder
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)
        #
        # set datasets
        tmps = '../../data/tools/model_building/dcluster/catalogue.csv'
        self.catalogue = os.path.abspath(os.path.join(BASE_PATH, tmps))
        print(self.catalogue)
        tmps = '../../data/tools/model_building/dcluster/classification.hdf5'
        self.classification = os.path.join(BASE_PATH, tmps)

    def tearDown(self):
        #
        # removing tmp folder
        shutil.rmtree(self.tmp)

    def testcase01(self):
        """
        """
        config = {}
        decluster(catalogue_hmtk_fname=self.catalogue,
                  declustering_meth='GardnerKnopoffType1',
                  declustering_params=None,
                  output_path=self.tmp,
                  labels=['a','b'],
                  tr_fname=self.classification,
                  subcatalogues=True)
        #
        # Read first mainshock catalogue
        c_fname = os.path.join('./../../test/tmp/catalogue_dec_a.p')
        self.assertTrue(os.path.exists(c_fname))
        cat = _load_catalogue(c_fname)
        self.assertTrue(len(cat.data['magnitude'] == 1))
        self.assertAlmostEqual(cat.data['magnitude'][0], 6.0)
        #
        # Read second mainshock catalogue
        c_fname = os.path.join('./../../test/tmp/catalogue_dec_b.p')
        self.assertTrue(os.path.exists(c_fname))
        cat = _load_catalogue(c_fname)
        self.assertTrue(len(cat.data['magnitude'] == 1))
        self.assertAlmostEqual(cat.data['magnitude'][0], 6.1)
        #
        # Check that the third mainshock catalogue does not exist
        c_fname = os.path.join('./../../test/tmp/catalogue_dec_c.p')
        self.assertFalse(os.path.exists(c_fname))


