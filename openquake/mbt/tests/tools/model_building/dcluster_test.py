import os
import shutil
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
        self.tmp = os.path.join(BASE_PATH, '../../tmp/')
        #
        # creating temporary folder
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)
        #
        # set datasets
        tmps = '../../data/tools/model_building/dcluster/catalogue.csv'
        self.catalogue = os.path.abspath(os.path.join(BASE_PATH, tmps))
        tmps = '../../data/tools/model_building/dcluster/classification.hdf5'
        self.classification = os.path.join(BASE_PATH, tmps)

    def tearDown(self):
        #
        # removing tmp folder
        # shutil.rmtree(self.tmp)
        pass

    def testcase01(self):
        """
        """
        config = {'time_distance_window': 'GardnerKnopoffWindow',
                  'fs_time_prop': 0.9}
        decluster(catalogue_hmtk_fname=self.catalogue,
                  declustering_meth='GardnerKnopoffType1',
                  declustering_params=config,
                  output_path=self.tmp,
                  labels=['a', 'b'],
                  tr_fname=self.classification,
                  subcatalogues=True,
                  format='pkl')
        #
        # Read first mainshock catalogue
        a_fname = os.path.abspath(os.path.join('./../../tmp/catalogue_dec_a.pkl'))
        self.assertTrue(os.path.exists(a_fname))
        cat = _load_catalogue(a_fname)
        self.assertTrue(len(cat.data['magnitude'] == 1))
        self.assertAlmostEqual(cat.data['magnitude'][0], 6.0)
        #
        # Read second mainshock catalogue
        b_fname = os.path.join('./../../tmp/catalogue_dec_b.pkl')
        self.assertTrue(os.path.exists(b_fname))
        cat = _load_catalogue(b_fname)
        self.assertTrue(len(cat.data['magnitude'] == 1))
        self.assertAlmostEqual(cat.data['magnitude'][0], 6.1)
        #
        # Check that the third mainshock catalogue does not exist
        c_fname = os.path.join('./../../tmp/catalogue_dec_c.pkl')
        self.assertFalse(os.path.exists(c_fname))
