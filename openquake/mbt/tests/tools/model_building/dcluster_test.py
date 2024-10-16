import os
import shutil
import pathlib
import tempfile
import unittest

from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.mbt.tools.model_building.dclustering import decluster
from openquake.mbt.tests import __file__ as tests__init__

TDIR = pathlib.Path(tests__init__).parent


class TrTestCase(unittest.TestCase):
    """
    This class tests the construction of subcatalogues done by the
    dclusterer function.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

        # Set datasets
        self.catalogue = str(
            TDIR / 'data' / 'tools' / 'model_building' / 'dcluster' /
            'catalogue.csv')
        self.classification = (
            TDIR / 'data' / 'tools' / 'model_building' / 'dcluster' /
            'classification.hdf5')

    def testcase01(self):
        config = {'time_distance_window': 'GardnerKnopoffWindow',
                  'fs_time_prop': 0.9}
        decluster(catalogue_hmtk_fname=self.catalogue,
                  declustering_meth='GardnerKnopoffType1',
                  declustering_params=config,
                  output_path=self.tmp,
                  labels=['a', 'b'],
                  tr_fname=self.classification,
                  subcatalogues=True,
                  fmat='pkl')

        # Read first mainshock catalogue
        a_fname = os.path.join(self.tmp, 'catalogue_dec__a.pkl')
        self.assertTrue(os.path.exists(a_fname))
        cat = _load_catalogue(a_fname)
        self.assertTrue(len(cat.data['magnitude'] == 1))
        self.assertAlmostEqual(cat.data['magnitude'][0], 6.0)

        # Read second mainshock catalogue
        b_fname = os.path.join(self.tmp, 'catalogue_dec__b.pkl')
        self.assertTrue(os.path.exists(b_fname))
        cat = _load_catalogue(b_fname)
        self.assertTrue(len(cat.data['magnitude'] == 1))
        self.assertAlmostEqual(cat.data['magnitude'][0], 6.1)

        # Check that the third mainshock catalogue does not exist
        c_fname = os.path.join(self.tmp, 'catalogue_dec__c.pkl')
        self.assertFalse(os.path.exists(c_fname))
