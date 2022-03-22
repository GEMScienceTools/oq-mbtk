import os
import h5py
import shutil
import numpy
import unittest

from openquake.mbt.tools.tr.classify import classify
from openquake.mbt.tools.tr.catalogue import get_catalogue

BASE_PATH = os.path.dirname(__file__)


class TrTestCase(unittest.TestCase):
    """
    This class tests the tectonic regionalisation workflow
    """

    def setUp(self):
        self.root_folder = os.path.join(BASE_PATH)
        self.tmp = os.path.join(BASE_PATH, '../../tmp/')
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)

    def tearDown(self):
        #
        # removing tmp folder
        # shutil.rmtree(self.tmp)
        pass

    def testcase03(self):
        """
        Testing TR
        """
        ini_fname = os.path.join(BASE_PATH, '../../data/tr03/tr03.ini')
        treg_filename = os.path.join(BASE_PATH, '../../tmp/test03.hdf5')
        catalogue = os.path.join(BASE_PATH,'../../data/tr03/cat_4001.pkl')
        c = get_catalogue(catalogue)
        c_num = len(c.data['eventID'])
        #
        # classify
        classify(ini_fname, True, self.root_folder)
        f = h5py.File(treg_filename, 'r')
        #
        # testing crustal active
        expected = numpy.ones(c_num)
        numpy.testing.assert_array_equal(f['crustal'][:], expected)
        #
        # testing slab
        expected = numpy.zeros(c_num)
        numpy.testing.assert_array_equal(f['slab_deep'][:], expected)
        #
        f.close()
