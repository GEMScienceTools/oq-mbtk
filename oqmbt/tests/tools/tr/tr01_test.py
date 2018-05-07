import os
import h5py
import shutil
import numpy
import unittest

from oqmbt.tools.tr.classify import classify

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

    def testcase01(self):
        """
        Testing TR
        """
        ini_fname = os.path.join(BASE_PATH, '../../data/tr01/tr01.ini')
        treg_filename = os.path.join(BASE_PATH, '../../tmp/test02.hdf5')
        #
        # classify
        classify(ini_fname, True, self.root_folder)
        f = h5py.File(treg_filename, 'r')
        #
        # testing crustal active
        expected = [1, 1, 0, 0, 0, 0, 1, 1, 1]
        numpy.testing.assert_array_equal(f['crustal'][:], expected)
        #
        # testing interface
        expected = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        numpy.testing.assert_array_equal(f['int_cam'][:], expected)
        #
        # testing slab
        expected = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        numpy.testing.assert_array_equal(f['slab_cam'][:], expected)
        #
        f.close()
