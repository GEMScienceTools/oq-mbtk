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
        shutil.rmtree(self.tmp)

    def testcase01(self):
        """
        Testing TR for an idealized case - only crustal
        """
        ini_fname = os.path.join(BASE_PATH, '../../data/tr/only_crustal.ini')
        treg_filename = os.path.join(BASE_PATH, '../../tmp/test01.hdf5')
        #
        # classify
        classify(ini_fname, self.root_folder, True)
        #
        #
        f = h5py.File(treg_filename, 'r')
        numpy.testing.assert_array_equal(f['crustal'][:], [1, 0, 1, 0, 1])
        f.close()

    def testcase02(self):
        """
        Testing TR for an idealized case - stable and active crust
        """
        ini_fname = os.path.join(BASE_PATH, '../../data/tr/acr_scr.ini')
        treg_filename = os.path.join(BASE_PATH, '../../tmp/test02.hdf5')
        #
        # classify
        classify(ini_fname, self.root_folder, True)
        f = h5py.File(treg_filename, 'r')
        #
        # testing crustal active
        expected = [1, 0, 1, 0, 0]
        numpy.testing.assert_array_equal(f['crustal_active'][:], expected)
        #
        # testing crustal stable
        expected = [0, 0, 0, 0, 1]
        numpy.testing.assert_array_equal(f['crustal_stable'][:], expected)
        #
        f.close()


    def testcase03(self):
        """
        Testing TR for an idealized case
        """
        ini_fname = os.path.join(BASE_PATH, '../../data/tr/sample.ini')
        treg_filename = os.path.join(BASE_PATH, '../../tmp/test02.hdf5')
        #
        # classify
        classify(ini_fname, self.root_folder, True)
        #
        # testing
        f = h5py.File(treg_filename, 'r')
        numpy.testing.assert_array_equal(f['crustal'][:], [0, 0, 1, 0, 1])
        #
        f.close()
