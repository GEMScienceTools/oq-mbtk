import os
import h5py
import shutil
import numpy
import unittest

from openquake.mbt.tools.tr.classify import classify

BASE_PATH = os.path.dirname(__file__)


class TrTestCase02(unittest.TestCase):
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
        Testing TR - Case 02 - Crustal and Subduction
        """
        tmps = '../../data/tr02/cls_v2_1500_2017.ini'
        ini_fname = os.path.join(BASE_PATH, tmps)
        tmps = '../../tmp/SARA_V2_1500_2017_nc.hdf5'
        treg_filename = os.path.join(BASE_PATH, tmps)
        #
        # classify
        classify(ini_fname, True, self.root_folder)
        f = h5py.File(treg_filename, 'r')
        #
        # testing crustal active
        expected = [1, 1, 0, 0, 0, 0, 0]
        numpy.testing.assert_array_equal(f['crustal'][:], expected)
        #
        # testing subduction interface
        expected = [0, 0, 0, 0, 0, 0, 1]
        numpy.testing.assert_array_equal(f['interface_1'][:], expected)
        #
        f.close()
