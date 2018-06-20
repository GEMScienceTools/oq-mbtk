import os
import h5py
import code
import shutil
import numpy
import unittest

from openquake.mbt.tools.tr.classify import classify
from openquake.mbt.tools.tr.change_class import change

BASE_PATH = os.path.dirname(__file__)


class ChangeTrTestCase(unittest.TestCase):
    """
    This class tests manual change from one class to another
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
        Testing TR change from crustal to stable crustal
        """
        ini_fname = os.path.join(BASE_PATH, '../../data/tr/acr_scr.ini')
        cat = os.path.join(BASE_PATH,'../../data/tr/catalogue.pkl')
        treg_filename = os.path.join(BASE_PATH, '../../tmp/test02.hdf5')
        event_change = os.path.join(BASE_PATH,'../../data/tr/ev_change.csv')
        #
        # classify
        classify(ini_fname, True, self.root_folder)
        f = h5py.File(treg_filename, 'r')
        #
        # manually change an event
        change(cat,treg_filename,event_change)

        # testing new file to see if the swap worked
        treg_filename2 = os.path.join(BASE_PATH, '../../tmp/test02_up.hdf5')
        f2 = h5py.File(treg_filename2, 'r')

        expected = [1, 0, 0, 0, 0]
        numpy.testing.assert_array_equal(f2['crustal_active'][:], expected)
        #
        # testing crustal stable
        expected = [0, 0, 1, 0, 1]
        numpy.testing.assert_array_equal(f2['crustal_stable'][:], expected)
        #
        f.close()

