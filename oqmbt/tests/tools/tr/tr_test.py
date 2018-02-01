import os
import unittest

from oqmbt.tools.tr.classify import classify

BASE_PATH = os.path.dirname(__file__)


class TrTestCase(unittest.TestCase):
    """
    This class tests the tectonic regionalisation workflow
    """

    def setUp(self):
        self.ini_fname = os.path.join(BASE_PATH, './../../data/tr/co18.ini')
        self.root = os.path.join(BASE_PATH, './../../data/tr/data')


    def testcase01(self):
        """
        """
        # classify(self.ini_fname)
        pass
