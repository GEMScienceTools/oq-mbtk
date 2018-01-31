import os
import unittest

from oqmbt.tools.tr.catalogue import get_catalogue

BASE_PATH = os.path.dirname(__file__)


class CatalogueReadTestCase(unittest.TestCase):
    """
    This class tests the tectonic regionalisation workflow
    """

    def setUp(self):
        tmps = './../../data/tr/catalogue_sample.csv'
        self.cat_fname = os.path.join(BASE_PATH, tmps)

    def testcase01(self):
        """
        """
        cat = get_catalogue(self.cat_fname)
        expected = 11
        self.assertEqual(expected, len(cat.data['longitude']))
