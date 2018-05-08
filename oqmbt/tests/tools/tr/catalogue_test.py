import os
import re
import unittest

from oqmbt.tools.tr.catalogue import get_catalogue

BASE_PATH = os.path.dirname(__file__)


class CatalogueReadTestCase(unittest.TestCase):
    """
    This class tests the tectonic regionalisation workflow
    """

    def testcase01(self):
        """
        Read .csv catalogue
        """
        tmps = './../../data/tr/catalogue_sample.csv'
        cat = get_catalogue(os.path.join(BASE_PATH, tmps))
        expected = 11
        self.assertEqual(expected, len(cat.data['longitude']))
        #
        # pickle file
        tmpo = os.path.join(BASE_PATH, tmps)
        assert os.path.exists(re.sub('ndk$', 'pkl', tmpo))
        print(re.sub('csv$', 'pkl', tmpo))
        os.remove(re.sub('csv$', 'pkl', tmpo))

    """
    def testcase02(self):
        tmps = './../../data/tr/gcmt_sample.ndk'
        cat = get_catalogue(os.path.join(BASE_PATH, tmps))
        expected = 9
        self.assertEqual(expected, len(cat.data['longitude']))
        assert 'latitude' in cat.data
        assert 'depth' in cat.data
        assert 'magnitude' in cat.data
        #
        # pickle file
        tmpo = os.path.join(BASE_PATH, tmps)
        assert os.path.exists(re.sub('ndk$', 'pkl', tmpo))
        os.remove(re.sub('ndk$', 'pkl', tmpo))
    """
