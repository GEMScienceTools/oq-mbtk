
import os
import unittest

from openquake.sub.profiles import ProfileSet


class ProfileTest(unittest.TestCase):
    """
    """

    BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

    def setUp(self):
        """
        """
        self.dname_profile = os.path.join(self.BASE_DATA_PATH, 'cs_cam')
        tmps = 'south_america_segment6_slab'
        self.fname_b = os.path.join(self.BASE_DATA_PATH, tmps)

    def test_reading_folder(self):
        """
        Read profiles from a folder
        """
        prfs = ProfileSet.from_files(self.dname_profile)
        self.assertEqual(27, len(prfs.profiles))

    def test_smooth(self):
        """
        Test spline construction
        """
        prfs = ProfileSet.from_files(self.fname_b)
        grd = prfs.smooth('cubic')
