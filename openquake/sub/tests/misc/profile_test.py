"""
"""

import os
import glob

import numpy as np
import unittest

from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.geo import Point, Line

from openquake.sub.misc.profile import (_read_profile, _resample_profile,
                                        profiles_depth_alignment)

BASE_DATA_PATH = os.path.dirname(__file__)


class AlignProfilesTest(unittest.TestCase):

    def setUp(self):
        self.pro1 = Line([Point(10.0, 45.0, 5.0),
                          Point(10.1, 45.1, 10.0),
                          Point(10.2, 45.2, 15.0),
                          Point(10.0, 45.3, 20.0)])
        self.pro2 = Line([Point(10.0, 45.0, 2.0),
                          Point(10.1, 45.1, 8.0),
                          Point(10.2, 45.2, 11.0),
                          Point(10.0, 45.3, 19.0)])
        self.pro3 = Line([Point(10.0, 45.0, 2.0),
                          Point(10.1, 45.1, 8.0),
                          Point(10.2, 45.2, 11.0),
                          Point(10.0, 45.3, 17.0)])
        self.pro4 = Line([Point(10.0, 45.0, 8.0),
                          Point(10.1, 45.1, 13.0),
                          Point(10.2, 45.2, 17.0),
                          Point(10.0, 45.3, 23.0)])
        self.pro5 = Line([Point(10.0, 45.0, 3.0),
                          Point(10.1, 45.1, 8.0),
                          Point(10.2, 45.2, 12.0)])

    def test_alignement_01(self):
        """
        Profiles with two equivalent options. Takes the one with less
        shifting
        """
        idx = profiles_depth_alignment(self.pro1, self.pro2)
        self.assertEqual(0, idx)

    def test_alignement_02(self):
        idx = profiles_depth_alignment(self.pro1, self.pro3)
        # self.assertEqual(-1, idx)

    def test_alignement_03(self):
        idx = profiles_depth_alignment(self.pro1, self.pro4)
        # self.assertEqual(1, idx)

    def test_alignement_04(self):
        idx = profiles_depth_alignment(self.pro1, self.pro5)
        # self.assertEqual(-1, idx)


class ResampleProfileTest(unittest.TestCase):
    """
    """

    def _test(self, filename):
        profile = _read_profile(filename)
        sampling_distance = 10.
        rprofile = _resample_profile(profile, sampling_distance)
        pro = np.array([(pnt.longitude, pnt.latitude, pnt.depth) for pnt in
                        rprofile.points])
        dsts = distance(pro[:-1, 0], pro[:-1, 1], pro[:-1, 2],
                        pro[1:, 0], pro[1:, 1], pro[1:, 2])
        np.testing.assert_allclose(dsts, sampling_distance, rtol=1)

    def test01(self):
        """
        Resample profile
        """
        filename = os.path.join(BASE_DATA_PATH, '../data/slab/cs/cs_16.csv')
        self._test(filename)

    def test02(self):
        """
        Resample set of profiles
        """
        path = os.path.join(BASE_DATA_PATH, '../data/slab/cs02/cs*.csv')
        for filename in glob.glob(path):
            self._test(filename)
