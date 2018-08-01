"""
"""

import os
import glob

import unittest

from openquake.sub.misc.edge import create_from_profiles, create_faults
from openquake.sub.misc.profile import _read_profile


BASE_DATA_PATH = os.path.dirname(__file__)


class CreateFaultsTest(unittest.TestCase):

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, '../data/slab/cs04/*.csv')
        profiles2 = []
        for filename in sorted(glob.glob(path)):
            profiles2.append(_read_profile(filename))
        #
        # building the mesh
        self.msh = create_from_profiles(profiles2, 50, 50, False)

    def test_create_faults_01(self):
        """
        Test the construction of a fault using the uppermost edge
        """
        print(self.msh.shape)
        for i in range(self.msh.shape[0]):
            #
            # create profiles - output is a list of list
            profs = create_faults(self.msh, i, 50, 45, 10)
            #
            # checking the number of profile-sets created
            if i == 5:
                self.assertEqual(len(profs), 2)
            elif i == 8:
                self.assertEqual(len(profs), 0)
            else:
                self.assertEqual(len(profs), 1)
