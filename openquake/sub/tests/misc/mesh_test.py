"""
"""

import os
import numpy
import unittest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openquake.sub.misc.profile import _read_profiles
from openquake.sub.misc.edge import create_from_profiles

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


class CreateMeshFromProfilesTest(unittest.TestCase):

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, 'south_america_segment6_slab')
        self.profiles, _ = _read_profiles(path)

    def test_mesh_creation(self):
        """
        Test construction of a mesh from profiles
        """
        sampling = 40
        idl = False
        smsh = create_from_profiles(self.profiles, sampling, sampling, idl)

        if False:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for i in range(smsh.shape[0]):
                ax.plot(smsh[i, :, 0], smsh[i, :, 1], smsh[i, :, 2]*0.1, '-r')
            for i in range(smsh.shape[1]):
                ax.plot(smsh[:, i, 0], smsh[:, i, 1], smsh[:, i, 2]*0.1, '-r')
            plt.show()

        idx = numpy.isfinite(smsh[:, :, 0])
        self.assertEqual(numpy.sum(numpy.sum(idx)), 202)
