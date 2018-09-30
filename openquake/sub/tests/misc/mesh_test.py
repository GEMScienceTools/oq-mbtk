"""
:module:`openquake.sub.tests.misc.mesh_test`
"""

import os
import numpy
import unittest

import matplotlib.pyplot as plt
# MN: 'Axes3D' imported but never used
from mpl_toolkits.mplot3d import Axes3D

from openquake.sub.misc.profile import _read_profiles
from openquake.sub.misc.edge import create_from_profiles
from openquake.hazardlib.geo.geodetic import distance

# from openquake.sub.tests.misc.utils_plot import plotter

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')


def ppp(profiles, smsh):
    """plotting"""
    scl = 0.1
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for l in profiles:
        coo = [[p.longitude, p.latitude, p.depth] for p in l]
        coo = numpy.array(coo)
        ax.plot(coo[:, 0], coo[:, 1], coo[:, 2]*scl, '--g', lw=2)
        ax.plot(coo[:, 0], coo[:, 1], coo[:, 2]*scl, 'og', lw=2)
    #
    for i in range(smsh.shape[0]):
        ax.plot(smsh[i, :, 0], smsh[i, :, 1], smsh[i, :, 2]*scl, '-r')
    for i in range(smsh.shape[1]):
        ax.plot(smsh[:, i, 0], smsh[:, i, 1], smsh[:, i, 2]*scl, '-r')
    ax.invert_zaxis()
    plt.show()


class IdealisedSimpleMeshTest(unittest.TestCase):
    """
    This is the simplest test implemented for the construction of the mesh. It
    uses just two parallel profiles gently dipping northward and it checks
    that the size of the cells agrees with the input parameters
    """

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, 'profiles05')
        self.profiles, _ = _read_profiles(path)

    def test_mesh_creation(self):
        """ Create the mesh: two parallel profiles - no top alignment """
        h_sampl = 4
        v_sampl = 4
        idl = False
        alg = False
        smsh = create_from_profiles(self.profiles, h_sampl, v_sampl, idl, alg)

        # plotter(self.profiles, smsh)

        #
        # Check the horizontal mesh spacing
        computed = []
        for i in range(0, smsh.shape[0]):
            tmp = []
            for j in range(0, smsh.shape[1]-1):
                k = j + 1
                dst = distance(smsh[i, j, 0], smsh[i, j, 1], smsh[i, j, 2],
                               smsh[i, k, 0], smsh[i, k, 1], smsh[i, k, 2])
                tmp.append(dst)
            computed.append(dst)
        computed = numpy.array(computed)
        self.assertTrue(numpy.all(abs(computed-h_sampl)/h_sampl < 0.05))
        #
        # Check the vertical mesh spacing
        computed = []
        for i in range(0, smsh.shape[0]-1):
            tmp = []
            k = i + 1
            for j in range(0, smsh.shape[1]):
                dst = distance(smsh[i, j, 0], smsh[i, j, 1], smsh[i, j, 2],
                               smsh[k, j, 0], smsh[k, j, 1], smsh[k, j, 2])
                tmp.append(dst)
            computed.append(dst)
        computed = numpy.array(computed)
        print(numpy.amax(abs(computed-v_sampl)/v_sampl))
        self.assertTrue(numpy.all(abs(computed-v_sampl)/v_sampl < 0.05))


class IdealisedSimpleDisalignedMeshTest(unittest.TestCase):
    """
    Similar to
    :class:`openquake.sub.tests.misc.mesh_test.IdealisedSimpleMeshTest`
    but with profiles at different depths
    """

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, 'profiles06')
        self.profiles, _ = _read_profiles(path)
        self.h_sampl = 4
        self.v_sampl = 4
        idl = False
        alg = False
        self.smsh = create_from_profiles(self.profiles, self.h_sampl,
                                         self.v_sampl, idl, alg)
        # ppp(self.profiles, self.smsh)

    def test_h_spacing(self):
        """ Check h-spacing: two misaligned profiles - no top alignment """
        smsh = self.smsh
        #
        # Check the horizontal mesh spacing
        computed = []
        for i in range(0, smsh.shape[0]):
            tmp = []
            for j in range(0, smsh.shape[1]-1):
                k = j + 1
                dst = distance(smsh[i, j, 0], smsh[i, j, 1], smsh[i, j, 2],
                               smsh[i, k, 0], smsh[i, k, 1], smsh[i, k, 2])
                tmp.append(dst)
            computed.append(dst)
        computed = numpy.array(computed)
        tmp = abs(computed-self.h_sampl)/self.h_sampl
        self.assertTrue(numpy.all(tmp < 0.05))

    def test_v_spacing(self):
        """ Check v-spacing: two misaligned profiles - no top alignment """
        smsh = self.smsh
        computed = []
        for i in range(0, smsh.shape[0]-1):
            tmp = []
            k = i + 1
            for j in range(0, smsh.shape[1]):
                dst = distance(smsh[i, j, 0], smsh[i, j, 1], smsh[i, j, 2],
                               smsh[k, j, 0], smsh[k, j, 1], smsh[k, j, 2])
                tmp.append(dst)
            computed.append(dst)
        computed = numpy.array(computed)
        tmp = abs(computed-self.h_sampl)/self.h_sampl
        self.assertTrue(numpy.all(tmp < 0.05))


class IdealisedAsimmetricMeshTest(unittest.TestCase):

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, 'profiles03')
        self.profiles, _ = _read_profiles(path)

    def test_mesh_creation(self):
        """ Test construction of the mesh """
        h_sampl = 5
        v_sampl = 5
        idl = False
        alg = False
        smsh = create_from_profiles(self.profiles, h_sampl, v_sampl, idl, alg)
        # ppp(self.profiles, smsh)
        # plotter(self.profiles, smsh)
        idx = numpy.isfinite(smsh[:, :, 0])

    def test_mesh_creation_with_alignment(self):
        """ Test construction of the mesh """
        h_sampl = 5
        v_sampl = 5
        idl = False
        alg = True
        smsh = create_from_profiles(self.profiles, h_sampl, v_sampl, idl, alg)
        # ppp(self.profiles, smsh)
        # plotter(self.profiles, smsh)
        idx = numpy.isfinite(smsh[:, :, 0])


class IdealizedATest(unittest.TestCase):

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, 'profiles04')
        self.profiles, _ = _read_profiles(path)

    def test_mesh_creation_no_alignment(self):
        """ Test construction of the mesh """
        h_sampl = 4
        v_sampl = 4
        idl = False
        alg = False
        smsh = create_from_profiles(self.profiles, h_sampl, v_sampl, idl, alg)
        # ppp(self.profiles, smsh)
        idx = numpy.isfinite(smsh[:, :, 0])

    def test_mesh_creation_with_alignment(self):
        """ Test construction of the mesh """
        h_sampl = 4
        v_sampl = 4
        idl = False
        alg = True
        smsh = create_from_profiles(self.profiles, h_sampl, v_sampl, idl, alg)
        # ppp(self.profiles, smsh)
        idx = numpy.isfinite(smsh[:, :, 0])


class SouthAmericaSegmentTest(unittest.TestCase):

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, 'south_america_segment6_slab')
        self.profiles, _ = _read_profiles(path)

    def test_mesh_creation(self):
        """ Create mesh from profiles for SA """
        sampling = 40
        idl = False
        alg = False
        smsh = create_from_profiles(self.profiles, sampling, sampling, idl,
                                    alg)
        # ppp(self.profiles, smsh)
        # plotter(self.profiles, smsh)
        idx = numpy.isfinite(smsh[:, :, 0])
        self.assertEqual(numpy.sum(numpy.sum(idx)), 202)
