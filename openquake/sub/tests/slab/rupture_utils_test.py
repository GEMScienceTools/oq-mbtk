
import os
import unittest
import numpy as np
from openquake.sub.slab.rupture_utils import get_discrete_dimensions
from openquake.sub.slab.rupture_utils import get_ruptures
from openquake.hazardlib.geo.mesh import Mesh

BASE_DATA_PATH = os.path.dirname(__file__)


class CreateRuptureTest(unittest.TestCase):

    def setUp(self):
        # load data and create the mesh
        path = os.path.join(BASE_DATA_PATH, '..', 'data', 'virtual_fault')
        x = np.loadtxt(os.path.join(path, 'virtual_fault.x'))
        y = np.loadtxt(os.path.join(path, 'virtual_fault.y'))
        z = np.loadtxt(os.path.join(path, 'virtual_fault.z'))
        self.mesh = Mesh(x, y, z)

    def test_get_ruptures_m(self):
        """
        TODO Test rupture generation
        """
        rups = [r for r in get_ruptures(self.mesh, 1, 1)]


class RuptureGenerationTest(unittest.TestCase):

    def setUp(self):
        # creating first mesh
        x = np.arange(10.0, 11.0, 0.1)
        y = np.arange(45.0, 45.5, 0.1)
        z = np.arange(0, 25, 5)
        xg, yg = np.meshgrid(x, y)
        _, zg = np.meshgrid(x, z)
        self.mesh = Mesh(xg, yg, zg)
        # creating second mesh
        xga, yga = np.meshgrid(x, y)
        _, zga = np.meshgrid(x, z)
        xga[0, 0] = np.nan
        yga[0, 0] = np.nan
        self.mesha = Mesh(xga, yga, zg)

    def test_rupture_2x2(self):
        """
        Checking the number of ruptures with ar of 1
        """
        rups = [r for r in get_ruptures(self.mesh, 2, 2)]
        self.assertEqual(len(rups), 36)

    def test_rupture_4x2(self):
        """
        Checking the number of ruptures with ar of 2
        """
        rups = [r for r in get_ruptures(self.mesh, 4, 2)]
        self.assertEqual(len(rups), 28)

    def test_rupture_2x4(self):
        """
        Checking the number of ruptures with ar of 0.5
        """
        rups = [r for r in get_ruptures(self.mesh, 2, 4)]
        self.assertEqual(len(rups), 18)

    def aa_test_rupture_2x2_nan(self):
        """
        Testing the case with a NaN value
        """
        rups = [r for r in get_ruptures(self.mesha, 2, 2)]
        self.assertEqual(len(rups), 35)


class RuptureDiscretizationTest(unittest.TestCase):

    def test_discrete_rupture(self):
        """
        Check the discretization of a rupture
        """
        area = 800.
        asr = 2.
        sampling = 10.
        le, wi = get_discrete_dimensions(area, sampling, asr)
        self.assertEqual(le, 40)
        self.assertEqual(wi, 20)
