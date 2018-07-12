"""
:module:`openquake.sub.tests.misc.utils`
"""

import os
import numpy as np
import unittest
from openquake.sub.misc.utils import create_inslab_meshes

BASE_DATA_PATH = os.path.abspath(os.path.dirname(__file__))


class CreateVirtualFaultsTest(unittest.TestCase):

    def setUp(self):
        # load data and create the mesh
        path = os.path.join(BASE_DATA_PATH, '..', 'data', 'misc', 'top_mesh')
        x = np.loadtxt(os.path.join(path, 'top_mesh.x'))
        y = np.loadtxt(os.path.join(path, 'top_mesh.y'))
        z = np.loadtxt(os.path.join(path, 'top_mesh.z'))
        self.mesh1 = np.stack((x, y, z), 2)
        # load data and create the mesh
        path = os.path.join(BASE_DATA_PATH, '..', 'data', 'misc', 'top_mesh_1')
        x = np.loadtxt(os.path.join(path, 'top_mesh_x.txt'))
        y = np.loadtxt(os.path.join(path, 'top_mesh_y.txt'))
        z = np.loadtxt(os.path.join(path, 'top_mesh_z.txt'))
        self.mesh2 = np.stack((x, y, z), 2)

    def test_create_inslab_profiles_1(self):
        """
        Test generation of virtual fault profiles 1
        """
        dips = [45, 135]
        thickness = 50
        sampling = 5
        oms = create_inslab_meshes(self.mesh1, dips, thickness, sampling)
        for dip in oms.keys():
            for prfl in oms[dip]:
                for prf in prfl:
                    ps = np.array([[p.longitude, p.latitude, p.depth] for p in
                                   prf.points])
                    assert not np.any(np.isnan(ps))

    def test_create_inslab_profiles_2(self):
        """
        Test generation of virtual fault profiles 2
        """
        dips = [45, 135]
        thickness = 50
        sampling = 5
        oms = create_inslab_meshes(self.mesh2, dips, thickness, sampling)
        for dip in oms.keys():
            for prfl in oms[dip]:
                for prf in prfl:
                    ps = np.array([[p.longitude, p.latitude, p.depth] for p in
                                   prf.points])
                    assert not np.any(np.isnan(ps))

