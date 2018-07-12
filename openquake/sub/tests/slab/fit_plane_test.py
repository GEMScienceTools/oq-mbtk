"""
"""
import numpy as np
import unittest
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.utils import plane_fit


class FitPlaneTest(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        lons = np.array([10.0, 10.0, 10.5, 11.0, 11.0, 10.5])
        lats = np.array([45.0, 45.2, 45.2, 45.2, 45.0, 45.0])
        deps = np.array([2.0, 10.0, 10.0, 10.0, 2.0, 2.0])
        self.mesh = Mesh(lons, lats, deps)

    def test_fit_plane_again(self):
        """
        This is a trivial test to check (again) the code we use for fitting
        the plane
        """
        a = 0.0
        b = 0.0
        c = 1
        d = -1
        expected_par = np.array([a, b, c])
        x, y = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
        z = -(a * x + b * y + d) / c
        xx = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        pnt, par = plane_fit(xx)
        np.testing.assert_almost_equal(expected_par, par)
