#!/usr/bin/env python
# coding: utf-8

import pathlib
import unittest
import numpy as np

from openquake.fnm.connections import get_angles
from openquake.hazardlib.geo.mesh import RectangularMesh
from openquake.hazardlib.geo.geodetic import npoints_towards

PLOTTING = False
HERE = pathlib.Path(__file__).parent


class TestGetAngle(unittest.TestCase):

    def test_case_01(self):
        """Test angle between subsections at 120°"""

        lons = np.array([[10.0, 10.2], [10.0, 10.2]])
        lats = np.array([[45.0, 45.0], [45.0, 45.0]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_a = RectangularMesh(lons, lats, deps)

        tlo, tla, tde = npoints_towards(10.25, 45.0, 0.0, 30, 20.0, 0.0, 3)
        lons = np.array([tlo[1:], tlo[1:]])
        lats = np.array([tla[1:], tla[1:]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_b = RectangularMesh(lons, lats, deps)

        computed, computed_t_ang = get_angles(msh_a, msh_b)

        expected = 60.0
        np.testing.assert_almost_equal(computed, expected, decimal=1)

        expected = 120.0
        np.testing.assert_almost_equal(computed_t_ang, expected, decimal=1)

    def test_case_02(self):
        """Test angle between subsections at 120°. II plane reversed normal"""

        lons = np.array([[10.0, 10.2], [10.0, 10.2]])
        lats = np.array([[45.0, 45.0], [45.0, 45.0]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_a = RectangularMesh(lons, lats, deps)

        tlo, tla, tde = npoints_towards(10.25, 45.0, 0.0, 30, 20.0, 0.0, 3)
        lons = np.array([np.flip(tlo[1:]), np.flip(tlo[1:])])
        lats = np.array([np.flip(tla[1:]), np.flip(tla[1:])])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_b = RectangularMesh(lons, lats, deps)

        computed, computed_t_ang = get_angles(msh_a, msh_b)

        expected = 60.0
        np.testing.assert_almost_equal(computed, expected, decimal=1)

        expected = 120.0
        np.testing.assert_almost_equal(computed_t_ang, expected, decimal=1)

    def test_case_03(self):
        """Test angle between subsections at 175°"""

        lons = np.array([[10.0, 10.2], [10.0, 10.2]])
        lats = np.array([[45.0, 45.0], [45.0, 45.0]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_a = RectangularMesh(lons, lats, deps)

        tlo, tla, tde = npoints_towards(10.25, 45.0, 0.0, 85, 20.0, 0.0, 3)
        lons = np.array([np.flip(tlo[1:]), np.flip(tlo[1:])])
        lats = np.array([np.flip(tla[1:]), np.flip(tla[1:])])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_b = RectangularMesh(lons, lats, deps)

        computed, computed_t_ang = get_angles(msh_a, msh_b)

        expected = 5.0
        np.testing.assert_almost_equal(computed, expected, decimal=1)

        expected = 175.0
        np.testing.assert_almost_equal(computed_t_ang, expected, decimal=1)

    def test_case_04(self):
        """Test angle between subsections at 80°"""

        # In this test the line passing through the top of trace B intersects
        # the top of trace A.

        lons = np.array([[10.0, 10.2], [10.0, 10.2]])
        lats = np.array([[45.0, 45.0], [45.0, 45.0]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_a = RectangularMesh(lons, lats, deps)

        tlo, tla, tde = npoints_towards(10.1, 45.0, 0.0, 350, 20.0, 0.0, 3)
        lons = np.array([np.flip(tlo[1:]), np.flip(tlo[1:])])
        lats = np.array([np.flip(tla[1:]), np.flip(tla[1:])])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_b = RectangularMesh(lons, lats, deps)

        computed, computed_t_ang = get_angles(msh_a, msh_b)

        expected = 80.0
        np.testing.assert_almost_equal(computed, expected, decimal=1)

        expected = 80.0
        np.testing.assert_almost_equal(computed_t_ang, expected, decimal=1)

    def test_case_05(self):
        """Test angle between parallel subsections"""

        lons = np.array([[10.0, 10.2], [10.0, 10.2]])
        lats = np.array([[45.0, 45.0], [45.0, 45.0]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_a = RectangularMesh(lons, lats, deps)

        lons = np.array([[10.0, 10.2], [10.0, 10.2]])
        lats = np.array([[45.1, 45.1], [45.1, 45.1]])
        deps = np.array([[0.0, 0.0], [10.0, 10.0]])
        msh_b = RectangularMesh(lons, lats, deps)

        computed, computed_t_ang = get_angles(msh_a, msh_b)

        expected = 0.0
        np.testing.assert_almost_equal(computed, expected, decimal=1)

        expected = 0.0
        np.testing.assert_almost_equal(computed_t_ang, expected, decimal=1)
