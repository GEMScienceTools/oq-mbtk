"""
"""

import os
import numpy
import unittest

from openquake.hazardlib.geo import Line, Point

from openquake.sub.utils import (get_line_of_intersection,
                                      get_direction_cosines,
                                      build_complex_surface_from_edges,
                                      _check_edges)

EDGE_FOLDER = os.path.join(os.path.dirname(__file__), 'data/edges')


class TestBuildComplexFaultSurface(unittest.TestCase):

    def test_build_surface_01(self):
        """
        """
        srfc = build_complex_surface_from_edges(EDGE_FOLDER)


class TestCheckEdges(unittest.TestCase):

    def setUp(self):
        edges = []
        edge1 = Line([Point(10.0, 45.0, 0.0), Point(10., 45.1, 0.0),
                      Point(10.0, 45.2, 0.0)])
        edge2 = Line([Point(10.1, 45.0, 10.0), Point(10.1, 45.1, 10.0),
                      Point(10.1, 45.2, 10.0)])
        edges.append(edge1)
        edges.append(edge2)
        self.edgesA = edges

        edges = []
        edge1 = Line([Point(10.0, 45.0, 0.0), Point(10., 45.1, 0.0),
                      Point(10.0, 45.2, 0.0)])
        edge2 = Line([Point(10.1, 45.2, 10.0), Point(10.1, 45.1, 10.0),
                      Point(10.1, 45.0, 10.0)])
        edges.append(edge1)
        edges.append(edge2)
        self.edgesB = edges

    def test_check_01(self):
        """
        """
        computed = _check_edges(self.edgesA)
        expected = [1, 1]
        numpy.testing.assert_allclose(computed, expected, rtol=1e-07)

    def test_check_02(self):
        """
        """
        computed = _check_edges(self.edgesB)
        expected = [1, -1]
        numpy.testing.assert_allclose(computed, expected, rtol=1e-07)


class TestGetDirectionCosines(unittest.TestCase):

    def test_vertical_fault01(self):
        strike = 0.0
        dip = 90.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([1.0, 0, 0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_vertical_fault02(self):
        strike = 45.0
        dip = 90.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([0.7071068, -0.7071068, 0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_vertical_fault03(self):
        strike = 225.0
        dip = 90.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([-0.7071068, +0.7071068, 0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_fault01(self):
        strike = 0.0
        dip = 45.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([0.7071068, 0.0, 0.7071068])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_fault02(self):
        strike = -45.0
        dip = 45.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([0.5, 0.5, 0.7071068])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_fault03(self):
        strike = 210
        dip = 30.
        actual = get_direction_cosines(strike, dip)
        desired = numpy.asarray([-0.4330127, 0.25,  0.86602540])
        numpy.testing.assert_almost_equal(actual, desired)


class TestLineOfPlaneIntersection(unittest.TestCase):

    def test_vertical_faults01(self):
        strike1 = 0.0
        dip1 = 90.0
        strike2 = 90.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([0.0, 0.0, -1.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_vertical_faults02(self):
        strike1 = 0.0
        dip1 = 90.0
        strike2 = 315.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([0.0, 0.0, 1.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_plane01(self):
        strike1 = 0.0
        dip1 = 90.0
        strike2 = 0.0
        dip2 = 45.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([0.0, -1.0, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_plane02(self):
        strike1 = 45.0
        dip1 = 45.0
        strike2 = 225.0
        dip2 = 45.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([-0.7071068, -0.7071068, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)

    """
    def test_dipping_plane03(self):
        strike1 = 140.
        dip1 = 50.0
        strike2 = 30.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([-0.7071068, -0.7071068, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)

    def test_dipping_plane04(self):
        strike1 = 60.
        dip1 = 32.0
        strike2 = 30.0
        dip2 = 90.0
        actual = get_line_of_intersection(strike1, dip1, strike2, dip2)
        desired = numpy.asarray([-0.7071068, -0.7071068, 0.0])
        numpy.testing.assert_almost_equal(actual, desired)
    """
