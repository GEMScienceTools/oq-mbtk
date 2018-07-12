
import os
import re
import numpy
import unittest

from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.sub.cross_sections import CrossSection, CrossSectionData
from openquake.sub.cross_sections import get_min_distance

from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.line import Line


tmp = 'data/crust/crust_except.xyz'
CRUST_DATA_PATH = os.path.join(os.path.dirname(__file__), tmp)
tmp_idl = 'data/crust/crust_except_idl.xyz'
CRUST_DATA_PATH_IDL = os.path.join(os.path.dirname(__file__), tmp_idl)


def _get_data(filename):
    datal = []
    for line in open(filename, 'r'):
        xx = re.split('\s+', re.sub('\s+$', '', re.sub('^\s+', '', line)))
        datal.append([float(val) for val in xx])
    return datal


class GetCrustalModelTest(unittest.TestCase):

    def test_nesw_cross_section(self):
        cs = CrossSection(45.0, 45.0, [100], [45])
        csd = CrossSectionData(cs)
        csd.set_crust1pt0_moho_depth(CRUST_DATA_PATH, bffer=200.)
        expected = [[44.5, 46.5], [45.5, 45.5], [45.5, 44.5], [46.5, 44.5]]
        numpy.testing.assert_equal(expected, csd.moho[:, 0:2])

    def test_ns_cross_section(self):
        cs = CrossSection(45.0, 45.0, [100], [0])
        csd = CrossSectionData(cs)
        csd.set_crust1pt0_moho_depth(CRUST_DATA_PATH, bffer=200.)
        expected = [[43.5, 45.5], [44.5, 45.5], [45.5, 45.5], [46.5, 45.5]]
        numpy.testing.assert_equal(expected, csd.moho[:, 0:2])

    def test_idl_cross_section(self):
        cs = CrossSection(-179.0, -50.0, [200], [-90])
        csd = CrossSectionData(cs)
        csd.set_crust1pt0_moho_depth(CRUST_DATA_PATH_IDL, bffer=100.)
        expected = [[-179.5,-49.5],[-179.5,-50.5],[178.5,-49.5],[179.5,-49.5],[178.5,-50.5],[179.5,-50.5]]
        numpy.testing.assert_equal(expected,csd.moho[:, 0:2])


def _get_data(filename):
    datal = []
    for line in open(filename, 'r'):
        xx = re.split('\s+', re.sub('\s+$', '', re.sub('^\s+', '', line)))
        datal.append([float(val) for val in xx])
    return datal


class GetCrustalModelTest(unittest.TestCase):

    def test_nesw_cross_section(self):
        cs = CrossSection(45.0, 45.0, [100], [45])
        csd = CrossSectionData(cs)
        csd.set_crust1pt0_moho_depth(CRUST_DATA_PATH, bffer=200.)
        expected = [[44.5, 46.5], [45.5, 45.5], [45.5, 44.5], [46.5, 44.5]]
        numpy.testing.assert_equal(expected, csd.moho[:, 0:2])

    def test_ns_cross_section(self):
        cs = CrossSection(45.0, 45.0, [100], [0])
        csd = CrossSectionData(cs)
        csd.set_crust1pt0_moho_depth(CRUST_DATA_PATH, bffer=200.)
        expected = [[43.5, 45.5], [44.5, 45.5], [45.5, 45.5], [46.5, 45.5]]
        numpy.testing.assert_equal(expected, csd.moho[:, 0:2])


class GetMMTest(unittest.TestCase):

    def test_simple_cs(self):
        """
        Test simple cross section
        """
        cs = CrossSection(10.0, 45.0, [100], [45])
        computed = cs.get_mm()
        expected = [cs.plo[0], cs.plo[1], cs.pla[0], cs.pla[1], 0]
        numpy.testing.assert_equal(computed, expected)

    def test_cs_across_idl(self):
        """
        Test cross section across idl
        """
        cs = CrossSection(-179.0, -50.0, [500], [-90])
        computed = cs.get_mm()
        expected = [cs.plo[0], cs.plo[1], cs.pla[0], cs.pla[1], 1]
        numpy.testing.assert_equal(computed, expected)

    def test_cs_across_idl_with_delta(self):
        """
        Test cross section across idl + delta
        """
        cs = CrossSection(-179.5, -50.0, [200], [90])
        computed = cs.get_mm(1.0)
        expected = [179.5, -175.70311203864779, -51.0, -48.966369263787726, 1]
        numpy.testing.assert_equal(computed, expected)


class MinDistTest(unittest.TestCase):

    def setUp(self):
        pass

    def test01(self):
        """
        TODO this test is not complete
        """
        lons = [
         10.0, 11.0, 12.0]
        lats = [45.0, 44.0, 43.0]
        lons = [10.0, 11.0]
        lats = [45.0, 46.0]
        pnts = numpy.array([[10.3, 45.4], [9.8, 45.4],
                            [10.3, 44.8], [9.8, 44.8]])
        pnts = numpy.array([[10.3, 45.0], [9.0, 45.0]])
        line = Line([Point(lo, la) for lo, la in zip(lons, lats)])
        get_min_distance(line, pnts)
