
import numpy
import unittest
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.point import Point
from openquake.sce.geometry.tools import tor2trace, get_sf_geometry_xml
from openquake.sce.geometry.tools import get_sf_hypocenter


class Tor2TraceTest(unittest.TestCase):
    def testcase01(self):
        tor_trace = Line([Point(10, 45, 5), Point(10, 45.1, 5.)])
        computed = tor2trace(tor_trace, 45)
        lo_expected = numpy.array([9.936408372454517, 9.936297093030417])
        la_expected = numpy.array([44.999982355175376, 45.099982293475584])
        lo_computed = numpy.array([p.longitude for p in computed])
        la_computed = numpy.array([p.latitude for p in computed])
        numpy.testing.assert_almost_equal(lo_computed, lo_expected)
        numpy.testing.assert_almost_equal(la_computed, la_expected)


class GetSFXmlTest(unittest.TestCase):
    def testcase01(self):
        upp_sd = 5.0
        low_sd = 15.0
        mag = 6.0
        rake = -90.
        dip = 45.
        tor_trace = Line([Point(10, 45, 5), Point(10, 45.1, 5.)])
        trace = tor2trace(tor_trace, dip)
        xml = get_sf_geometry_xml(trace, upp_sd, low_sd, mag, rake, dip)
        print(xml)
