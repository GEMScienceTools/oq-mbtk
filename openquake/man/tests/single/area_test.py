import unittest

from openquake.man.single.areas import _get_area, get_rates_density

from shapely.wkt import loads

from openquake.hazardlib.source import AreaSource
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Point, Polygon
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.pmf import PMF


class TestGetArea(unittest.TestCase):

    def setUp(self):
        self.pol = Polygon([Point(longitude=0.0, latitude=0.0),
                            Point(longitude=1.0, latitude=0.0),
                            Point(longitude=1.0, latitude=1.0),
                            Point(longitude=0.0, latitude=1.0)])

    def test01(self):
        # Initial approximated value obtained by roughly multiplying the lenght
        # of one degree of longitude time one degree of latitude
        expected = 12308.236750903827
        computed = _get_area(loads(self.pol.wkt))
        self.assertEqual(expected, computed)


class TestAreaSourceDensity(unittest.TestCase):

    def setUp(self):

        mfd = TruncatedGRMFD(min_mag=4.0, max_mag=6.0, bin_width=0.1,
                             a_val=2.0, b_val=1.0)
        msr = WC1994()
        tom = PoissonTOM(1.0)
        pol = Polygon([Point(longitude=0.0, latitude=0.0),
                       Point(longitude=1.0, latitude=0.0),
                       Point(longitude=1.0, latitude=1.0),
                       Point(longitude=0.0, latitude=1.0)])
        npd = PMF([(1.0, NodalPlane(0.0, 90.0, 0.0))])
        hpd = PMF([(0.7, 10.), (0.3, 20.0)])

        self.src1 = AreaSource(source_id='1',
                               name='1',
                               tectonic_region_type='Test',
                               mfd=mfd,
                               rupture_mesh_spacing=1,
                               magnitude_scaling_relationship=msr,
                               rupture_aspect_ratio=1.,
                               temporal_occurrence_model=tom,
                               upper_seismogenic_depth=0,
                               lower_seismogenic_depth=100.,
                               nodal_plane_distribution=npd,
                               hypocenter_distribution=hpd,
                               polygon=pol,
                               area_discretization=10.)

    def test01(self):
        # Initial value obtained by dividing the rate by the area
        # 10**(a-bmt)-10**(a-bmu) / area
        expected = 1.7567728862622177e-08
        computed = get_rates_density([self.src1], mmint=5.5)
        self.assertEqual(expected, computed['1'])
