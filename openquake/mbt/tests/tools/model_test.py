import numpy
import unittest

from openquake.man.checking_utils.source_model_utils import _split_point_source

from openquake.hazardlib.source import PointSource
from openquake.hazardlib.mfd import TruncatedGRMFD, EvenlyDiscretizedMFD
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.pmf import PMF


class TestSplitSources(unittest.TestCase):

    def setUp(self):

        mfd = TruncatedGRMFD(min_mag=4.0,
                             max_mag=6.0,
                             bin_width=0.1,
                             a_val=2.0,
                             b_val=1.0)
        msr = WC1994()
        tom = PoissonTOM(1.0)
        loc = Point(longitude=0.0,
                    latitude=0.0)
        npd = PMF([(1.0, NodalPlane(0.0, 90.0, 0.0))])
        hpd = PMF([(0.7, 10.), (0.3, 20.0)])

        self.src1 = PointSource(source_id='1',
                                name='1',
                                tectonic_region_type='Test',
                                mfd=mfd,
                                rupture_mesh_spacing=1,
                                magnitude_scaling_relationship=msr,
                                rupture_aspect_ratio=1.,
                                temporal_occurrence_model=tom,
                                upper_seismogenic_depth=0,
                                lower_seismogenic_depth=100.,
                                location=loc,
                                nodal_plane_distribution=npd,
                                hypocenter_distribution=hpd)

        mfd = EvenlyDiscretizedMFD(min_mag=4.0,
                                   bin_width=0.1,
                                   occurrence_rates=[3., 2., 1.])

        self.src2 = PointSource(source_id='1',
                                name='1',
                                tectonic_region_type='Test',
                                mfd=mfd,
                                rupture_mesh_spacing=1,
                                magnitude_scaling_relationship=msr,
                                rupture_aspect_ratio=1.,
                                temporal_occurrence_model=tom,
                                upper_seismogenic_depth=0,
                                lower_seismogenic_depth=100.,
                                location=loc,
                                nodal_plane_distribution=npd,
                                hypocenter_distribution=hpd)

    def test01(self):
        """
        Check the a value
        """
        srcl = _split_point_source(self.src1)
        self.assertEqual(srcl[0].mfd.a_val, 1.8450980400142569)
        self.assertEqual(srcl[1].mfd.a_val, 1.4771212547196624)

    def test02(self):
        """
        Check occurrence rates
        """
        srcl = _split_point_source(self.src2)
        tmp = numpy.array(srcl[0].mfd.get_annual_occurrence_rates())
        com = numpy.array([r[1] for r in tmp])
        print(com)
        exp = numpy.array([2.1, 1.4, 0.7])
        numpy.testing.assert_allclose(com, exp, rtol=1e-5, atol=0)
