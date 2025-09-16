import os
import numpy
import unittest

from openquake.hazardlib.source import PointSource
from openquake.hazardlib.mfd import TruncatedGRMFD, EvenlyDiscretizedMFD
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.pmf import PMF

from openquake.man.checking_utils import source_model_utils as model


BASE_DATA_PATH = os.path.dirname(__file__)


class TestReadModel(unittest.TestCase):

    def test_read_source_model(self):
        """ read simple source model """
        fname = os.path.join(
            BASE_DATA_PATH, 'data_source_model_utils_test', 'model', 'source_model.xml')
        srcs, _ = model.read(fname)
        self.assertEqual(len(srcs), 1)


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
        srcl = model._split_point_source(self.src1)
        self.assertEqual(srcl[0].mfd.a_val, 1.8450980400142569)
        self.assertEqual(srcl[1].mfd.a_val, 1.4771212547196624)

    def test02(self):
        srcl = model._split_point_source(self.src2)
        
        com = numpy.array(srcl[0].mfd.occurrence_rates)
        exp = numpy.array(self.src2.mfd.occurrence_rates)*0.7
        numpy.testing.assert_array_equal(com, exp)
        
        com = numpy.array(srcl[1].mfd.occurrence_rates)
        exp = numpy.array(self.src2.mfd.occurrence_rates)*0.3
        numpy.testing.assert_array_equal(com, exp)
