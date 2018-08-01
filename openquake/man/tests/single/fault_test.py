import numpy as np
import unittest

from openquake.man.single.faults import fault_surface_distance

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Line, Point


class TestFaultSurfaceDistance(unittest.TestCase):

    def setUp(self):
        """
        """
        mspa = 2.5
        #
        # Simple fault source
        mfd = EvenlyDiscretizedMFD(6.0, 0.1, [1.])
        msr = WC1994()
        tom = PoissonTOM(1.0)
        trace = Line([Point(10, 45.), Point(10., 45.2)])
        sfs = SimpleFaultSource(source_id='1',
                                name='1',
                                tectonic_region_type='none',
                                mfd=mfd,
                                rupture_mesh_spacing=mspa,
                                magnitude_scaling_relationship=msr,
                                rupture_aspect_ratio=1.0,
                                temporal_occurrence_model=tom,
                                upper_seismogenic_depth=0.,
                                lower_seismogenic_depth=10.,
                                fault_trace=trace,
                                dip=90.,
                                rake=90)
        self.srcs = [sfs]
        #
        #
        mfd = EvenlyDiscretizedMFD(6.0, 0.1, [1.])
        msr = WC1994()
        tom = PoissonTOM(1.0)
        trace = Line([Point(10.2, 45.), Point(10.2, 45.2)])
        sfs = SimpleFaultSource(source_id='2',
                                name='2',
                                tectonic_region_type='none',
                                mfd=mfd,
                                rupture_mesh_spacing=mspa,
                                magnitude_scaling_relationship=msr,
                                rupture_aspect_ratio=1.0,
                                temporal_occurrence_model=tom,
                                upper_seismogenic_depth=0.,
                                lower_seismogenic_depth=10.,
                                fault_trace=trace,
                                dip=90.,
                                rake=90)
        self.srcs.append(sfs)
        #
        #
        mfd = EvenlyDiscretizedMFD(6.0, 0.1, [1.])
        msr = WC1994()
        tom = PoissonTOM(1.0)
        trace = Line([Point(10.4, 45.), Point(10.4, 45.2)])
        sfs = SimpleFaultSource(source_id='3',
                                name='3',
                                tectonic_region_type='none',
                                mfd=mfd,
                                rupture_mesh_spacing=mspa,
                                magnitude_scaling_relationship=msr,
                                rupture_aspect_ratio=1.0,
                                temporal_occurrence_model=tom,
                                upper_seismogenic_depth=0.,
                                lower_seismogenic_depth=10.,
                                fault_trace=trace,
                                dip=90.,
                                rake=90)
        self.srcs.append(sfs)
        #
        #
        mfd = EvenlyDiscretizedMFD(6.0, 0.1, [1.])
        msr = WC1994()
        tom = PoissonTOM(1.0)
        trace = Line([Point(10.5, 45.), Point(10.6, 45.2)])
        sfs = SimpleFaultSource(source_id='4',
                                name='4',
                                tectonic_region_type='none',
                                mfd=mfd,
                                rupture_mesh_spacing=mspa,
                                magnitude_scaling_relationship=msr,
                                rupture_aspect_ratio=1.0,
                                temporal_occurrence_model=tom,
                                upper_seismogenic_depth=0.,
                                lower_seismogenic_depth=10.,
                                fault_trace=trace,
                                dip=90.,
                                rake=90)
        self.srcs.append(sfs)

    def test01(self):
        """
        """
        computed = fault_surface_distance(self.srcs, 5.0)
        expected = np.array([
            [0., 15.67589056, 31.35175709, 39.313281],
            [40.,  0., 15.67589056, 23.587993],
            [40., 40.,  0.,  7.862668],
            [40., 40., 40.,  0.]])
        np.testing.assert_allclose(computed, expected, rtol=1e-2)
