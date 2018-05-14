
import unittest
import numpy as np

from openquake.hazardlib.source import PointSource
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.geo import Point
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.const import TRT
from openquake.hazardlib.mfd import TruncatedGRMFD

from openquake.man.single.points import get_cell_areas


class TestGetGridAreas(unittest.TestCase):

    def setUp(self):
        """
        """
        #
        # coordinates of point sources
        lons = [10.00, 10.10, 10.20,
                10.00, 10.10, 10.20,
                10.00, 10.10, 10.20]
        lats = [45.10, 45.10, 45.10,
                45.05, 45.05, 45.05,
                45.00, 45.00, 45.00]
        deps = [10.00, 10.00, 10.00,
                10.00, 10.00, 10.00,
                10.00, 10.00, 10.00]
        self.lons = lons
        self.lats = lats
        #
        # set main parameters
        mfd = TruncatedGRMFD(min_mag=4.0,
                             max_mag=6.0,
                             bin_width=0.1,
                             a_val=2.0,
                             b_val=1.0)
        msr = WC1994()
        tom = PoissonTOM(1.0)
        npd = PMF([(1.0, NodalPlane(0.0, 90.0, 0.0))])
        hdd = PMF([(0.7, 4.), (0.3, 8.0)])
        #
        # create the list of sources
        srcs = []
        for idx, (lon, lat, dep) in enumerate(zip(lons, lats, deps)):
            src = PointSource(source_id='1',
                              name='Test',
                              tectonic_region_type=TRT.ACTIVE_SHALLOW_CRUST,
                              mfd=mfd,
                              rupture_mesh_spacing=5.0,
                              magnitude_scaling_relationship=msr,
                              rupture_aspect_ratio=1.0,
                              temporal_occurrence_model=tom,
                              upper_seismogenic_depth=0,
                              lower_seismogenic_depth=10.,
                              location=Point(lon, lat, dep),
                              nodal_plane_distribution=npd,
                              hypocenter_distribution=hdd)
            srcs.append(src)
        self.points = srcs

    def test01(self):
        """
        Results computed using the tools available at
        https://geographiclib.sourceforge.io/
        """
        expected = np.array([43736430.1, 43736430.1, 43736430.1,
                             43774201.0, 43774201.0, 43774201.0,
                             43811937.7, 43811937.7, 43811937.7])/1e6
        computed, _, _, _ = get_cell_areas(self.points)
        np.testing.assert_allclose(computed, expected, rtol=1e-2)
