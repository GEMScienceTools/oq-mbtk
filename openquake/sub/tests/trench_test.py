
import os
import re

import numpy as np
import unittest

from openquake.sub.cross_sections import Trench

from openquake.hazardlib.geo.geodetic import distance


class TrenchDiscretizationTest(unittest.TestCase):
    """
    """

    BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

    def setUp(self):
        #
        # trench filename
        fname_trench = os.path.join(self.BASE_DATA_PATH, 'ccara_ca.xy')
        # load the axis of the trench i.e. the curve used as a reference for
        # the construction of the cross sections
        fin = open(fname_trench, 'r')
        trench = []
        for line in fin:
            aa = re.split('\s+', re.sub('^\s+', '', line))
            trench.append((float(aa[0]), float(aa[1])))
        fin.close()
        #
        # instantiate the trench object
        self.trench = Trench(np.array(trench))

    def test_resampling_01(self):
        """
        Test trench axis resampling using a distance of 20 km
        """
        #
        # resample the trench axis - output is a numpy array
        sampling = 20.
        rtrench = self.trench.resample(sampling)
        idx = len(rtrench.axis)-2
        pts = rtrench.axis
        deps = np.zeros_like(pts[0:idx, 0])
        #
        # compute distance between consecutive points
        dsts = distance(pts[0:idx, 0], pts[0:idx, 1], deps,
                        pts[1:idx+1, 0], pts[1:idx+1, 1], deps)
        expected = np.ones_like(dsts)*sampling
        #
        # check that the spacing between points corresponds to the
        # sampling distance
        np.testing.assert_allclose(dsts, expected, rtol=1, atol=0.)
