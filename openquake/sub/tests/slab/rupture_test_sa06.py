"""
:module:`openquake.sub.tests.slab.rupture_test_sa06`
"""

import os
import unittest
import numpy as np

from openquake.sub.slab.rupture import create_ruptures
from openquake.hazardlib.scalerel.strasser2010 import StrasserIntraslab

BASE_DATA_PATH = os.path.dirname(__file__)


class RuptureCreationSATest(unittest.TestCase):
    """
    Test the construction of ruptures
    """

    def setUp(self):
        path = os.path.join('..', 'data', 'misc', 'top_mesh')
        x = np.loadtxt(os.path.join(path, 'top_mesh.x'))
        y = np.loadtxt(os.path.join(path, 'top_mesh.y'))
        z = np.loadtxt(os.path.join(path, 'top_mesh.z'))
        self.mesh = np.stack((x, y, z), 2)

    """
    def test(self):
        mmin = 6.0
        mmax = 7.0
        binw = 0.1
        agr = 4.0
        bgr = 1.0
        mfd = TruncatedGRMFD(mmin, mmax, binw, agr, bgr)
        msr = StrasserIntraslab()
        dips = [45]
        sampling = 5
        asprs = {1: 1.0}
        float_strike = 1
        float_dip = 1
        r =
        values =
        ohs =
        hdf5_filename = os.path.join(BASE_DATA_PATH, '..', 'tmp')
        uniform_fraction = 
        proj =
        idx = False
        align = False
        create_ruptures(mfd, dips, sampling, msr, asprs, float_strike,
                        float_dip, r, values, ohs, 1., hdf5_filename,
                        uniform_fraction, proj, idl, align)
    """
