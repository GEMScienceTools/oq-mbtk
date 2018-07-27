import numpy as np
import unittest

from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.mbt.notebooks.compare_and_report.tools import (
    get_cumulative, interpolate_ccumul)


class TestComputeCCumulative(unittest.TestCase):

    def setUp(self):
        min_mag = 6.5
        max_mag = 7.0
        bin_width = 0.1
        a_val = 3.0
        b_val = 1.0
        self.mfd = TruncatedGRMFD(min_mag, max_mag, bin_width, a_val, b_val)

    def test_get_cumulative(self):
        """
        Test the values of magnitude computed and the complementary
        cumulative rates
        """
        mac, occ = get_cumulative(self.mfd)
        expected = np.array([6.5, 6.6, 6.7, 6.8, 6.9])
        np.testing.assert_array_almost_equal(np.array(mac), expected)
        incremental = np.array([6.503912286587971e-05, 5.166241165406994e-05,
                                4.103691225077654e-05, 3.259677806669462e-05,
                                2.589254117941671e-05])
        tocc = sum(incremental)
        self.assertAlmostEqual(tocc, occ[0])
        self.assertAlmostEqual(incremental[-1], occ[-1])
        self.assertAlmostEqual(sum(incremental[-2:]), occ[3])


class TestCCumulativeInterpolation(unittest.TestCase):

    def setUp(self):
        min_mag = 6.5
        max_mag = 7.0
        bin_width = 0.1
        a_val = 3.0
        b_val = 1.0
        self.mfd = TruncatedGRMFD(min_mag, max_mag, bin_width, a_val, b_val)
        self.ms = []
        self.os = []
        #
        # loading information for the original MFD
        for mag, occ in self.mfd.get_annual_occurrence_rates():
            self.ms.append(mag)
            self.os.append(occ)
        self.os = np.array(self.os)

    def test_interpolate_01(self):
        """
        Test calculation of exceedance rate for magnitude equal to bin limit
        """
        exrate = interpolate_ccumul(self.mfd, 6.8)
        self.assertAlmostEqual(exrate, sum(self.os[-2:]))

    def test_interpolate_02(self):
        """
        Test calculation of exceedance rate for a given magnitude
        """
        exrate = interpolate_ccumul(self.mfd, 6.84)
        # rate computed by hand
        self.assertAlmostEqual(exrate, 4.5450608e-05)

    def test_interpolate_03(self):
        """
        Test calculation of exceedance rate within the last bin
        """
        exrate = interpolate_ccumul(self.mfd, 6.94)
        # rate computed by hand
        self.assertAlmostEqual(exrate, 1.285383e-05)
