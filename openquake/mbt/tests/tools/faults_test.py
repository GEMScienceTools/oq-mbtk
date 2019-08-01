import numpy
import unittest

from openquake.mbt.tools.faults import rates_for_double_truncated_mfd
from openquake.mbt.tools.faults import _make_range

from openquake.mbt.tools.mfd import mag_to_mo


class RatesDoubleTruncatedFromSlipTestCase(unittest.TestCase):
    """
    This class tests the calculation of occurrence rates for a discrete double
    truncated Gutenberg-Richter magnitude-frequency distribution.
    """

    def testcase01(self):
        """
        First test case comparing the seismic moment computed using its
        classical definition and the scalar seismic moment computed from
        the discrete magnitude-frequency distribution.
        """
        area = 100  # km
        slip_rate = 2  # mm
        m_low = 6.5
        m_upp = 7.5
        b_gr = 1.1
        bin_width = 0.05
        # to pass test
        m_min = 6.5

        # Compute moment in Nm
        expected = (32 * 1e9) * (area * 1e6) * (slip_rate * 1e-3)

        # Compute rates and magnitudes (i.e. bin centers)
        rates = rates_for_double_truncated_mfd(area, slip_rate, m_low, m_upp, m_min,
                                               b_gr, bin_width)

        mags = numpy.arange(m_low+bin_width/2, m_upp, bin_width)

        # Compute moment from rates
        computed = sum([rate*mag_to_mo(mag) for rate, mag in zip(rates, mags)])

        # Check that the two values matches
        self.assertLess(abs(computed - expected)/expected*100., 1)




class TestMomentReleaseRateNonUniformBinEdge(unittest.TestCase):
    """
    This class tests that moment release rates on the fault equal
    the moment accumulation rate even if M_max doesn't initally
    fall on a bin edge
    """

    def setUp(self):

        self.area = 315
        self.slip_rate = 0.2
        self.m_low = 6.5
        self.b_gr = 1.0
        self.bin_width = 0.1
        self.rigidity = 32e9

        self.moment_accum_rate = (self.area * 1e6 * self.slip_rate * 1e-3
                                  * self.rigidity)

    def test_moment_release_rate(self):

        for _M_max in numpy.arange(6.501, 6.501, 8.501, 0.01):

            bin_rates = rates_for_double_truncated_mfd(self.area,
                                                       self.slip_rate,
                                                       self.m_low,
                                                       self.m_min,
                                                       _M_max,
                                                       self.b_gr,
                                                       self.bin_width)

            mags = [mag + self.bin_width / 2. for mag in
                    _make_range(self.m_low, _M_max, self.bin_width)]
            release_rate = sum([rate * mag_to_mo(mag)
                                for rate, mag in zip(bin_rates, mags)])

            release_rate_error = abs((self.moment_accum_rate - release_rate)
                                     / self.moment_accum_rate * 100)



            self.assertLess(release_rate_error, 1)
