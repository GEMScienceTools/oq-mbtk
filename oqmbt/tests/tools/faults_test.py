import numpy
import unittest

from oqmbt.tools.faults import rates_for_double_truncated_mfd
from oqmbt.tools.mfd import mag_to_mo

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

        area = 100 # km
        slip_rate = 2 # mm
        m_low = 6.5
        m_upp = 7.5
        b_gr = 1.1
        bin_width = 0.05

        # Compute moment in Nm
        expected = (32 * 1e9) * (area * 1e6) * (slip_rate * 1e-3)

        # Compute rates and magnitudes (i.e. bin centers)
        rates = rates_for_double_truncated_mfd(area, slip_rate, m_low, m_upp,
                                               b_gr, bin_width)
        mags = numpy.arange(m_low+bin_width/2, m_upp, bin_width)

        # Compute moment from rates
        computed = sum([rate*mag_to_mo(mag) for rate, mag in zip(rates, mags)])

        # Check that the two values matches
        self.assertLess(abs(computed - expected)/expected*100., 1)
