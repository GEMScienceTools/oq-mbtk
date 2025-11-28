import unittest

from openquake.man.checking_utils.mfds_and_rates_utils import slip_from_mo, SHEAR_MODULUS


class TestSlipFromMo(unittest.TestCase):
    def testcase01(self):
        slip = 2.0e-3
        mu = SHEAR_MODULUS
        area = 100.0e6
        mo = slip*mu*area
        computed = slip_from_mo(mo, area*1e-6)
        self.assertAlmostEqual(computed, slip)
