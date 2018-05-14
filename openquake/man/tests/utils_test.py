import unittest

import openquake.man.utils as utils


class TestSlipFromMo(unittest.TestCase):
    def testcase01(self):
        slip = 2.0e-3
        mu = utils.SHEAR_MODULUS
        area = 100.0e6
        mo = slip*mu*area
        computed = utils.slip_from_mo(mo, area*1e-6)
        self.assertAlmostEqual(computed, slip)
