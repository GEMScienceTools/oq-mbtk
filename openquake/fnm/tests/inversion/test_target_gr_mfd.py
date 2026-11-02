# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import unittest

import numpy as np

from openquake.hazardlib.mfd.tapered_gr_mfd import mag_to_mo
from openquake.fnm.inversion.utils import (
    get_mfd_occurrence_rates,
    make_target_gr_mfd,
)


class TestTargetGrMfd(unittest.TestCase):

    def test_target_gr_mfd_aligned_and_moment_balanced(self):
        mags = np.array([5.91, 6.0, 6.3, 6.74, 7.2])
        moment_rate = 1e16
        mfd = make_target_gr_mfd(mags, moment_rate, b_val=1.0)
        rates = get_mfd_occurrence_rates(mfd, mag_decimals=1)
        # One bin per 0.1 between the rounded min and max magnitudes
        expected = [round(5.9 + 0.1 * i, 1) for i in range(14)]
        self.assertEqual(sorted(rates), expected)
        # Rates decrease by 10**(-b * 0.1) per bin
        r = np.array([rates[m] for m in expected])
        np.testing.assert_allclose(r[1:] / r[:-1], 10 ** -0.1)
        moment = sum(mag_to_mo(m) * v for m, v in rates.items())
        np.testing.assert_allclose(moment, moment_rate, rtol=1e-9)
