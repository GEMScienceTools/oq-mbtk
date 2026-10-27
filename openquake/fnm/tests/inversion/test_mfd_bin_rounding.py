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

import logging
import unittest

from openquake.fnm.inversion.utils import get_mfd_occurrence_rates


class TestMfdBinRounding(unittest.TestCase):

    def test_collapsed_bins_are_summed(self):
        with self.assertLogs(level=logging.WARNING):
            rates = get_mfd_occurrence_rates({5.96: 1.0, 6.04: 2.0},
                                             mag_decimals=1)
        self.assertEqual(rates, {6.0: 3.0})

    def test_distinct_bins_unchanged(self):
        rates = get_mfd_occurrence_rates({6.01: 1.0, 6.1: 2.0},
                                         mag_decimals=1)
        self.assertEqual(rates, {6.0: 1.0, 6.1: 2.0})
