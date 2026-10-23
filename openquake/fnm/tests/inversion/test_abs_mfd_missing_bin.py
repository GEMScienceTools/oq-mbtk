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

from openquake.fnm.inversion.soe_builder import make_abs_mfd_eqns


class TestAbsMfdMissingBin(unittest.TestCase):

    def setUp(self):
        self.rups = [{"M": 6.0}, {"M": 6.1}, {"M": 6.2}]

    def test_warns_on_missing_bin(self):
        mfd = {6.0: 1e-3, 6.2: 1e-4}
        with self.assertLogs(level=logging.WARNING) as cm:
            make_abs_mfd_eqns(self.rups, mfd)
        self.assertIn("6.1", "".join(cm.output))

    def test_excluded_ruptures_ignored(self):
        # the rupture with M 6.1 is not included in these equations
        mfd = {6.0: 1e-3, 6.2: 1e-4}
        with self.assertNoLogs(level=logging.WARNING):
            make_abs_mfd_eqns(self.rups, mfd, rup_include_list=[0, 2])
