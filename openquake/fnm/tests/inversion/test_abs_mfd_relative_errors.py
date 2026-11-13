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

from openquake.fnm.inversion.soe_builder import make_abs_mfd_eqns, make_eqns
from openquake.fnm.tests.inversion.simple_test_data import (
    simple_test_rups,
    simple_test_faults,
)


class TestAbsMfdRelativeErrors(unittest.TestCase):

    def setUp(self):
        self.rups = [{"M": 6.0}, {"M": 6.1}, {"M": 6.2}]
        self.mfd = {6.0: 1e-3, 6.1: 5e-4, 6.2: 1e-4}

    def test_relative_errors(self):
        _, rhs, err, meta = make_abs_mfd_eqns(
            self.rups, self.mfd, err_mode="relative", rel_err=0.2,
            weight=2.0)
        np.testing.assert_allclose(err, 2.0 / (0.2 * rhs))
        self.assertEqual(meta["details"]["err_mode"], "relative")

    def test_poisson_errors_unchanged(self):
        _, rhs, err, meta = make_abs_mfd_eqns(self.rups, self.mfd)
        np.testing.assert_allclose(err, 1.0 / np.sqrt(rhs))
        self.assertEqual(meta["details"]["err_mode"], "poisson")

    def test_invalid_err_mode(self):
        with self.assertRaises(ValueError):
            make_abs_mfd_eqns(self.rups, {6.0: 1e-3}, err_mode="foo")

    def test_make_eqns_passes_err_mode(self):
        mfd = {6.0: 1e-3, 6.5: 3e-4, 7.0: 1e-4}
        _, rhs, err, meta = make_eqns(
            simple_test_rups, simple_test_faults, mfd=mfd,
            mfd_abs_err_mode="relative", mfd_abs_rel_err=0.5,
            return_metadata=True)
        block = [m for m in meta if m["type"] == "mfd_abs"][0]
        sl = slice(block["start_idx"], block["end_idx"])
        np.testing.assert_allclose(err[sl], 1.0 / (0.5 * rhs[sl]))
