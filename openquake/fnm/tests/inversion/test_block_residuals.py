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

from openquake.fnm.inversion.soe_builder import make_eqns
from openquake.fnm.inversion.utils import get_block_residuals
from openquake.fnm.tests.inversion.simple_test_data import (
    simple_test_rups,
    simple_test_faults,
)


class TestBlockResiduals(unittest.TestCase):

    def test_blocks_cover_all_rows(self):
        mfd = {6.0: 1e-3, 6.5: 3e-4, 7.0: 1e-4}
        lhs, rhs, err, meta = make_eqns(
            simple_test_rups, simple_test_faults, mfd=mfd,
            return_metadata=True)
        x = np.full(lhs.shape[1], 1e-4)
        df = get_block_residuals(lhs, rhs, err, x, meta)
        self.assertEqual(set(df.index), {"slip_rate", "mfd_abs"})
        self.assertEqual(df.n_eqs.sum(), lhs.shape[0])
        res = (lhs @ x - rhs) * err
        n_slip = df.loc["slip_rate", "n_eqs"]
        np.testing.assert_allclose(df.loc["slip_rate", "max_abs"],
                                   np.abs(res[:n_slip]).max())
        np.testing.assert_allclose(df.loc["mfd_abs", "rms"],
                                   np.sqrt(np.mean(res[n_slip:] ** 2)))
