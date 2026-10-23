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
import scipy.sparse as ssp

from openquake.fnm.inversion.solver import solve_nnls_pg


def _badly_scaled_system(seed=0):
    # Rows with weights spanning four orders of magnitude, as when slip-rate
    # and MFD equations are stacked
    rng = np.random.default_rng(seed)
    A = rng.random((40, 80))
    b = A @ rng.random(80) * rng.uniform(0.5, 1.5, 40)
    w = np.logspace(0, 4, 40)
    return ssp.csr_array(A), b, w


class TestNnlsAdaptiveRestart(unittest.TestCase):

    def _misfits(self, restart):
        A, b, w = _badly_scaled_system()
        _, hist = solve_nnls_pg(A, b, weights=w, max_iters=3000,
                                accept_grad=-1, accept_norm=-1,
                                stall_val=-1, restart=restart)
        return hist[hist > 0.0]

    def test_restart_damps_oscillations(self):
        # Without restart the accelerated misfit goes up and down; with
        # restart the momentum is dropped at each increase
        def n_increases(h):
            return int(np.sum(np.diff(h) > 0))
        self.assertLess(n_increases(self._misfits(True)),
                        n_increases(self._misfits(False)))

    def test_restart_does_not_worsen_the_solution(self):
        self.assertLessEqual(self._misfits(True)[-1],
                             self._misfits(False)[-1] * (1 + 1e-6))
