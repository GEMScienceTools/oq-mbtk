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
from scipy.optimize import nnls

from openquake.fnm.inversion.solver import solve_nnls_pg


def _badly_scaled_system(seed=0):
    # Rows with weights spanning four orders of magnitude, as when slip-rate
    # and MFD equations are stacked
    rng = np.random.default_rng(seed)
    A = rng.random((40, 80))
    b = A @ rng.random(80) * rng.uniform(0.5, 1.5, 40)
    w = np.logspace(0, 4, 40)
    return ssp.csr_array(A), b, w


class TestNnlsStepSize(unittest.TestCase):

    def test_step_size_with_clustered_spectrum(self):
        # One singular value at 1 and many at sqrt(0.5): a few power
        # iterations underestimate the Lipschitz constant, the step is too
        # large and the accelerated iterations oscillate without converging
        n = 300
        s = np.full(n, np.sqrt(0.5))
        s[0] = 1.0
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = ssp.csr_array(Q * s)
        b = A @ (rng.random(n) + 0.1)
        x, _ = solve_nnls_pg(A, b, max_iters=5000, accept_grad=-1,
                             accept_norm=-1, stall_val=-1)
        self.assertLess(np.linalg.norm(A @ x - b), 1e-8)

    def test_badly_scaled_matches_scipy_nnls(self):
        A, b, w = _badly_scaled_system()
        x, _ = solve_nnls_pg(A, b, weights=w, max_iters=50000,
                             accept_grad=-1, accept_norm=-1, stall_val=1e-12)
        Aw = A.toarray() * w[:, None]
        _, ref = nnls(Aw, b * w, maxiter=5000)
        misfit = np.linalg.norm(Aw @ x - b * w)
        self.assertLess(misfit, ref * (1 + 1e-3) + 1e-9)

    def test_reproducible(self):
        A, b, w = _badly_scaled_system()
        x1, h1 = solve_nnls_pg(A, b, weights=w, max_iters=2000)
        x2, h2 = solve_nnls_pg(A, b, weights=w, max_iters=2000)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(h1, h2)
