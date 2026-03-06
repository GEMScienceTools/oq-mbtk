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

import numpy as np

from openquake.fnm.inversion.solver import weight_from_error, weights_from_errors


def test_weight_from_error_nan_uses_zero_error():
    w = weight_from_error(np.nan, zero_error=2.0)
    assert np.isfinite(w)
    assert w == 0.5


def test_weight_from_error_inf_is_zero_weight():
    w = weight_from_error(np.inf)
    assert np.isfinite(w)
    assert w == 0.0


def test_weights_from_errors_nan_vector_no_nans():
    w = weights_from_errors([np.nan, 0.0, 1.0], zero_error=1.0, min_error=1e-6)
    assert np.all(np.isfinite(w))
    # nan -> zero_error -> 1.0; 0.0 -> zero_error -> 1.0; 1.0 -> 1.0
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0])

