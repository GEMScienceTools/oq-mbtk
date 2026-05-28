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

import os
import unittest

import numpy as np
import scipy.sparse as ssp

from openquake.fnm.inversion.solver import (
    get_obs_equalization_weights,
    solve_nnls_pg,
    weight_from_error,
    weights_from_errors,
)
from openquake.fnm.inversion.soe_builder import (
    make_eqns,
    make_fault_rel_mfd_equation_components,
)
from openquake.fnm.inversion.utils import (
    get_fault_moment_rate,
    rup_df_to_rupture_dicts,
    subsection_df_to_fault_dicts,
)
from openquake.fnm.all_together_now import build_fault_network


# ---------------------------------------------------------------------------
# weight_from_error
# ---------------------------------------------------------------------------


def test_weight_from_error_nan_uses_zero_error():
    w = weight_from_error(np.nan, zero_error=2.0)
    assert np.isfinite(w)
    assert w == 0.5


def test_weight_from_error_inf_is_zero_weight():
    w = weight_from_error(np.inf)
    assert np.isfinite(w)
    assert w == 0.0


def test_weight_from_error_normal_value():
    assert weight_from_error(2.0) == 0.5


def test_weight_from_error_zero_without_zero_error_uses_min_error():
    assert weight_from_error(0.0, min_error=0.5) == 2.0


def test_weight_from_error_below_min_error_is_clamped():
    assert weight_from_error(1e-15, min_error=1.0) == 1.0


def test_weight_from_error_max_weight_cap():
    assert weight_from_error(0.001, max_weight=10.0) == 10.0


def test_weight_from_error_nan_without_zero_error_uses_min_error():
    assert weight_from_error(np.nan, min_error=0.01) == 100.0


# ---------------------------------------------------------------------------
# weights_from_errors
# ---------------------------------------------------------------------------


def test_weights_from_errors_nan_vector_no_nans():
    w = weights_from_errors([np.nan, 0.0, 1.0], zero_error=1.0, min_error=1e-6)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0])


def test_weights_from_errors_basic_reciprocals():
    w = weights_from_errors([1.0, 2.0, 4.0])
    np.testing.assert_allclose(w, [1.0, 0.5, 0.25])


def test_weights_from_errors_lil_faults_slip_rate_errors():
    # err vector returned by make_eqns slip-rate only on lil_test_faults
    w = weights_from_errors([2000.0, 2000.0, 10000.0])
    np.testing.assert_allclose(w, [5e-4, 5e-4, 1e-4])


# ---------------------------------------------------------------------------
# get_obs_equalization_weights
# ---------------------------------------------------------------------------


def test_get_obs_equalization_weights_uniform():
    w = get_obs_equalization_weights(np.ones(5) * 3.0)
    np.testing.assert_allclose(w, np.ones(5) * 3.0)


def test_get_obs_equalization_weights_zero_replaced_by_eps():
    w = get_obs_equalization_weights(np.array([0.0, 1.0, 2.0]), eps=0.5)
    np.testing.assert_allclose(w, [0.5, 1.0, 2.0])


def test_get_obs_equalization_weights_auto_eps_is_min_abs():
    w = get_obs_equalization_weights(np.array([0.1, 1.0, 2.0]))
    np.testing.assert_allclose(w, [0.1, 1.0, 2.0])


# ---------------------------------------------------------------------------
# solve_nnls_pg – synthetic cases with analytic solutions
# ---------------------------------------------------------------------------


def test_solve_nnls_pg_identity_system():
    A = ssp.eye(3, format="csr")
    b = np.array([1.0, 2.0, 3.0])
    x, _ = solve_nnls_pg(A, b, max_iters=10000, accept_norm=1e-12, accept_grad=1e-10)
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0], atol=1e-6)


def test_solve_nnls_pg_identity_nonneg_constraint_clips_negative():
    # b[1] < 0 so the non-negativity constraint is active; optimal x[1] = 0
    A = ssp.eye(3, format="csr")
    b = np.array([2.0, -1.0, 3.0])
    x, _ = solve_nnls_pg(A, b, max_iters=5000, accept_norm=1e-12, accept_grad=1e-10)
    np.testing.assert_allclose(x, [2.0, 0.0, 3.0], atol=1e-6)


def test_solve_nnls_pg_identity_nonneg_residual():
    A = ssp.eye(3, format="csr")
    b = np.array([2.0, -1.0, 3.0])
    x, _ = solve_nnls_pg(A, b, max_iters=5000, accept_norm=1e-12, accept_grad=1e-10)
    np.testing.assert_almost_equal(np.linalg.norm(A @ x - b), 1.0, decimal=6)


# ---------------------------------------------------------------------------
# solve_nnls_pg – real problem from lil_test_faults
# ---------------------------------------------------------------------------


class TestSolveNnlsPgFromLilFaults(unittest.TestCase):
    def setUp(self):
        TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        FAULT_FILE = os.path.join(TEST_DATA_DIR, "lil_test_faults.geojson")

        settings = {
            "subsection_size": [12.0, 10.0],
            "lower_seis_depth": 10.0,
            "calculate_rates_from_slip_rates": True,
            "filter_by_plausibility": False,
            "export_fault_mfds": True,
            "parallel_subfault_build": False,
        }

        self.fault_network = build_fault_network(
            fault_geojson=FAULT_FILE, settings=settings
        )
        self.fault_network["subfault_df"]["moment"] = self.fault_network[
            "subfault_df"
        ].apply(get_fault_moment_rate, axis=1)

        self.rups = rup_df_to_rupture_dicts(
            self.fault_network["rupture_df"],
            mag_col="mag",
            displacement_col="displacement",
        )
        self.faults = subsection_df_to_fault_dicts(
            self.fault_network["subfault_df"],
            slip_rate_col="net_slip_rate",
            slip_rate_err_col="net_slip_rate_err",
        )

    def test_slip_rate_system_solution_nonneg(self):
        A, b, err = make_eqns(
            rups=self.rups,
            faults=self.faults,
            slip_rate_eqns=True,
            return_sparse=True,
        )
        w = weights_from_errors(err)
        x, _ = solve_nnls_pg(
            A, b, weights=w, max_iters=20000, accept_norm=1e-14, accept_grad=1e-10
        )
        assert np.all(x >= 0.0)

    def test_slip_rate_system_solution_values(self):
        A, b, err = make_eqns(
            rups=self.rups,
            faults=self.faults,
            slip_rate_eqns=True,
            return_sparse=True,
        )
        w = weights_from_errors(err)
        x, _ = solve_nnls_pg(
            A, b, weights=w, max_iters=20000, accept_norm=1e-14, accept_grad=1e-10
        )
        np.testing.assert_allclose(
            x,
            np.array([2.773e-4, 7.810e-4, 2.773e-4, 1.519e-5, 8.124e-4]),
            rtol=1e-2,
        )

    def test_slip_rate_system_weighted_residual_norm(self):
        A, b, err = make_eqns(
            rups=self.rups,
            faults=self.faults,
            slip_rate_eqns=True,
            return_sparse=True,
        )
        w = weights_from_errors(err)
        Aw = ssp.diags(w) @ A
        bw = b * w
        x, _ = solve_nnls_pg(
            A, b, weights=w, max_iters=20000, accept_norm=1e-14, accept_grad=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(Aw @ x - bw), 5.376e-8, rtol=1e-2
        )

    def test_full_system_solution_nonneg(self):
        fault_rel_mfds = make_fault_rel_mfd_equation_components(
            self.rups,
            self.fault_network,
            fault_key="subfaults",
            rup_key="rupture_df",
            full_counting=False,
        )
        A, b, err = make_eqns(
            rups=self.rups,
            faults=self.faults,
            slip_rate_eqns=True,
            fault_rel_mfds=fault_rel_mfds,
            return_sparse=True,
        )
        w = weights_from_errors(err)
        x, _ = solve_nnls_pg(
            A, b, weights=w, max_iters=20000, accept_norm=1e-14, accept_grad=1e-10
        )
        assert np.all(x >= 0.0)

    def test_full_system_weighted_residual_norm(self):
        fault_rel_mfds = make_fault_rel_mfd_equation_components(
            self.rups,
            self.fault_network,
            fault_key="subfaults",
            rup_key="rupture_df",
            full_counting=False,
        )
        A, b, err = make_eqns(
            rups=self.rups,
            faults=self.faults,
            slip_rate_eqns=True,
            fault_rel_mfds=fault_rel_mfds,
            return_sparse=True,
        )
        w = weights_from_errors(err)
        Aw = ssp.diags(w) @ A
        bw = b * w
        x, _ = solve_nnls_pg(
            A, b, weights=w, max_iters=20000, accept_norm=1e-14, accept_grad=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(Aw @ x - bw), 7.141e-7, rtol=1e-2
        )

