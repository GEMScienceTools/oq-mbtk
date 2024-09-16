# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import unittest

import pytest
import numpy as np

from openquake.fnm.inversion.soe_builder import (
    make_slip_rate_eqns,
    get_mag_counts,
    rel_gr_mfd_rates,
    make_rel_gr_mfd_eqns,
    make_abs_mfd_eqns,
    make_slip_rate_smoothing_eqns,
    hz,
)

from openquake.fnm.tests.inversion.simple_test_data import (
    rup_A,
    rup_B,
    rup_C,
    rup_D,
    f1,
    f2,
    simple_test_rups,
    simple_test_faults,
    simple_test_fault_adjacence,
)


def test_make_slip_rate_eqns():
    lhs, rhs, err = make_slip_rate_eqns(simple_test_rups, simple_test_faults)

    np.testing.assert_array_almost_equal(
        lhs, np.array([[1.0, 0.0, 2.5, 1.75], [0.0, 1.0, 2.5, 1.75]])
    )

    np.testing.assert_array_almost_equal(rhs, np.array([0.001, 0.001]))


def test_make_slip_rate_smoothing_eqns():
    lhs, rhs, err = make_slip_rate_smoothing_eqns(
        simple_test_fault_adjacence,
        simple_test_faults,
        rups=simple_test_rups,
    )

    np.testing.assert_array_almost_equal(
        lhs,
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
            ]
        ),
    )

    # np.testing.assert_array_almost_equal(rhs, np.array([0.0, 0.0, 0.0, 0.0]))


def test_get_mag_counts():
    mag_counts = get_mag_counts(simple_test_rups)

    assert mag_counts == {6.0: 2, 7.0: 1, 6.5: 1}


@unittest.skip("Not sure of correct rates")
def test_rel_gr_mfd_rates():
    rel_rates = rel_gr_mfd_rates([6.0, 6.5, 7.0], b=1.0)
    _rel_rates = {
        6.0: 1.0,
        6.5: 0.0038451338418645682,
        7.0: 0.09999999999999999,
    }

    for mag, rate in rel_rates.items():
        assert np.isclose(rate, _rel_rates[mag])


@unittest.skip("Not sure of correct rates")
def test_make_rel_gr_mfd_eqns():
    lhs, rhs, err = make_rel_gr_mfd_eqns(simple_test_rups, b=1.0)

    np.testing.assert_array_almost_equal(
        lhs,
        np.array(
            [[-1.0, -1.0, 0.0, 3.16227766], [-1.0, -1.0, 10.0, 0.0]],
        ),
    )

    np.testing.assert_array_almost_equal(
        err, np.array([1.77827941, 3.16227766])
    )

    np.testing.assert_array_almost_equal(rhs, np.array([0.0, 0.0]))


@unittest.skip("Not sure of correct rates")
def test_and_solve_slip_rate_and_rel_gr_eqns(inversion_tol=1e-10):
    lhs, rhs, err = make_slip_rate_eqns(simple_test_rups, simple_test_faults)
    lhs2, rhs2, err = make_rel_gr_mfd_eqns(simple_test_rups, b=1.0)

    lhs = np.vstack([lhs, lhs2])
    rhs = np.hstack([rhs, rhs2])

    soln = np.linalg.solve(lhs, rhs)
    np.testing.assert_array_almost_equal(
        soln,
        np.array(
            [3.83612506e-04, 3.83612506e-04, 7.67225013e-05, 2.42617852e-04]
        ),
    )

    resids = lhs @ soln - rhs
    for resid in resids:
        assert np.isclose(resid, 0.0, atol=inversion_tol)


def test_make_abs_mfd_eqns_nonnormalized():
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)

    lhs, rhs, err = make_abs_mfd_eqns(simple_test_rups, mfd, normalize=False)

    lhs_ = np.array(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )

    rhs_ = np.array([8.52639416e-04, 2.69628258e-04, 8.52639416e-05])

    np.testing.assert_array_almost_equal(lhs, lhs_)
    np.testing.assert_array_almost_equal(rhs, rhs_)


def test_make_abs_fault_mfd_eqns_nonnormalized_cumulative():
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)

    lhs, rhs, err = make_abs_mfd_eqns(
        simple_test_rups, mfd, normalize=False, cumulative=True
    )

    lhs_ = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )

    rhs_ = np.array([1.20753162e-03, 3.54892200e-04, 8.52639416e-05])

    np.testing.assert_array_almost_equal(lhs, lhs_)
    np.testing.assert_array_almost_equal(rhs, rhs_)


def test_make_abs_mfd_eqns_normalized():
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)

    lhs, rhs, err = make_abs_mfd_eqns(simple_test_rups, mfd, normalize=True)

    lhs_ = np.array(
        [
            [3708.810078, 3708.810078, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3708.810078],
            [0.0, 0.0, 3708.810078, 0.0],
        ]
    )

    rhs_ = np.array([3.162278, 1.0, 0.316228])

    np.testing.assert_array_almost_equal(lhs, lhs_)
    np.testing.assert_array_almost_equal(rhs, rhs_)


def test_and_solve_slip_rate_and_abs_mfd_eqns():
    lhs, rhs, err = make_slip_rate_eqns(simple_test_rups, simple_test_faults)
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)
    lhs2, rhs2, err = make_abs_mfd_eqns(simple_test_rups, mfd)

    lhs = np.vstack([lhs, lhs2])
    rhs = np.hstack([rhs, rhs2])

    soln = np.linalg.lstsq(lhs, rhs, rcond=-1)[0]

    resids = lhs @ soln - rhs
