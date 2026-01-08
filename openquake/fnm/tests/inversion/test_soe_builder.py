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

import os
import unittest

import numpy as np

from openquake.fnm.inversion.soe_builder import (
    make_slip_rate_eqns,
    rel_gr_mfd_rates,
    make_rel_gr_mfd_eqns,
    mean_slip_rate,
    get_mag_counts,
    make_abs_mfd_eqns,
    make_slip_rate_smoothing_eqns,
    get_fault_moment,
    get_slip_rate_fraction,
    make_fault_mfd_equation_components,
    make_eqns,
    hz,
)

from openquake.fnm.inversion.utils import (
    rup_df_to_rupture_dicts,
    subsection_df_to_fault_dicts,
    get_fault_moment_rate,
)

from openquake.fnm.all_together_now import (
    build_fault_network,
    build_system_of_equations,
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
    lhs, rhs, err, _ = make_slip_rate_eqns(
        simple_test_rups, simple_test_faults
    )

    lhs = lhs.todense()

    np.testing.assert_array_almost_equal(
        lhs, np.array([[1.0, 0.0, 2.5, 1.75], [0.0, 1.0, 2.5, 1.75]])
    )

    np.testing.assert_array_almost_equal(rhs, np.array([0.001, 0.001]))


@unittest.skip("function not implemented with new methods")
def test_make_slip_rate_smoothing_eqns():
    lhs, rhs, err = make_slip_rate_smoothing_eqns(
        simple_test_fault_adjacence,
        simple_test_faults,
        rups=simple_test_rups,
    )

    lhs = lhs.todense()

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


def test_rel_gr_mfd_rates():
    rel_rates = rel_gr_mfd_rates([6.0, 6.5, 7.0], b=1.0)
    _rel_rates = {
        6.0: 1.0,
        6.5: 0.31622776601683794,
        7.0: 0.1,
    }

    for mag, rate in rel_rates.items():
        assert np.isclose(rate, _rel_rates[mag])


# @unittest.skip("Not sure of correct rates")
def test_make_rel_gr_mfd_eqns():
    lhs, rhs, err, _ = make_rel_gr_mfd_eqns(simple_test_rups, b=1.0)

    lhs = lhs.todense()

    np.testing.assert_array_almost_equal(
        lhs,
        np.array(
            [[-1.0, -1.0, 0.0, 3.16227766], [-1.0, -1.0, 10.0, 0.0]],
        ),
    )

    np.testing.assert_array_almost_equal(
        # err, np.array([1.77827941, 3.16227766])
        err,
        np.array([3.162278, 10.0]),
    )

    np.testing.assert_array_almost_equal(rhs, np.array([0.0, 0.0]))


# @unittest.skip("Not sure of correct rates")
def test_and_solve_slip_rate_and_rel_gr_eqns(inversion_tol=1e-10):
    lhs, rhs, err, _ = make_slip_rate_eqns(
        simple_test_rups, simple_test_faults
    )
    lhs = lhs.todense()
    lhs2, rhs2, err, _ = make_rel_gr_mfd_eqns(simple_test_rups, b=1.0)
    lhs2 = lhs2.todense()

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


def test_make_abs_mfd_eqns_nonnormalized_incremental():
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)

    lhs, rhs, err, _ = make_abs_mfd_eqns(
        simple_test_rups, mfd, normalize=False
    )
    lhs = lhs.todense()

    lhs_ = np.array(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )

    rhs_ = np.array([8.52639416e-04, 2.69628258e-04, 8.52639416e-05])

    np.testing.assert_array_almost_equal(lhs, lhs_)
    np.testing.assert_array_almost_equal(rhs, rhs_)


def test_make_abs_mfd_eqns_nonnormalized_cumulative():
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)

    lhs, rhs, err, _ = make_abs_mfd_eqns(
        simple_test_rups, mfd, normalize=False, cumulative=True
    )
    lhs = lhs.todense()

    lhs_ = np.array(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )

    rhs_ = np.array([0.00410418, 0.00126951, 0.00037311])

    np.testing.assert_array_almost_equal(lhs, lhs_)
    np.testing.assert_array_almost_equal(rhs, rhs_)


def test_make_abs_mfd_eqns_normalized():
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)

    lhs, rhs, err, _ = make_abs_mfd_eqns(simple_test_rups, mfd, normalize=True)
    lhs = lhs.todense()

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
    lhs, rhs, err, _ = make_slip_rate_eqns(
        simple_test_rups, simple_test_faults
    )
    lhs = lhs.todense()
    mfd = hz.mfd.TruncatedGRMFD(5.0, 8.0, 0.1, 3.61759073, 1.0)
    lhs2, rhs2, err, _ = make_abs_mfd_eqns(simple_test_rups, mfd)
    lhs2 = lhs2.todense()

    lhs = np.vstack([lhs, lhs2])
    rhs = np.hstack([rhs, rhs2])

    soln = np.linalg.lstsq(lhs, rhs, rcond=-1)[0]

    resids = lhs @ soln - rhs


def test_make_abs_mfd_eqns_faults():
    total_mfd = hz.mfd.TruncatedGRMFD(5.0, 7.1, 0.1, 3.61759073, 1.0)
    f0_mfd = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 7.1, 0.1, 1.0, total_mfd._get_total_moment_rate() / 2
    )
    f1_mfd = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 7.1, 0.1, 1.0, total_mfd._get_total_moment_rate() / 2
    )

    fault_mfds = {
        'f1': {
            'mfd': f0_mfd,
            'rups_include': [0, 2, 3],
            'rup_fractions': [1.0, 0.5, 0.5],
        },
        'f2': {
            'mfd': f1_mfd,
            'rups_include': [1, 2, 3],
            'rup_fractions': [1.0, 0.5, 0.5],
        },
    }

    lhs0, rhs0, err0, _ = make_abs_mfd_eqns(
        simple_test_rups,
        fault_mfds['f1']['mfd'],
        rup_include_list=fault_mfds['f1']['rups_include'],
        rup_fractions=fault_mfds['f1']['rup_fractions'],
    )
    lhs0 = lhs0.todense()
    lhs1, rhs1, err1, _ = make_abs_mfd_eqns(
        simple_test_rups,
        fault_mfds['f2']['mfd'],
        rup_include_list=fault_mfds['f2']['rups_include'],
        rup_fractions=fault_mfds['f2']['rup_fractions'],
    )
    lhs1 = lhs1.todense()

    np.testing.assert_equal(
        lhs0,
        np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.5, 0.0]]
        ),
    )
    np.testing.assert_allclose(
        rhs0, np.array([3.46094034e-04, 1.09444543e-04, 3.46094034e-05])
    )

    np.testing.assert_equal(
        lhs1,
        np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.5, 0.0]]
        ),
    )

    np.testing.assert_allclose(
        rhs1, np.array([3.46094034e-04, 1.09444543e-04, 3.46094034e-05])
    )

    lhsm = np.vstack((lhs0, lhs1))
    rhsm = np.hstack((rhs0, rhs1))

    np.testing.assert_equal(
        lhsm,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        rhsm,
        np.array(
            [
                3.46094034e-04,
                1.09444543e-04,
                3.46094034e-05,
                3.46094034e-04,
                1.09444543e-04,
                3.46094034e-05,
            ]
        ),
    )


def test_make_eqns_fault_mfds_only():
    total_mfd = hz.mfd.TruncatedGRMFD(5.0, 7.1, 0.1, 3.61759073, 1.0)
    f0_mfd = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 7.1, 0.1, 1.0, total_mfd._get_total_moment_rate() / 2
    )
    f1_mfd = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 7.1, 0.1, 1.0, total_mfd._get_total_moment_rate() / 2
    )

    fault_mfds = {
        'f0': {
            'mfd': f0_mfd,
            'rups_include': [0, 2, 3],
            'rup_fractions': [1.0, 0.5, 0.5],
        },
        'f1': {
            'mfd': f1_mfd,
            'rups_include': [1, 2, 3],
            'rup_fractions': [1.0, 0.5, 0.5],
        },
    }
    lhs, rhs, err = make_eqns(
        simple_test_rups,
        faults=None,
        mfd=None,
        slip_rate_eqns=None,
        fault_abs_mfds=fault_mfds,
        return_sparse=False,
    )

    np.testing.assert_equal(
        lhs,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        rhs,
        np.array(
            [
                3.46094034e-04,
                1.09444543e-04,
                3.46094034e-05,
                3.46094034e-04,
                1.09444543e-04,
                3.46094034e-05,
            ]
        ),
    )


def test_make_eqns_abs_and_fault_mfds():
    total_mfd = hz.mfd.TruncatedGRMFD(5.0, 7.1, 0.1, 3.61759073, 1.0)
    f0_mfd = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 7.1, 0.1, 1.0, total_mfd._get_total_moment_rate() / 2
    )
    f1_mfd = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 7.1, 0.1, 1.0, total_mfd._get_total_moment_rate() / 2
    )

    fault_mfds = {
        'f0': {
            'mfd': f0_mfd,
            'rups_include': [0, 2, 3],
            'rup_fractions': [1.0, 0.5, 0.5],
        },
        'f1': {
            'mfd': f1_mfd,
            'rups_include': [1, 2, 3],
            'rup_fractions': [1.0, 0.5, 0.5],
        },
    }

    lhs, rhs, err = make_eqns(
        simple_test_rups,
        faults=None,
        mfd=total_mfd,
        slip_rate_eqns=None,
        fault_abs_mfds=fault_mfds,
        return_sparse=False,
    )
    np.testing.assert_equal(
        lhs,
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        rhs,
        np.array(
            [
                8.52639416e-04,
                2.69628258e-04,
                8.52639416e-05,
                3.46094034e-04,
                1.09444543e-04,
                3.46094034e-05,
                3.46094034e-04,
                1.09444543e-04,
                3.46094034e-05,
            ]
        ),
    )


class TestEqnsFromLilFaults(unittest.TestCase):
    def setUp(self):

        TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
        FAULT_FILE = os.path.join(TEST_DATA_DIR, "lil_test_faults.geojson")

        settings = {
            "subsection_size": [12.0, 10.0],
            "lower_seis_depth": 10.0,
            "calculate_rates_from_slip_rates": True,
            "filter_by_plausibility": False,
            "export_fault_mfds": True,
        }

        self.fault_network = build_fault_network(
            fault_geojson=FAULT_FILE, settings=settings
        )

        self.fault_network['subfault_df']['moment'] = self.fault_network[
            'subfault_df'
        ].apply(get_fault_moment_rate, axis=1)

        self.rups = rup_df_to_rupture_dicts(
            self.fault_network['rupture_df'],
            mag_col="mag",
            displacement_col="displacement",
        )

        self.faults = subsection_df_to_fault_dicts(
            self.fault_network["subfault_df"],
            slip_rate_col="net_slip_rate",
            slip_rate_err_col="net_slip_rate_err",
        )

    def test_make_equations_just_slip_rates(self):
        lhs, rhs, err = make_eqns(
            rups=self.rups,
            faults=self.faults,
            return_sparse=False,
        )

        np.testing.assert_almost_equal(
            lhs,
            np.array(
                [
                    [0.397, 0.559, 0.0, 0.0, 0.564],
                    [0.0, 0.559, 0.397, 0.0, 0.564],
                    [0.0, 0.0, 0.0, 0.351, 0.564],
                ]
            ),
        )

        np.testing.assert_almost_equal(rhs, np.array([0.001, 0.001, 0.001]))

        np.testing.assert_almost_equal(
            err, np.array([2000.0, 2000.0, 10000.0])
        )

    def test_make_fault_mfd_equation_components_no_scale(self):

        fault_abs_mfds = make_fault_mfd_equation_components(
            self.fault_network['fault_mfds'],
            self.rups,
            self.fault_network,
            fault_key='subfaults',
            rup_key='rupture_df',
            seismic_slip_rate_frac=1.0,
        )

        fault_abs_mfds_correct = {
            0: {
                'mfd': {
                    5.05: 0.0012835881512141293,
                    5.1499999999999995: 0.001019590310266925,
                    5.249999999999999: 0.0008098893712963095,
                    5.349999999999999: 0.0006433179946237562,
                    5.449999999999998: 0.0005110056470358534,
                    5.549999999999998: 0.0004059062135441289,
                    5.649999999999998: 0.00032242276606812554,
                    5.749999999999997: 0.0002561095066058142,
                    5.849999999999997: 0.00020343501227830345,
                    5.949999999999997: 0.00016159417418413742,
                    6.049999999999996: 0.00012835881512141402,
                    6.149999999999996: 0.0001019590310266933,
                    6.249999999999996: 8.098893712963159e-05,
                    6.349999999999995: 6.433179946237612e-05,
                    6.449999999999995: 5.110056470358578e-05,
                },
                'rups_include': [0, 1, 4],
                'rup_fractions': [1.0, 0.4997, 0.3569],
            },
            1: {
                'mfd': {
                    5.05: 0.0012851614241037607,
                    5.1499999999999995: 0.001020840005344084,
                    5.249999999999999: 0.0008108820393808937,
                    5.349999999999999: 0.0006441064989110507,
                    5.449999999999998: 0.0005116319782544529,
                    5.549999999999998: 0.0004064037261153517,
                    5.649999999999998: 0.00032281795435057815,
                    5.749999999999997: 0.0002564234158165986,
                    5.849999999999997: 0.00020368435922756922,
                    5.949999999999997: 0.00016179223750618173,
                    6.049999999999996: 0.00012851614241037707,
                    6.149999999999996: 0.00010208400053440928,
                    6.249999999999996: 8.108820393808999e-05,
                    6.349999999999995: 6.441064989110565e-05,
                    6.449999999999995: 5.116319782544568e-05,
                },
                'rups_include': [1, 2, 4],
                'rup_fractions': [0.5003, 1.0, 0.3573],
            },
            2: {
                'mfd': {
                    5.05: 0.0010281237538428759,
                    5.1499999999999995: 0.0008166677264681136,
                    5.249999999999999: 0.0006487022335217099,
                    5.349999999999999: 0.0005152825000149987,
                    5.449999999999998: 0.0004093034386212287,
                    5.549999999999998: 0.0003251212778665792,
                    5.649999999999998: 0.00025825301071906274,
                    5.749999999999997: 0.00020513765811670413,
                    5.849999999999997: 0.000162946633847315,
                    5.949999999999997: 0.00012943311201820166,
                    6.049999999999996: 0.00010281237538428841,
                    6.149999999999996: 8.166677264681202e-05,
                    6.249999999999996: 6.487022335217152e-05,
                    6.349999999999995: 5.152825000150028e-05,
                    6.449999999999995: 4.093034386212321e-05,
                },
                'rups_include': [3, 4],
                'rup_fractions': [1.0, 0.2858],
            },
        }

        for fault_key, mfd_stuff in fault_abs_mfds.items():
            for key, test_value in mfd_stuff.items():
                if key == 'mfd':
                    np.testing.assert_almost_equal(
                        np.array(
                            sorted(fault_abs_mfds[fault_key][key].keys())
                        ),
                        np.array(
                            sorted(
                                fault_abs_mfds_correct[fault_key][key].keys()
                            )
                        ),
                    )
                    np.testing.assert_almost_equal(
                        np.array(
                            sorted(fault_abs_mfds[fault_key][key].values())
                        ),
                        np.array(
                            sorted(
                                fault_abs_mfds_correct[fault_key][key].values()
                            )
                        ),
                    )
                else:
                    assert (
                        fault_abs_mfds[fault_key][key]
                        == fault_abs_mfds_correct[fault_key][key]
                    )

    def test_make_equations_from_fault_mfds(self):

        total_fault_moment = self.fault_network['subfault_df']['moment'].sum()

        fault_abs_mfds = make_fault_mfd_equation_components(
            self.fault_network['fault_mfds'],
            self.rups,
            self.fault_network,
            fault_key='subfaults',
            rup_key='rupture_df',
            seismic_slip_rate_frac=1.0,
        )

        lhs, rhs, err = make_eqns(
            rups=self.rups,
            faults=None,
            slip_rate_eqns=False,
            mfd=None,
            fault_abs_mfds=fault_abs_mfds,
            return_sparse=False,
        )

        np.testing.assert_almost_equal(
            lhs,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.4997, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.3569],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.5003, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.3573],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.2858],
                ]
            ),
        ),

        np.testing.assert_almost_equal(
            rhs,
            np.array(
                [
                    1.28358815e-04,
                    1.01959031e-04,
                    5.11005647e-05,
                    0.00000000e00,
                    1.28516142e-04,
                    1.02084001e-04,
                    5.11631978e-05,
                    0.00000000e00,
                    1.02812375e-04,
                    8.16667726e-05,
                    4.09303439e-05,
                    0.00000000e00,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            err,
            np.array(
                [
                    8.82647205e01,
                    9.90346453e01,
                    1.39890155e02,
                    1.00000000e10,
                    8.82106779e01,
                    9.89740084e01,
                    1.39804503e02,
                    1.00000000e10,
                    9.86227943e01,
                    1.10656595e02,
                    1.56306595e02,
                    1.00000000e10,
                ]
            ),
            decimal=3,
        )

    def test_make_equations_from_fault_abs_mfds(self):
        total_fault_moment = self.fault_network['subfault_df']['moment'].sum()
        total_abs_mfd = hz.mfd.TruncatedGRMFD.from_moment(
            min_mag=5.9,
            max_mag=6.6,
            bin_width=0.1,
            b_val=1.0,
            moment_rate=total_fault_moment,
        )

        fault_abs_mfds = make_fault_mfd_equation_components(
            self.fault_network['fault_mfds'],
            self.rups,
            self.fault_network,
            fault_key='subfaults',
            rup_key='rupture_df',
            seismic_slip_rate_frac=1.0,
        )

        lhs, rhs, err = make_eqns(
            rups=self.rups,
            faults=None,
            slip_rate_eqns=False,
            mfd=total_abs_mfd,
            fault_abs_mfds=fault_abs_mfds,
            return_sparse=False,
        )

        np.testing.assert_almost_equal(
            lhs,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.4997, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.3569],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.5003, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.3573],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.2858],
                ]
            ),
        )

        np.testing.assert_almost_equal(
            rhs,
            np.array(
                [
                    4.57959562e-04,
                    3.63770211e-04,
                    1.82316986e-04,
                    1.44819529e-04,
                    1.28358815e-04,
                    1.01959031e-04,
                    5.11005647e-05,
                    0.00000000e00,
                    1.28516142e-04,
                    1.02084001e-04,
                    5.11631978e-05,
                    0.00000000e00,
                    1.02812375e-04,
                    8.16667726e-05,
                    4.09303439e-05,
                    0.00000000e00,
                ]
            ),
            decimal=3,
        )

        np.testing.assert_almost_equal(
            err,
            np.array(
                [
                    4.67289943e01,
                    5.24307939e01,
                    7.40604649e01,
                    8.30972084e01,
                    8.82647205e01,
                    9.90346453e01,
                    1.39890155e02,
                    1.00000000e10,
                    8.82106779e01,
                    9.89740084e01,
                    1.39804503e02,
                    1.00000000e10,
                    9.86227943e01,
                    1.10656595e02,
                    1.56306595e02,
                    1.00000000e10,
                ]
            ),
            decimal=3,
        )

    def test_mean_slip_rate(self):
        msr = mean_slip_rate(self.rups[4]['faults'], self.faults)
        np.testing.assert_approx_equal(msr, 1.0)

    def test_get_fault_moment(self):
        fault_moment = get_fault_moment(self.faults)
        np.testing.assert_approx_equal(
            fault_moment, 1.1186199511831996e16, significant=4
        )

    def test_get_slip_rate_fraction(self):
        fault_moment = get_fault_moment(self.faults)
        print(fault_moment)
        total_abs_mfd = hz.mfd.TaperedGRMFD.from_moment(
            min_mag=5.9,
            max_mag=6.6,
            corner_mag=6.3,
            bin_width=0.1,
            b_val=1.0,
            moment_rate=fault_moment,
        )

        np.testing.assert_approx_equal(
            get_slip_rate_fraction(self.faults, total_abs_mfd),
            1.0,
            significant=3,
        )


# this is for reference
rups = [
    {
        'idx': 0,
        'M': 6.1,
        'D': 0.397,
        'faults': [0],
        'faults_orig': {'f1': 1.0},
        'subfault_fracs': {0: 1.0},
    },
    {
        'idx': 1,
        'M': 6.4,
        'D': 0.559,
        'faults': [0, 1],
        'faults_orig': {'f1': 1.0},
        'subfault_fracs': {0: 0.4997, 1: 0.5003},
    },
    {
        'idx': 2,
        'M': 6.1,
        'D': 0.397,
        'faults': [1],
        'faults_orig': {'f1': 1.0},
        'subfault_fracs': {1: 1.0},
    },
    {
        'idx': 3,
        'M': 6.0,
        'D': 0.351,
        'faults': [2],
        'faults_orig': {'f2': 1.0},
        'subfault_fracs': {2: 1.0},
    },
    {
        'idx': 4,
        'M': 6.5,
        'D': 0.564,
        'faults': [0, 1, 2],
        'faults_orig': {'f1': 0.7, 'f2': 0.3},
        'subfault_fracs': {0: 0.3569, 1: 0.3573, 2: 0.2858},
    },
]

faults = [
    {
        'id': 0,
        'slip_rate': 1.0,
        'slip_rate_err': 0.5,
        'trace': [
            [-122.6737, 45.48704, 0.0],
            [-122.69758583405802, 45.520564357112974, 0.0],
            [-122.72921077819535, 45.55061628429187, 0.0],
            [-122.762795159138, 45.5797933832881, 0.0],
        ],
        'area': 124.74786985561391,
    },
    {
        'id': 1,
        'slip_rate': 1.0,
        'slip_rate_err': 0.5,
        'trace': [
            [-122.762795159138, 45.5797933832881, 0.0],
            [-122.79641974866986, 45.60895764186604, 0.0],
            [-122.83010988063222, 45.63809472923184, 0.0],
            [-122.86391574284539, 45.66717618379236, 0.0],
        ],
        'area': 124.90077126834305,
    },
    {
        'id': 2,
        'slip_rate': 1.0,
        'slip_rate_err': 0.1,
        'trace': [
            [-122.51594000000001, 45.47618, 0.0],
            [-122.58006299150225, 45.47668892736773, 0.0],
            [-122.64418710170465, 45.477161977220895, 0.0],
        ],
        'area': 99.9200936207929,
    },
]
