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

import numpy as np

from openquake.fnm.inversion.simulated_annealing import (
    simulated_annealing,
    weighted_lls_misfit,
)


class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        self.d = np.array([2e-3, 3e-3, 1e-3])
        self.x0 = np.ones(3) * 1e-4

    def test_misfit_type_is_used(self):
        hists = {}
        for misfit_type in ('linear', 'log'):
            _, hists[misfit_type] = simulated_annealing(
                self.A,
                self.d,
                x0=self.x0,
                max_iters=200,
                seed=3,
                accept_norm=-np.inf,
                misfit_type=misfit_type,
            )
        self.assertFalse(np.allclose(hists['linear'], hists['log']))
        # the log misfit of these residuals is negative
        self.assertLess(hists['log'][0], 0.0)

    def test_bad_misfit_type(self):
        with self.assertRaises(ValueError):
            simulated_annealing(self.A, self.d, misfit_type='quadratic')

    def test_returns_best_solution_when_accepting_worse(self):
        x, hist = simulated_annealing(
            self.A,
            self.d,
            x0=self.x0,
            max_iters=500,
            seed=7,
            initial_temp=1e-6,
            sample_with_T=True,
            accept_norm=0.0,
        )
        misfit = weighted_lls_misfit(self.A @ x, self.d, np.ones(3))
        np.testing.assert_allclose(misfit, np.min(hist), rtol=1e-9)
