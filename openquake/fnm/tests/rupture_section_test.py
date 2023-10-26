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
from openquake.fnm.rupture import get_ruptures_section


class TestRupturesOneSection(unittest.TestCase):

    def test_from_set_sections(self):

        tmp = np.array([[[0.,  0., 6., 12.],
                         [0.,  6., 6., 12.],
                         [0., 12., 6., 12.],
                         [0., 18., 6., 12.]]])
        rups = get_ruptures_section(tmp)
        expected = np.array([[0,  0,  6, 12, 0, 1, 0, 0],
                             [0,  0, 12, 12, 1, 1, 0, 1],
                             [0,  0, 18, 12, 2, 1, 0, 2],
                             [0,  0, 24, 12, 3, 1, 0, 3],
                             [0,  6,  6, 12, 4, 1, 0, 4],
                             [0,  6, 12, 12, 5, 1, 0, 5],
                             [0,  6, 18, 12, 6, 1, 0, 6],
                             [0, 12,  6, 12, 7, 1, 0, 7],
                             [0, 12, 12, 12, 8, 1, 0, 8],
                             [0, 18,  6, 12, 9, 1, 0, 9]])
        np.testing.assert_array_equal(rups, expected)
