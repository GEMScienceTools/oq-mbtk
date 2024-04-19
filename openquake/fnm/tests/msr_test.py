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

import numpy as np
import unittest
from openquake.fnm.msr import area_to_mag
from openquake.fnm.section import split_into_subsections
from openquake.fnm.rupture import get_ruptures_section
from openquake.fnm.rupture import _get_rupture_area
from openquake.fnm.tests.connection_test import _get_surfs


class TestGetMagnitude(unittest.TestCase):

    def test_get_magnitude(self):
        """Test magnitude calculation """

        surfs = _get_surfs()

        nc_strike = 12
        nc_dip = -1

        mesh = surfs[0].mesh
        tmp_ul = split_into_subsections(mesh, nc_strike, nc_dip)
        rups = get_ruptures_section(tmp_ul)

        # Get the rupture surfaces
        area0 = _get_rupture_area(surfs, rups[rups[:, 4] == 0, :])
        area1 = _get_rupture_area(surfs, rups[rups[:, 4] == 1, :])
        area2 = _get_rupture_area(surfs, rups[rups[:, 4] == 2, :])

        # Compute magnitudes
        mag0 = area_to_mag(area0)
        mag1 = area_to_mag(area1)
        mag2 = area_to_mag(area2)

        expected = np.array([6.156638, 6.458269, 6.157839])
        expected = np.array([6.158363, 6.459393, 6.158363])
        computed = np.array([mag0, mag1, mag2])

        np.testing.assert_allclose(expected, computed)
