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
import pathlib
import unittest


from openquake.fnm.inversion.utils import (
    geom_from_fault_trace,
    project_faults_and_polies,
    lines_in_polygon,
    get_rupture_displacement,
    weighted_mean,
    slip_vector_azimuth,
    check_fault_in_poly,
    faults_in_polies,
    get_rup_poly_fracs,
    rup_df_to_rupture_dicts,
    subsection_df_to_fault_dicts,
    get_rupture_regions,
    _nearest,
    make_fault_mfd,
    get_mag_counts,
    get_mfd_occurrence_rates,
    set_single_fault_rupture_rates_by_mfd,
    set_single_fault_rup_rates,
    _get_surface_moment_rate,
    get_fault_moment_rate,
    _get_fault_by_id,
    get_ruptures_on_fault,
    get_rup_rates_from_fault_slip_rates,
)

from openquake.fnm.all_together_now import build_fault_network

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

HERE = pathlib.Path(__file__).parent.absolute()


def test_get_mag_counts():
    rups = [rup_A, rup_B, rup_C, rup_D]
    mag_counts_default = get_mag_counts(rups)
    assert mag_counts_default == {6.0: 2, 6.5: 1, 7.0: 1}

    mag_counts_incremental = get_mag_counts(rups, incremental=True)
    assert mag_counts_incremental == {6.0: 2, 6.5: 1, 7.0: 1}

    mag_counts_cumulative = get_mag_counts(rups, cumulative=True)
    assert mag_counts_cumulative == {6.0: 2, 6.5: 3, 7.0: 4}


class Test3Faults(unittest.TestCase):
    def setUp(self):
        test_data_dir = HERE / ".." / 'data'
        fgj_name = os.path.join(test_data_dir, "motagua_3_faults.geojson")
