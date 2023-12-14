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
import json
import pathlib
import unittest
import numpy as np

from openquake.fnm.once_more_with_feeling import (
    simple_fault_from_feature,
    subdivide_simple_fault_surface,
    subdivide_rupture_mesh,
    get_subsections_from_fault,
    make_subfault_df,
    group_subfaults_by_fault,
    make_rupture_df,
    get_boundary_3d,
    make_subfault_gdf,
    make_rupture_gdf,
    merge_meshes,
    make_mesh_from_subfaults,
    make_sf_rupture_mesh,
    make_sf_rupture_meshes,
)

HERE = pathlib.Path(__file__).parent.absolute()


class Test3Faults(unittest.TestCase):
    def setUp(self):
        test_data_dir = HERE / 'data'
        fgj_name = os.path.join(test_data_dir, "motagua_3_faults.geojson")

        with open(fgj_name) as f:
            fgj = json.load(f)
        self.features = fgj['features']

    def test_simple_fault_from_feature(self):
        fault = simple_fault_from_feature(
            self.features[0], edge_sd=2.0, lsd_default=20.0, usd_default=0.0
        )
