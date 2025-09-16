# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import geopandas as gpd
import unittest

from openquake.man.checking_utils.source_plotting_utils import get_fault_geojsons

BASE_DATA_PATH = os.path.dirname(__file__)

exp_polys_path = os.path.join(
    BASE_DATA_PATH, 'data', 'expected', 'fault_sections.geojson')


class TestPlotFaults(unittest.TestCase):

    def test_get_fault_geoJSONs(self):
        """
        Check execution of the functions which convert the fault sources
        within a hazard model into geoJSONs.

        Uses a source XML with a single SimpleFaultSource.
        """
        data_dir = os.path.join(BASE_DATA_PATH, 'data')
        results = get_fault_geojsons(data_dir, inv_time=1, rms=5)

        # Check the geoJSON contents
        obs_polys = results[0]
        exp_polys = gpd.read_file(exp_polys_path)
        obs_p_coo = obs_polys.geometry[0].exterior.coords._coords
        exp_p_coo = exp_polys.geometry[0].exterior.coords._coords
        np.testing.assert_allclose(obs_p_coo, exp_p_coo)

        # Check the geojsons exist
        assert os.path.exists(results[2]) # polys geojson
        assert os.path.exists(results[3]) # traces geojson
        
        # Then remove them
        os.remove(results[2])
        os.remove(results[3])
