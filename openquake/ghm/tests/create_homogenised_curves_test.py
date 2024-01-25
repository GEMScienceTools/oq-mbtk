# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
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

"""
Module create_map_test
"""

import os
import re
import shutil
import tempfile
import unittest
import numpy as np
import geopandas as gpd

from openquake.ghm.create_homogenised_curves import proc

#
HERE = os.path.dirname(__file__)


@unittest.skipUnless('GEM_DATA' in os.environ, 'please set GEMDATA')
class CreateCurvesTestCase(unittest.TestCase):
    """ Testing the calculation of hazard curves in case of a cluster """

    def test_homogenise_eur_mie(self):
        """ Testing homogenisation between mie and eur"""

        # 1 - contacts shapefile
        fname = 'contacts_between_models.shp'
        contacts_shp = os.path.join(HERE, '..', 'data', 'gis', fname)

        # 2 - output folder
        outpath = tempfile.mkdtemp()

        # 3 - folder with data
        datafolder = os.path.join(HERE, 'data', 'mosaic')

        # 4 - folder with the spatial index
        fname = 'trigrd_split_9_spacing_13'
        sidx_fname = os.path.join(os.environ['GEMDATA'], 'global_grid', fname)

        # 5 - Boundaries file
        data_folder = os.environ['GEM_DATA']
        fname = 'gadm_410_level_0.gpkg'
        boundaries_shp = os.path.join(data_folder, 'gis', 'grid', fname)

        # 6 - imt string
        imt_str = 'PGA'

        # 7 - shapefile of inland areas
        fname = 'inland.shp'
        inland_shp = boundaries_shp

        # 8 - keys of models used for the testing
        models_list = ['mie', 'eur']

        # Homogenizing
        proc(contacts_shp,          # 1
             outpath,               # 2
             datafolder,            # 3
             sidx_fname,            # 4
             boundaries_shp,        # 5
             imt_str,               # 6
             inland_shp,            # 7
             models_list,           # 8
             only_buffers=False,
             buf=35,
             h3_resolution=6,
             mosaic_key='GID_0',
             vs30_flag=False,
             overwrite=True,
             sub=True)

        print(outpath)
