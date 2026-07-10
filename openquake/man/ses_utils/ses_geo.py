# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation and Électricité de France
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
#
# This script is produced within the scope of Work Package 5, named Simulation 
# platform, under SIGMA3 project. For more detailed information about 
# the project, please visit to https://sigma-programs.com/.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
module :mod:`openquake.man.ses_utils.ses_geo` provides a function for 
parsing geospatial data files (e.g., GeoJSON, Shapefile) and converting 
their geometries into OQ Engine polygon instances.
"""

import pathlib
import numpy as np
import geopandas as gpd

from openquake.hazardlib.geo import Polygon, Point

def get_oq_polygons(fname: str):
    """
    Returns a list of OQ Engine Polygon :class:`openquake.hazardlib.geo.Polygon`
    instances

    :param fname:
        The name of a shapefile or a geojson containing one polygon
    """

    if not pathlib.Path(fname).exists():
        raise IOError(f'File {fname} does not exists')

    gdf = gpd.read_file(fname)
    polys = []
    for i_row, row in gdf.iterrows():
        coo = np.array([c for c in row.geometry.exterior.coords])
        polys.append(Polygon([Point(p[0], p[1]) for p in coo]))
    return polys
