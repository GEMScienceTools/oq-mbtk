#!/usr/bin/env python
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

import os
import h3
import json
import shapely
import geopandas as gpd
from openquake.wkf.utils import create_folder, get_list


def discretize_zones_with_h3_grid(h3_level: str, fname_poly: str,
                                  folder_out: str, use: str = []):
    
    if len(use) > 0:
        use = get_list(use)
        
    h3_level = int(h3_level)
    create_folder(folder_out)

    tmp = "mapping_h{:d}.csv".format(h3_level)
    fname_out = os.path.join(folder_out, tmp)

    # Read polygons
    polygons_gdf = gpd.read_file(fname_poly)

    # Select point in polygon
    fout = open(fname_out, 'w')
    for idx, poly in polygons_gdf.iterrows():
    
        if len(use) > 0 and poly.id not in use:
            continue

        tmps = shapely.geometry.mapping(poly.geometry)
        geojson_poly = eval(json.dumps(tmps))
        if geojson_poly['type'] == 'MultiPolygon':
            from shapely.geometry import shape, mapping
            # Check that there are no polygons inside
            multipoly = shape(geojson_poly)
            assert len(multipoly.geoms) == 1
            geojson_poly = mapping(multipoly.geoms[0])

        # Revert the positions of lons and lats
        coo = [[c[1], c[0]] for c in geojson_poly['coordinates'][0]]
        geojson_poly['coordinates'] = [coo]

        # Discretizing
        hexagons = list(h3.polyfill(geojson_poly, h3_level))
        for hxg in hexagons:
            if isinstance(poly.id, str):
                fout.write("{:s},{:s}\n".format(hxg, poly.id))
            else:
                fout.write("{:s},{:d}\n".format(hxg, poly.id))

    fout.close()
