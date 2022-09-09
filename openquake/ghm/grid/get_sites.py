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
import re
import h3
import toml
import numpy as np
import geopandas as gpd
import openquake.ghm.mosaic as mosaic

from pathlib import Path
from shapely.geometry import Polygon
from openquake.ghm.utils import explode
from openquake.baselib import sap

EXAMPLE = """
[main]

# Name of the shapefile containing the borders of countries
borders_fname = '../data/gis/world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'

# Buffer distance [m]
buffer = 50000

# Grid resolution
h3_resolution = 5
"""


def get_poly_from_str(tstr):
    """
    :param str tstr:
        A string with a sequence of lon, lat tuples
    :returns:
        A :class:`shapely.geometry.Polygon` instance
    """
    li = re.split('\\s+', tstr)
    coo = []
    for i in range(0, len(li), 2):
        coo.append([float(li[i]), float(li[i+1])])
    coo = np.array(coo)
    return Polygon(coo)


def create_query(inpt, field, labels):
    """
    :param inpt:
    :param field:
    :param labels:
    :returns:
    """
    sel = None
    for lab in labels:
        if sel is None:
            sel = inpt[field] == lab
        else:
            sel = sel | (inpt[field] == lab)
    return inpt.loc[sel]


def main(model, folder_out, fname_conf, example=False):
    """
    Tool for creating an equally spaced set of points covering a model in the
    global hazard mosaic.
    """

    # Prints an example of configuration file
    if example:
        print(EXAMPLE)
        exit(0)

    # Set model key
    conf = toml.load(fname_conf)

    in_file = conf['main']['borders_fname']
    buffer_dist = conf['main']['buffer']
    h3_resolution = conf['main']['h3_resolution']

    # Set the name of the shapefile
    # if model == 'ucerf':
    #    fname = 'cb_2017_us_state_500k.shp'
    #    print('ucerf')
    # in_file = os.path.join(inpath, fname)

    country_column = 'FIPS_CNTRY'
    country_column = 'GID_0'

    # Read polygon file
    tmpdf = gpd.read_file(in_file)
    tmpdf = create_query(tmpdf, country_column, mosaic.DATA[model])

    # Explode the geodataframe and set MODEL attribute
    inpt = explode(tmpdf)
    inpt['MODEL'] = model

    # Select polygons the countries composing the given model
    # selection = create_query(inpt, country_column, mosaic.DATA[model])
    # selection = selection.set_crs('epsg:4326')
    selection = inpt

    # Merge the polygons into a single one
    one_polygon = selection.dissolve(by='MODEL')

    # Process the polygons included in the selection. One polygon per row
    old_key = ''
    sites_indices = []
    for nrow, row in selection.iterrows():

        # Info
        key = row[country_column]
        if old_key != key:
            print(f'Processing: {key}')
            old_key = key

        # Create a list of polygons in geojson format
        for poly in [row.geometry]:

            # Create geodataframe and add a buffer
            gds = gpd.GeoSeries([poly])
            gds = gds.set_crs('epsg:4326')
            projected = gds.to_crs('epsg:3857')
            projected = projected.buffer(buffer_dist)
            gds = projected.to_crs('epsg:4326')

            # Create geojson and find the indexes of the points inside
            feature_coll = gds.__geo_interface__
            tmp = feature_coll['features'][0]['geometry']
            tidx_a = h3.polyfill_geojson(tmp, h3_resolution)

            # In this case we need to further refine the selection
            if model in mosaic.SUBSETS and key in mosaic.SUBSETS[model]:
                for tstr in mosaic.SUBSETS[model][key]:
                    tpoly = get_poly_from_str(tstr)
                    feature_coll = gpd.GeoSeries([tpoly]).__geo_interface__
                    tmp = feature_coll['features'][0]['geometry']
                    tidx_b = h3.polyfill_geojson(tmp, h3_resolution)
                    tidx_a = list(set(tidx_a) & set(tidx_b))

            sites_indices.extend(tidx_a)

    sites_indices = list(set(sites_indices))

    Path(folder_out).mkdir(parents=True, exist_ok=True)

    # Output shapefile file
    out_file = os.path.join(folder_out, f'{model}.geojson')
    one_polygon.columns = one_polygon.columns.astype(str)
    one_polygon.to_file(out_file, driver='GeoJSON')

    selection.to_file('/tmp/chk.shp')

    # Output file with grid
    sidxs = sorted(sites_indices)
    sites = np.fliplr(np.array([h3.h3_to_geo(h) for h in sidxs]))
    out_file = os.path.join(folder_out, f'{model}_res{h3_resolution}.csv')
    np.savetxt(out_file, sites, delimiter=",")


main.model = 'Model key e.g. eur'
main.folder_out = 'Name of the output folder'
main.fname_conf = 'Name of the configuration file'
msg = 'Print an example of configuration and exit'
main.example = msg

if __name__ == '__main__':
    sap.run(main)
