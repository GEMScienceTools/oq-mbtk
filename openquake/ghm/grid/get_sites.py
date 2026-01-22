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
import sys
from pathlib import Path

import h3
import toml
import numpy as np
import geopandas as gpd
import openquake.ghm.mosaic as mosaic

from shapely.geometry import Polygon
from openquake.ghm.utils import explode
from openquake.baselib import sap

INFO = False

EXAMPLE = """
[main]

# Name of the shapefile containing the borders of countries
# borders_fname = '../data/gis/world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
borders_fname = '/Users/mpagani/Documents/2022/diary/08/19_remapping_mosaic/gadm_410_level_0.gpkg'

# This can be either 'FIPS_CNTRY' or 'GID_0' depending on the shapefile
country_column = 'FIPS_CNTRY'

# Buffer distance [m]
buffer = 50000

# Grid resolution
h3_resolution = 6

[site_model]

# This file is usually quite big and, therefore, stored locally. Path must be adjusted accordingly
ncfile = "/Users/mpagani/Repos/gem-hazard-data/vs30/global_vs30.grd"
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

    # Prints an example of configuration file
    if example:
        print(EXAMPLE)
        sys.exit(0)

    # Set model key
    conf = toml.load(fname_conf)
    root_path = Path(fname_conf).parents[0]

    # Getting the coordinates of the sites
    sites, sites_indices, one_polygon, selection = _get_sites(
        model, folder_out, conf, root_path)

    Path(folder_out).mkdir(parents=True, exist_ok=True)

    # Output shapefile file
    out_file = os.path.join(folder_out, f'{model}.geojson')
    one_polygon.columns = one_polygon.columns.astype(str)
    one_polygon.to_file(out_file, driver='GeoJSON')
    check_file = os.path.join(folder_out, 'check.shp')
    selection.to_file(check_file)

    # Params
    h3_resolution = conf['main']['h3_resolution']

    # Output file with grid
    out_file = os.path.join(folder_out, f'{model}_res{h3_resolution}.csv')
    np.savetxt(out_file, sites, delimiter=",")


def _get_sites(model, folder_out, conf, crs= 'epsg:3857', root_path=''):
    """
    Tool for creating an equally spaced set of points covering a model in the
    global hazard mosaic.

    param crs: 
        optional crs, useful at high/low latitude where 3857 breaks down
        select an alternative from epsg.io


    :param root_path:
        The path to the file with the confifuration i.e. the file used as a
        reference for specifying the relative paths in the configuration
        dictionary
    """
    if not os.path.isabs(conf['main']['borders_fname']):
        in_file = os.path.join(root_path, conf['main']['borders_fname'])
    else:
        in_file = conf['main']['borders_fname']

    buffer_dist = conf['main']['buffer']
    h3_resolution = conf['main']['h3_resolution']

    # Set the name of the shapefile
    # if model == 'ucerf':
    #    fname = 'cb_2017_us_state_500k.shp'
    #    print('ucerf')
    # in_file = os.path.join(inpath, fname)

    country_column = conf['main']['country_column']
    SUBSETS = mosaic.SUBSETS[country_column]
    DATA = mosaic.DATA[country_column]

    # Read polygon file
    tmpdf = gpd.read_file(in_file)
    tmpdf = create_query(tmpdf, country_column, DATA[model])

    # Explode the geodataframe and set MODEL attribute
    # inpt = explode(tmpdf)
    inpt = tmpdf.explode(index_parts=True)
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
        if old_key != key and INFO:
            print(f'Processing: {key}')
            old_key = key

        # Create a list of polygons in geojson format
        for poly in [row.geometry]:

            # Create geodataframe and add a buffer
            gds = gpd.GeoSeries([poly])
            gds = gds.set_crs('epsg:4326')
            projected = gds.to_crs(crs)
            projected = projected.buffer(buffer_dist)
            # This returns a geodataseries that we convert to geodataframe
            gds = projected.to_crs('epsg:4326')
            gdata = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gds))

            # Create geojson and find the indexes of the points inside
            eee = gds.explode(index_parts=True)
            feature_coll = eee.__geo_interface__
            tmp = feature_coll['features'][0]['geometry']
            try:
                tidx_a = h3.polyfill_geojson(tmp, h3_resolution)
            except:
                breakpoint()

            # In this case we need to further refine the selection
            if model in SUBSETS and key in SUBSETS[model]:
                for tstr in SUBSETS[model][key]:
                    tpoly = get_poly_from_str(tstr)
                    pol = gpd.GeoDataFrame(index=[0], crs='epsg:4326',
                                           geometry=[tpoly])
                    # Intersection between the geodataframe filled with points
                    # and the area defining a subportion of a nation
                    intsc = gpd.sjoin(pol, gdata, predicate='intersects')
                    if len(intsc) < 1:
                        continue
                    # Select points
                    feature_coll = gpd.GeoSeries([tpoly]).__geo_interface__
                    tmp = feature_coll['features'][0]['geometry']
                    tidx_b = h3.polyfill_geojson(tmp, h3_resolution)
                    tidx_c = list(set(tidx_a) & set(tidx_b))
                    sites_indices.extend(tidx_c)
            else:
                sites_indices.extend(tidx_a)

    sites_indices = list(set(sites_indices))
    sidxs = sorted(sites_indices)
    tmp = np.array([h3.h3_to_geo(h) for h in sidxs])
    sites = np.fliplr(tmp)

    return sites, sites_indices, one_polygon, selection


main.model = 'Model key e.g. eur'
main.folder_out = 'Name of the output folder'
main.fname_conf = 'Name of the configuration file'
msg = 'Print an example of configuration and exit'
main.example = msg

if __name__ == '__main__':
    sap.run(main)
