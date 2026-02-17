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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import json
import h3
import toml
import shapely
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from openquake.wkf.utils import get_list
from openquake.wkf.utils import create_folder


def _get_rates(geohashes, a_value):

    # Get coordinates and compute area [km2] of each cell
    area = np.array([h3.cell_area(idx) for idx in geohashes])

    # Compute the occurrence rate per km2 from a_gr and b_gr
    numm = 10**(a_value)

    # Compute the output a_value in each cell
    a_cell = np.log10(numm * area)

    return a_cell, area


def create_missing(geohashes, a_value, b_value, a_cell=None, area=None):
    """
    Create a dataframe with the same structure of the one containing
    basic information on point sources but for the points requiring the
    definition of a baseline seismicity.

    :param geohashes:
        A :class:`list` instance with the indexes of the point sources to be
        created
    :param a_value:
        The a_gr value per km2
    :param b_value:
        The b_gr value
    :param mmin:
        The minimum value of magnitude to be used for the calculation of
        seismicity
    """

    # Coordinates
    coo = np.array([h3.cell_to_latlng(idx) for idx in geohashes])

    # Compute the output a_value in each cell
    if a_cell is None:
        a_cell, _ = _get_rates(geohashes, a_value)
    b_cell = np.ones_like(coo[:, 0]) * b_value

    # Output dataframe
    sdf = pd.DataFrame({'lon': coo[:, 1], 'lat': coo[:, 0], 'agr': a_cell,
                        'bgr': b_cell})

    return sdf


def add_baseline_seismicity(folder_name: str, folder_name_out: str,
                            fname_config: str, fname_poly: str, use=[],
                            skip=[]):
    """
    Add baseline seismicity to the sources in the `folder_name`. The
    configuration file must contain

    :param folder_name:
        The name of the folder containing the files with GR parameters for the
        points in each zone considered
    :param folder_name_out:
        The folder where to write the results
    :param config_file:
        A .toml file with the configuration parameters
    :param shapefile:
        The name of the shapefile containing the geometry of the polygons used
    :param use:
        A list with the IDs of sources that will be used
    :param skip:
        A list with the IDs of sources that should be skipped [NOT ACTIVE!!!]
    :returns:
        An updated set of .csv files
    """

    if folder_name in ['None', 'none', "'None'"]:
        folder_name = None

    if len(use) > 0:
        use = get_list(use)

    if len(skip) > 0:
        if isinstance(skip, str):
            skip = get_list(skip)
        print('Skipping: ', skip)

    # Create output folder
    create_folder(folder_name_out)

    # Parsing config. The basel_agr value is the log of the rate per km2 per
    # year for earthquakes larger than 0
    model = toml.load(fname_config)
    h3_level = model['baseline']['h3_level']
    basel_agr = model['baseline']['a_value']
    basel_bgr = model['baseline']['b_value']
    set_all_cells = model['baseline'].get('set_all_cells', False)

    # Read polygons
    polygons_gdf = gpd.read_file(fname_poly)

    # Loop over the polygons
    polygons_gdf.sort_values(by="id", ascending=True, inplace=True)
    polygons_gdf.reset_index(drop=True, inplace=True)
    for src_id, poly in polygons_gdf.iterrows():

        if (len(use) > 0 and src_id not in use) or (src_id in skip):
            continue

        tmp = shapely.geometry.mapping(poly.geometry)
        geojson_poly = eval(json.dumps(tmp))

        # Take the exterior in a Polygon and the first geometry in a
        # MultiPolygon
        if geojson_poly['type'] == "MultiPolygon":
            tmp_coo = geojson_poly['coordinates'][0][0]
            message = 'Taking the first polygon of a multipolygon'
            warnings.warn(message, UserWarning)
        elif geojson_poly['type'] == "Polygon":
            tmp_coo = geojson_poly['coordinates'][0]
        else:
            raise ValueError('Unsupported Geometry')

        # Revert the positions of lons and lats
        coo = [[c[1], c[0]] for c in tmp_coo]
        geojson_poly['coordinates'] = [coo]
        geojson_poly['type'] = "Polygon"

        # Discretizing the polygon i.e. find all the hexagons covering the
        # polygon describing the current zone
        hexagons = list(h3.polygon_to_cells(geojson_poly, h3_level))

        # Read the file with the points obtained by the smoothing
        print("Source ID", poly.id)
        if folder_name is None:
            tmp_data = {'lon': [], 'lat': [], 'agr': [], 'bgr': []}
            df = pd.DataFrame(data=tmp_data)
        else:
            fname = os.path.join(folder_name, f'{poly.id}.csv')
            df = pd.read_csv(fname)

        # Create a list with the geohashes of the points with a rate. This is
        # the output of the smoothing.
        srcs_idxs = [
            h3.latlng_to_cell(la, lo, h3_level) for lo, la in zip(df.lon, df.lat)]
        hxg_idxs = [hxg for hxg in hexagons]

        # `missing` contains the number of cells used to discretize the polygon
        # and without a rate
        missing = list(set(hxg_idxs) - set(srcs_idxs))

        # This instead finds the cells with a rate lower that the minimum rate
        # defined in the configuration file
        tmp = np.nonzero([df.agr <= basel_agr])[0]

        # If we don't miss cells and rates are all above the threshold there
        # is nothing else to do
        fname = os.path.join(folder_name_out, f'{poly.id}.csv')
        if len(missing) == 0 and len(tmp) == 0:
            df.to_csv(fname, index=False)
            continue

        # Get the indexes of the point sources with low rates
        idxs = np.nonzero(df.agr.to_numpy() <= basel_agr)[0]
        low = [srcs_idxs[i] for i in idxs]

        # Remove the sources with activity below the threshold since these
        # will be replaced by new new point sources
        df.drop(df.index[idxs], inplace=True)

        # Find the h3 indexes of the point sources either without seismicity
        # or with a rate below the baseline
        both = set(missing) | set(low)

        # Adding baseline seismicity to the dataframe for the current source
        if len(both) > 0:
            if set_all_cells is False:
                tmp_df = create_missing(both, basel_agr, basel_bgr)
                df = pd.concat([df, tmp_df])
            else:
                df = create_missing(hxg_idxs, basel_agr, basel_bgr)

        # Creating output file
        assert len(hxg_idxs) == df.shape[0]
        df.to_csv(fname, index=False)
