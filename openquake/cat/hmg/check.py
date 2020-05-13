#!/usr/bin/env python
# coding: utf-8

# Copyright (C) 2020 GEM Foundation
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
import sys
import toml
import pandas as pd
import datetime as dt
import geopandas as gpd

from tqdm import tqdm
from openquake.baselib import sap
from geojson import LineString, Feature, FeatureCollection, dump


def get_features(cat, idx, idxsel):
    """
    :param cat:
    :param idx:
    :param idxsel:
    """
    features = []

    lon1 = cat.loc[idx, 'longitude']
    lat1 = cat.loc[idx, 'latitude']
    evid = cat.loc[idx, 'eventID']

    for i in idxsel:
        lon2 = cat.loc[i, 'longitude']
        lat2 = cat.loc[i, 'latitude']
        line = LineString([(lon1, lat1), (lon2, lat2)])
        features.append(Feature(geometry=line, properties={"eventID": evid}))

    return features


def process(cat, sidx, delta_ll, delta_t, fname_geojson):
    """
    :param cat
        A pandas geodataframe instance
    :param sidx:
        Spatial index
    :param delta_ll:
        Delta longitude/latitude
    :param delta_t:
        Delta time
    :param fname_geojson:
        Name of the .geojson file
    """

    features = []
    delta_t = dt.timedelta(seconds=delta_t)
    cnt = 0

    # Loop over the earthquakes in the catalogue
    for index, row in tqdm(cat.iterrows()):
        print(" ------------- ")

        # Select events that occurred close in space
        minlo = row.longitude - delta_ll
        minla = row.latitude - delta_ll
        maxlo = row.longitude + delta_ll
        maxla = row.latitude + delta_ll
        idx_space = list(sidx.intersection((minlo, minla, maxlo, maxla)))

        # Select events that occurred close in time
        print(abs(cat.loc[:, 'datetime'] - row.datetime))
        tmp = abs(cat.loc[:, 'datetime'] - row.datetime) < delta_t
        idx_time = list(tmp[tmp].index)

        # Find the index of the events that are matching temporal and spatial
        # constraints
        print("space", idx_space)
        print("time", idx_time)

        idx = set(idx_space) & set(idx_time)

        if len(idx) > 1:
            cnt += 1
            features.extend(get_features(cat, index, idx))

    # Create the geojson file
    feature_collection = FeatureCollection(features)
    with open(fname_geojson, 'w') as fou:
        dump(feature_collection, fou)

    # Info
    if cnt > 0:
        print("Created file: {:s}".format(fname_geojson))

    return cnt


def check_catalogue(catalogue_fname, settings_fname):
    """
    :fname catalogue_fname:
        An .h5 file with the homogenised catalogue
    :fname settings_fname:
        The name of a file containing the settings used to create a catalogue
    """

    # Read configuration
    settings = toml.load(settings_fname)

    # Load the catalogue
    _, file_extension = os.path.splitext(catalogue_fname)
    print(file_extension)
    if file_extension in ['.h5', '.hdf5']:
        cat = pd.read_hdf(catalogue_fname)
    elif file_extension == '.csv':
        cat = pd.read_csv(catalogue_fname)
    else:
        raise ValueError("File format not supported")

    # Getting a geodataframe
    if type(cat).__name__ != 'GeoDataFrame':
        cat = gpd.GeoDataFrame(cat, geometry=gpd.points_from_xy(cat.longitude,
                                                                cat.latitude))

    # Create the spatial index
    sindex = cat.sindex

    # Add datetime field
    if "datetime" not in cat.keys():
        cat['datetime'] = pd.to_datetime(cat[['year', 'month', 'day', 'hour',
                                              'minute', 'second']])

    # Set filename
    out_path = settings["general"]["output_path"]
    geojson_fname = os.path.join(out_path, "check.geojson")

    # Processing the catalogue
    delta_ll = settings["general"]["delta_ll"]
    delta_t = settings["general"]["delta_t"]
    nchecks = process(cat, sindex, delta_ll, delta_t, geojson_fname)

    return nchecks


def main(argv):
    """ Plots shallow sources"""

    p = sap.Script(check_catalogue)
    msg = 'Name of a .h5 file containing the homogenised catalogue'
    p.arg(name='catalogue_fname', help=msg)
    p.arg(name='settings_fname', help='.toml file with the model settings')
    p.arg(name='out_folder', help='path of the output folder')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
