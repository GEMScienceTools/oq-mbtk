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
import sys
import toml
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime

from openquake.baselib import sap
from openquake.mbi.cat.create_csv import create_folder
from geojson import LineString, Feature, FeatureCollection, dump
from openquake.cat.isf_catalogue import get_threshold_matrices
from openquake.hazardlib.geo.geodetic import geodetic_distance


def get_features(cat, idx, idxsel):
    """
    :param cat:
        A pandas geodataframe instance containing a homogenised catalogue as
        obtained from :method:`openquake.cat.hmg.merge.hmg.process_dfs`
    :param idx: index of event
    :param idxsel: index of close event
    """
    features = []

    lon1 = float(cat.loc[idx, 'longitude'])
    lat1 = float(cat.loc[idx, 'latitude'])
    tmp = cat.loc[idx, 'eventID']
    mag1 = float(cat.loc[idx, 'value'])
    time1 = cat.loc[idx, 'datetime']
    print('ref time ', time1)
    if type(tmp).__name__ == 'str':
        evid = tmp
    elif type(tmp).__name__ in ['int', 'int64', 'int32']:
        evid = "{:d}".format(cat.loc[idx, 'eventID'])
    else:
        fmt = "Unsupported format for EventID: {:s}"
        raise ValueError(fmt.format(type(tmp).__name__))

    # mag2 = cat.loc[idxsel, 'value'].apply(lambda x: float(x))
    # reference agency used for idx
    ref_agency = cat.loc[idx, 'Agency']

    for i in idxsel:
        lon2 = float(cat.loc[i, 'longitude'])
        lat2 = float(cat.loc[i, 'latitude'])

        londiff = abs(lon1 - lon2)
        latdiff = abs(lat1 - lat2)
        km_diff = geodetic_distance(lon1, lat1, lon2, lat2)
        # Magnitude difference between events
        mag_diff = abs(mag1 - float(cat.loc[i, 'value']))

        # Time difference between events
        t_del = abs(time1 - cat.loc[i, 'datetime']).total_seconds()
        line = LineString([(lon1, lat1), (lon2, lat2)])
        props = {"eventID": evid, "magDiff": mag_diff, "delta_t": t_del,
                 "lon_diff": londiff, "lat_diff": latdiff, "km_diff": km_diff,
                 "m1": mag1, "agency": cat.loc[i, 'Agency'],
                 "ref_agency": ref_agency, "mag_type":  cat.loc[i, 'magType']}
        features.append(Feature(geometry=line, properties=props))

    return features


def process(cat, sidx, delta_ll, delta_t, fname_geojson, use_kms=False):
    """
    :param cat
        A pandas geodataframe instance containing a homogenised catalogue as
        obtained from :method:`openquake.cat.hmg.merge.hmg.process_dfs`
    :param sidx:
        Spatial index for the geodataframe as obtained by `gdf.sindex`
    :param delta_ll:
        A float defining the longitude/latitude tolerance used for checking
    :param delta_t:
        A float [in seconds] the time tolerance used to search for duplicated
        events.
    :param fname_geojson:
        Name of the output .geojson file which will contains the lines
        connecting the possibly duplicated events.
    :param use_kms:
        Specify if distance buffer should use kms (default is False, use
        degrees)
    """

    features = []
    found = set()
    # delta_t = dt.timedelta(seconds=delta_t)
    # Get the edges of magnitude and time plus the matrixes with the
    # delta values that should be used
    gtm = get_threshold_matrices
    mag_low_edges, time_low_edges, time_d, ll_d = gtm(delta_t, delta_ll)
    cnt = 0

    from tqdm import tqdm
    # Loop over the earthquakes in the catalogue
    # datetime can only cover 548 years starting in 1677
    # general advice will be to exclude historic events and
    # add those later
    subcat = cat[(cat['year'] > 1800) & (cat['value'] > 1.0)]
    
    for index, row in tqdm(subcat.iterrows()):
        # Take the index from delta_ll - this is needed
        # when delta_ll varies with time.
        # magnitude = row.value
        idx_mag = max(np.argwhere(row.value > mag_low_edges))[0]
        idx_t = max(np.argwhere(np.float64(row.year) >= time_low_edges))[0]

        ll_thrs = ll_d[idx_t][idx_mag]
        sel_thrs = time_d[idx_t][idx_mag]
        #sel_thrs = sel_thrs.total_seconds()
        
        # Find events close in time
        tmp_dff = abs(subcat.loc[:, 'datetime'] - pd.to_datetime(row.datetime))
        threshold = datetime.timedelta(seconds=sel_thrs)
        tmp = tmp_dff.astype('timedelta64[s]') < threshold
        idx_time = list(tmp[tmp].index)

        if use_kms is False:
            # Select events that occurred close in space
            minlo = row.longitude - ll_thrs
            minla = row.latitude - ll_thrs
            maxlo = row.longitude + ll_thrs
            maxla = row.latitude + ll_thrs
            idx_dist_ind = list(sidx.intersection((minlo, minla, maxlo, maxla)))
            idx_dist = cat.index[idx_dist_ind]

        else:
            tmp_dist = abs(geodetic_distance(
                row.longitude, row.latitude, subcat.loc[:, 'longitude'],
                subcat.loc[:, 'latitude'])) < ll_thrs
            idx_dist = list(tmp_dist[tmp_dist].index)

        # Find the index of the events that are matching temporal and spatial
        # constraints
        idx = (set(idx_dist) & set(idx_time)) - found

        if len(idx) > 1:
            cnt += 1
            features.extend(get_features(subcat, index, idx))
            for i in idx:
                found.add(i)

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

    print("Checking catalogue")

    # Read configuration
    settings = toml.load(settings_fname)
    print(settings)

    # Load the catalogue
    _, file_extension = os.path.splitext(catalogue_fname)
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
                                              'minute', 'second']],
                                         errors='coerce')

    # Set filename
    out_path = settings["general"]["output_path"]
    geojson_fname = os.path.join(out_path, "check.geojson")
    create_folder(out_path)
    print('Created: {:s}'.format(out_path))

    # Processing the catalogue
    delta_ll = settings["general"]["delta_ll"]
    delta_t = settings["general"]["delta_t"]
    # Check for use_kms parameter and set to False if not in settings
    # use_kms = settings["general"].get("use_kms", False)
    nchecks = process(cat, sindex, delta_ll, delta_t, geojson_fname)
    return nchecks


def main(argv):
    """ """

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
