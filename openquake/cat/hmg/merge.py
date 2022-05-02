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


import warnings

import os
import toml
import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd

from openquake.cat.parsers.isf_catalogue_reader import ISFReader
from openquake.cat.parsers.converters import GenericCataloguetoISFParser

warnings.filterwarnings('ignore')


def get_delta_t(tmpl):
    """
    :param tmpl:
        Either a float (or string) or an iterable containing tuples with
        an int (year from which this delta time applies) and a float (time
        in seconds)
    :return:
        Either a float or a list
    """
    if not hasattr(tmpl, '__iter__'):
        return dt.timedelta(seconds=float(tmpl))

    # Creating list
    out = []
    for tmp in tmpl:
        out.append([dt.timedelta(seconds=float(tmp[0])), int(tmp[1])])
    return out


def coords_prime_origins(catalogue):
    """
    :param catalogue:
        An ISFCatalogue instance
    """
    # Get coordinates of primes events
    data = []
    for iloc, event in enumerate(catalogue.events):
        for iori, origin in enumerate(event.origins):
            if not origin.is_prime and len(event.origins) > 1:
                continue
            else:
                if len(origin.magnitudes) == 0:
                    continue
            # Saving information regarding the prime origin
            data.append((origin.location.longitude,
                         origin.location.latitude, iloc, iori))
    return np.array(data)


def magnitude_selection(catalogue: str, min_mag: float):
    """
    :param catalogue:
        An instance of :class:`openquake.cat.isf_catalogue.ISFCatalogue`
    :param min_mag:
        Minimum magnitude
    """
    # Filter events
    iii = []
    for iloc, event in enumerate(catalogue.events):
        for iori, origin in enumerate(event.origins):
            # Skipping events that are not prime
            if not origin.is_prime and len(event.origins) > 1:
                continue
            else:
                if len(origin.magnitudes) > 0:
                    for m in origin.magnitudes:
                        if m.value > (min_mag-0.001):
                            iii.append(iloc)
                            continue
    return catalogue.get_catalogue_subset(iii)


def geographic_selection(catalogue, shapefile_fname, buffer_dist=0.0):
    """
    :param catalogue:
        An instance of :class:`openquake.cat.isf_catalogue.ISFCatalogue`
    :param shapefile_fname:
        Name of a shapefile
    :param buffer_dist:
        A distance in decimal degrees
    """

    # Getting info on prime events
    data = coords_prime_origins(catalogue)
    tmp = np.array(data[:, 2:4], dtype=int)

    # Create geodataframe with the catalogue
    origins = pd.DataFrame(tmp, columns=['iloc', 'iori'])
    tmp = gpd.points_from_xy(data[:, 0], data[:, 1])
    origins = gpd.GeoDataFrame(origins, geometry=tmp, crs="EPSG:4326")

    # Reading shapefile and dissolving polygons into a single one
    boundaries = gpd.read_file(shapefile_fname)
    boundaries['dummy'] = 'dummy'
    geom = boundaries.dissolve(by='dummy').geometry[0]

    # Adding a buffer - Assuming units are decimal degreess
    if buffer_dist > 0:
        geom = geom.buffer(buffer_dist)

    # Selecting origins - Tried two methods both give the same result
    #pip = origins.within(geom)
    #aaa = origins.loc[pip]
    tmpgeo = {'col1': ['tmp'], 'geometry': [geom]}
    gdf = gpd.GeoDataFrame(tmpgeo, crs="EPSG:4326")
    aaa = gpd.sjoin(origins, gdf, how="inner", op='intersects')

    # This is for checking purposes
    aaa.to_file("/tmp/within.geojson", driver='GeoJSON')

    return catalogue.get_catalogue_subset(list(aaa["iloc"].values))


def load_catalogue(fname, cat_type, cat_code, cat_name):
    """
    :param fname:
    :param cat_type:
    :param cat_code:
    :param cat_name:
    """
    if cat_type == "isf":
        parser = ISFReader(fname)
        cat = parser.read_file(cat_code, cat_name)
    elif cat_type == "csv":
        parser = GenericCataloguetoISFParser(fname)
        cat = parser.parse(cat_code, cat_name)
    else:
        raise ValueError("Unsupported catalogue type")
    fmt = '    The original catalogue contains {:d} events'
    print(fmt.format(len(cat.events)))
    return cat


def process_catalogues(settings_fname):
    """
    :fname settings_fname:
        Name of the .toml file containing the information about the
        catalogues to be merged
    """

    # Read configuration
    settings = toml.load(settings_fname)
    path = os.path.dirname(settings_fname)

    # Reading shapefile and setting the buffer
    tmps = settings["general"].get("region_shp", None)
    if tmps is not None:
        fname_shp = os.path.join(path, tmps)
        buffr = float(settings["general"].get("region_buffer", 0.))

    if len(settings["catalogues"]) < 1:
        raise ValueError("Please specify a catalogue in the settings")

    # Processing the catalogues
    for icat, tdict in enumerate(settings["catalogues"]):

        # Get settings
        fname = os.path.join(path, tdict["filename"])
        cat_type = tdict["type"]
        cat_code = tdict["code"]
        cat_name = tdict["name"]

        print("\nCatalogue:", cat_name)

        # Reading the first catalogue
        if icat == 0:

            catroot = load_catalogue(fname, cat_type, cat_code, cat_name)
            nev = catroot.get_number_events()
            print("   Catalogue contains: {:d} events".format(nev))

            select_flag = tdict.get("select_region", False)
            if select_flag:
                msg = "Selecting earthquakes within the region shapefile"
                print("      " + msg)
                catroot = geographic_selection(catroot, fname_shp, buffr)
                msg = "Selected {:d} earthquakes".format(len(catroot))
                print("      " + msg)

            min_mag = settings["general"].get("minimum_magnitude", False)
            if min_mag:
                msg = "Selecting earthquakes above {:f}".format(min_mag)
                print("      " + msg)
                catroot = magnitude_selection(catroot, min_mag)

            # Add the spatial index
            if 'sidx' not in catroot.__dict__:
                print("      Building index")
                catroot._create_spatial_index()

            # Set log files
            if "log_file" not in tdict:
                logfle = "/tmp/tmp_merge_{:02d}.tmp".format(icat)
            else:
                logfle = tdict["log_file"]
            print("   Log file: {:s}".format(logfle))

        else:

            # Loading the catalogue
            tmpcat = load_catalogue(fname, cat_type, cat_code, cat_name)
            nev = tmpcat.get_number_events()
            print("   Catalogue contains: {:d} events".format(nev))

            select_flag = tdict.get("select_region", False)
            if select_flag:
                msg = "Selecting earthquakes within the region shapefile"
                print("      " + msg)
                tmpcat = geographic_selection(tmpcat, fname_shp, buffr)
                msg = "Selected {:d} earthquakes".format(len(tmpcat))
                print("      " + msg)

            # Setting the parameters for merging
            delta_ll = tdict["delta_ll"]
            delta_t = get_delta_t(tdict["delta_t"])
            timezone = dt.timezone(dt.timedelta(hours=int(tdict["timezone"])))
            buff_ll = tdict["buff_ll"]
            buff_t = dt.timedelta(seconds=tdict["buff_t"])
            use_ids = tdict["use_ids"]

            # Set log files
            if "log_file" not in tdict:
                logfle = "/tmp/tmp_merge_{:02d}.tmp".format(icat)
            else:
                logfle = tdict["log_file"]
            print("   Log file: {:s}".format(logfle))

            # Merging
            meth = catroot.add_external_idf_formatted_catalogue
            out = meth(tmpcat, delta_ll, delta_t, timezone, buff_t, buff_ll,
                       use_ids, logfle)

            # Updating spatial index
            print("      Updating index")
            catroot._create_spatial_index()

        nev = catroot.get_number_events()
        print("   Whole catalogue contains: {:d} events".format(nev))

    # Building dataframes
    otab, mtab = catroot.build_dataframe()

    # Creating output folder
    out_path = settings["general"].get("output_path", "./out")
    out_path = os.path.join(path, out_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    prefix = settings["general"].get("output_prefix", "")
    fname_or = os.path.join(out_path, "{:s}otab.h5".format(prefix))
    fname_mag = os.path.join(out_path, "{:s}mtab.h5".format(prefix))

    # Saving results
    print("\nSaving results to: \n{:s}\n{:s}".format(fname_or, fname_mag))
    otab.to_hdf(fname_or, '/origins', append=False)
    mtab.to_hdf(fname_mag, '/magnitudes', append=False)

    print("\nLog file: \n{:s}".format(logfle))
