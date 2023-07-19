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
from openquake.cat.isf_catalogue import get_delta_t

warnings.filterwarnings('ignore')


def coords_prime_origins(catalogue):
    """
    Given an instance of an ISFCatalogue returns an array where each row
    contains the longitude, latitude, the index of location and the one
    of the origin.

    :param catalogue:
        An :class:`openquake.cat.isf_catalogue.ISFCatalogue` instance
    :returns:
        An instance of :class:`numpy.ndarray`. The cardinality of the output
        has a cardinality equal to the number of earthquakes in the
        catalogue with a prime solution.
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
    Given a catalogue and a shapefile with a polygon or a set of polygons,
    select all the earthquakes inside the union of all the polygons and
    return a new catalogue instance.

    :param catalogue:
        An instance of :class:`openquake.cat.isf_catalogue.ISFCatalogue`
    :param shapefile_fname:
        Name of a shapefile
    :param buffer_dist:
        A distance in decimal degrees
    :returns:
        An instance of :class:`openquake.cat.isf_catalogue.ISFCatalogue`
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
    # pip = origins.within(geom)
    # aaa = origins.loc[pip]
    tmpgeo = {'col1': ['tmp'], 'geometry': [geom]}
    gdf = gpd.GeoDataFrame(tmpgeo, crs="EPSG:4326")
    aaa = gpd.sjoin(origins, gdf, how="inner", op='intersects')

    # This is for checking purposes
    aaa.to_file("/tmp/within.geojson", driver='GeoJSON')

    return catalogue.get_catalogue_subset(list(aaa["iloc"].values))


def load_catalogue(fname: str, cat_type: str, cat_code: str, cat_name: str):
    """
    Given the name of a file (the supported formats are 'csv' and 'isf') read
    its content and return a catalogue instance.

    :param fname:
        Name of the file with the catalogue
    :param cat_type:
        Type of catalogue. Options are 'isf' and 'csv'
    :param cat_code:
        The code to be assigned to earthquakes from this catalogue
    :param cat_name:
        The name of this catalogue
    :return:
        An instance of :class:`openquake.cat.isf_catalogue.ISFCatalogue`
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


def process_catalogues(settings_fname: str) -> None:
    """
    Given a .toml file containing the list of catalogues to be merged,
    process the catalogues and save the results in the output folder
    specified in the settings file.

    :fname settings_fname:
        Name of the .toml file containing the information about the
        catalogues to be merged
    """

    # Read configuration file
    settings = toml.load(settings_fname)
    path = os.path.dirname(settings_fname)

    # Read the name of the shapefile and - if defined - the info on
    # the buffer (otherwise `buffr` is 0)
    tmps = settings["general"].get("region_shp", None)
    if tmps is not None:
        fname_shp = os.path.join(path, tmps)
        buffr = float(settings["general"].get("region_buffer", 0.))

    # Check that the file
    if len(settings["catalogues"]) < 1:
        raise ValueError("Please specify a catalogue in the settings")
        
    #if "log_file" in settings["general"]:
    #    log_fle = settings["general"]["log_file"]

    # Process the catalogue. `tdict` is dictionary with the info
    # required to merge one specific catalogue.
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
            print(f"   Catalogue contains: {nev:d} events")

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
                if "log_file" in settings["general"]:
                    logfle = settings["general"]["log_file"]
                else:
                    logfle = os.path.join(path, f"tmp_merge_{icat:02d}.tmp".format(icat))
            else:
                logfle = tdict["log_file"]
            print("   Log file: {:s}".format(logfle))

        # Process the additional catalogues
        else:

            # Load the catalogue and get the number of events
            tmpcat = load_catalogue(fname, cat_type, cat_code, cat_name)
            nev = tmpcat.get_number_events()
            print(f"   Catalogue contains: {nev:d} events")

            # If requested, select the earthquakes within the polygon
            # specified in the configuration file
            select_flag = tdict.get("select_region", False)
            if select_flag:
                msg = "Selecting earthquakes within the region shapefile"
                print("      " + msg)
                tmpcat = geographic_selection(tmpcat, fname_shp, buffr)
                msg = "Selected {:d} earthquakes".format(len(tmpcat))
                print("      " + msg)

            # Set the parameters required for merging the new catalogue
            # including a delta-distance and delta-time.
            # - `delta_ll` is a float or a string defining a distance
            #   in degrees or kms if use_kms = True. Can be specified as
            #   a function of magnitude.
            # - `delta_t` is an integer or a string defining a delta
            #   time in seconds. Can be specified as a function of magnitude
            delta_ll = tdict["delta_ll"]
            delta_t = get_delta_t(tdict["delta_t"])
            # - `timezone` an integer
            tzone = int(tdict.get("timezone", 0))
            timezone = dt.timezone(dt.timedelta(hours=tzone))
            # - buffer distances for time and distance used for TODO
            buff_ll = tdict["buff_ll"]
            buff_t = dt.timedelta(seconds=tdict["buff_t"])
            # - `use_ids` a boolean specifying is the ids of this catalogue
            #   should be used to find corresponding earthquakes in the
            #   catalogues already merged
            use_ids = tdict.get("use_ids", False)
            # - `use_kms` specifies if delta_ll distances are in kms or degrees
            use_kms = tdict.get("use_kms", False)

            # Set the name of the log file
            if "log_file" not in tdict:
                if "log_file" in settings["general"]:
                    logfle = settings["general"]["log_file"]
                else:
                    logfle = os.path.join(path, f"tmp_merge_{icat:02d}.tmp" )
            else:
                logfle = tdict["log_file"]
            #
            print(f"   Log file: {logfle:s}".format())
            print(logfle)
            # Perform the merge
            meth = catroot.add_external_idf_formatted_catalogue
            out = meth(tmpcat, delta_ll, delta_t, timezone, buff_t, buff_ll, use_kms,
                       use_ids, logfle)

            # Update the spatial index
            print("      Updating index")
            catroot._create_spatial_index()
        
        nev = catroot.get_number_events()
        print(f"   Whole catalogue contains: {nev:d} events")

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

    print(f"\nLog file: \n{logfle:s}")
