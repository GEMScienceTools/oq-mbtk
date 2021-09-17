#!/usr/bin/env python
# coding: utf-8

import os
import h3
import toml
import json
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.wkf.utils import create_folder
from openquake.baselib import sap
from shapely.geometry import Point
from shapely import wkt


def create_missing(missing, h3_level, a_value, b_value):
    """
    Create a dataframe with the same structure of the dataframe containing 
    basic information on point sources but for the points requiring a 
    baseline seismicity.
    """
    coo = np.array([h3.h3_to_geo(idx) for idx in missing])
    a = np.ones_like(coo[:,0]) * a_value
    b = np.ones_like(coo[:,0]) * b_value
    df = pd.DataFrame({'lon': coo[:,0], 'lat': coo[:,1], 'agr': a, 'bgr': b})
    return df


def add_baseline_seismicity(folder_name: str, folder_name_out: str, 
                            fname_config: str, fname_poly: str, skip=[]):
    """
    
    :param folder_name:
        The name of the folder containing the files with GR parameters for the 
        points in each zone considered
    :param folder_name_out:
        The folder where to write the results
    :param config_file:
        A .toml file with the configuration parameters
    :param shapefile:
        The name of the shapefile containing the geometry of the polygons used
    :param skip:
        A list with the sources that should be skipped [NOT ACTIVE!!!]
    :returns:
        An updated set of .csv files 
    """

    # Create output folder
    create_folder(folder_name_out)

    # Parsing config
    model = toml.load(fname_config)
    h3_level = model['baseline']['h3_level']
    basel_agr = model['baseline']['a_value']
    basel_bgr = model['baseline']['b_value']
    
    # Read polygons
    polygons_gdf = gpd.read_file(fname_poly)

    # Loop over the polygons
    polygons_gdf.sort_values(by="id", ascending=True, inplace=True)
    polygons_gdf.reset_index(drop=True, inplace=True)

    for idx, poly in polygons_gdf.iterrows():

        geojson_poly = eval(json.dumps(shapely.geometry.mapping(poly.geometry)))

        # Revert the positions of lons and lats
        coo = [[c[1], c[0]] for c in geojson_poly['coordinates'][0]]
        geojson_poly['coordinates'] = [coo]

        # Discretizing the polygon i.e. find all the hexagons covering the 
        # polygon describing the current zone
        hexagons = list(h3.polyfill(geojson_poly, h3_level))

        # Read the file with the points obtained by the smoothing
        print("Source ID", poly.id)
        fname = os.path.join(folder_name, '{:s}.csv'.format(poly.id))
        df = pd.read_csv(fname)

        srcs_idxs = [h3.geo_to_h3(la, lo, h3_level) for lo, la in zip(df.lon, df.lat)]
        hxg_idxs = [hxg for hxg in hexagons]

        missing = list(set(hxg_idxs) - set(srcs_idxs))
        tmp = np.nonzero([df.agr <= basel_agr])[0]

        # If we don't miss cells and rates are all above the threshold there 
        # is nothing else to do 
        fname = os.path.join(folder_name_out, "{:s}.csv".format(poly.id))
        if len(missing) == 0 and len(tmp) == 0:
            df.to_csv(fname, index=False)
            continue

        # Get the indexes of the point sources with low rates
        idxs = np.nonzero(df.agr.to_numpy() <= basel_agr)[0]
        low = [srcs_idxs[i] for i in idxs]

        # Removing the sources with activity below the threshold
        df.drop(df.index[idxs], inplace=True)

        # Find the h3 indexes of the point sources either without seismicity 
        # or with a rate below the baseline
        both = set(missing) | set(low)

        # Adding baseline seismicity to the dataframe for the current source
        if len(both) > 0:
            tmp_df = create_missing(both, h3_level, basel_agr, basel_bgr)
            df = df.append(tmp_df)

        # Creating output file
        assert len(hxg_idxs) == df.shape[0]
        df.to_csv(fname, index=False)


def main(folder_name: str, folder_name_out: str, fname_config: str, 
         fname_poly: str, skip=[]):
       
    add_baseline_seismicity(folder_name, folder_name_out, fname_config, 
                            fname_poly, skip)


main.folder_name = "The name of the folder with smoothing results per source"
main.folder_name_out = "The name of the folder where to store the results"
main.fname_config = ".toml configuration file"
main.fname_poly = "The name of the shapefile with the polygons"
main.skip = "A string containing a list of source IDs"

if __name__ == '__main__':
    sap.run(main)
