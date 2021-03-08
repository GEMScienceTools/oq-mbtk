#!/usr/bin/env python
# coding: utf-8

import os
import h3
import json
import toml
import shapely
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from openquake.wkf.utils import create_folder
from openquake.baselib import sap


def mmax_per_zone(fname_poly: str, fname_cat: str, fname_conf: str, cat_lab):

    # Parsing config
    model = toml.load(fname_conf)

    # Read polygons
    polygons_gdf = gpd.read_file(fname_poly)

    # Create geodataframe with the catalogue 
    df = pd.read_csv(fname_cat)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy
                           in zip(df.longitude, df.latitude)])

    # Iterate over sources
    for idx, poly in polygons_gdf.iterrows():

        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')
        within = gpd.sjoin(gdf, gdf_poly, op='within')
        mmax = np.max(within.magnitude.to_numpy())
        lab = 'mmax_{:s}'.format(cat_lab)
        model['sources'][poly.id][lab] = float(mmax)

    # Saving results into the config file
    with open(fname_conf, 'w') as f:
        f.write(toml.dumps(model))
        print('Updated {:s}'.format(fname_conf))


descr = 'The name of the shapefile with the polygons'
mmax_per_zone.fname_poly = descr
descr = 'The name of the .csv file with the catalogue'
mmax_per_zone.fname_cat = descr
descr = 'The name of configuration file'
mmax_per_zone.fname_conf = descr
descr = 'The label used to identify the catalogue'
mmax_per_zone.cat_lab = descr

if __name__ == '__main__':
    sap.run(mmax_per_zone)
