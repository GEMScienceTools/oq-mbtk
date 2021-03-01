#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.baselib import sap
from shapely.geometry import Point
from shapely import wkt
from openquake.wkf.utils import create_folder


def create_subcatalogues(fname_polygons: str, fname_cat: str, folder_out: str):

    # Create output folder
    create_folder(folder_out)

    # Create geodataframe with the catalogue 
    df = pd.read_csv(fname_cat)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy
                           in zip(df.longitude, df.latitude)])

    # Read polygons
    polygons_gdf = gpd.read_file(fname_polygons)

    # Select point in polygon
    columns = ['eventID', 'year', 'month', 'day', 'magnitude', 'longitude', 
               'latitude', 'depth']
    
    # Iterate over sources
    for idx, poly in polygons_gdf.iterrows():
        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')
        within = gpd.sjoin(gdf, gdf_poly, op='within')
        # Create output file
        if isinstance(poly.id, int):
            fname = 'subcatalogue_zone_{:d}.csv'.format(poly.id)
        else:
            fname = 'subcatalogue_zone_{:s}.csv'.format(poly.id)
        out_fname = os.path.join(folder_out, fname)
        within.to_csv(out_fname, index=False, columns=columns)

create_subcatalogues.fname_polygons = 'Name of a shapefile with polygons'
create_subcatalogues.fname_cat = 'Name of the .csv file with the catalog'
create_subcatalogues.folder_out = 'Name of the output folder'

if __name__ == '__main__':
    sap.run(create_subcatalogues)
