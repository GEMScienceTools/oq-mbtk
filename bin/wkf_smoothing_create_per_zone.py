#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.wkf.utils import create_folder
from openquake.baselib import sap
from shapely.geometry import Point
from shapely import wkt


def create_smoothing_per_zone(fname_points: str, fname_polygons: str, 
                              folder_out: str='/tmp', skip=[]):
    """
    Creates subsets of points, one for each of the polygons included in the 
    `fname_polygons` shapefile. The attibute table must have an 'id' attribute.
    
    :param fname_points:
    :param fname_polygons:
        The shapefile with the polygons
    :param folder_out:
        The name of the folder where to write the output
    :returns:
        This function creates a number of .csv files in `folder_out` 
    """

    create_folder(folder_out)

    # Create geodataframe 
    df = pd.read_csv(fname_points)
    gdf = gpd.GeoDataFrame(df.drop(['lon', 'lat'], axis=1), crs='epsg:4326',
                           geometry=[Point(xy) for xy in zip(df.lon, df.lat)])
    
    # Read polygons
    polygons_gdf = gpd.read_file(fname_polygons)
    
    # Select point in polygon
    for idx, poly in polygons_gdf.iterrows():
        
        if poly.id in skip:
            continue
        
        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')
        within = gpd.sjoin(gdf, gdf_poly, op='within')

        within['lon'] = within['geometry'].x
        within['lat'] = within['geometry'].y
        
        if isinstance(poly.id, str):
            fout = os.path.join(folder_out, '{:s}.csv'.format(poly.id))
        else:
            fout = os.path.join(folder_out, '{:d}.csv'.format(poly.id))
        
        if len(within):
            within.to_csv(fout, index=False, columns=['lon', 'lat', 'nocc'])

create_smoothing_per_zone.fname_points = ".csv file created by the smoothing code"
create_smoothing_per_zone.fname_polygons = "Shapefile with the polygons"
create_smoothing_per_zone.folder_out = "The name of the output folder where to save .csv files"
create_smoothing_per_zone.skip = "A string containing a list of source IDs"

if __name__ == '__main__':
    sap.run(create_smoothing_per_zone)
