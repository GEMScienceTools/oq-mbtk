#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.wkf.utils import create_folder, get_list
from openquake.baselib import sap
from shapely.geometry import Point
from shapely import wkt


def main(fname_points: str, fname_polygons: str, folder_out: str='/tmp', *,
        skip: str=[], use: str=[]):    
    create_smoothing_per_zone(fname_points, fname_polygons, folder_out, skip, use)


def create_smoothing_per_zone(fname_points: str, fname_polygons: str,
                              folder_out: str='/tmp',  skip:str=[], use:str = []):
    """
    Creates subsets of points, one for each of the polygons included in the
    `fname_polygons` shapefile. The attibute table must have an 'id' attribute.

    :param fname_points:
        Name of the file with the output of the smoothing
    :param fname_polygons:
        The shapefile with the polygons
    :param folder_out:
        The name of the folder where to write the output
    :returns:
        A number of .csv files in `folder_out`
    """

    create_folder(folder_out)
    
    if len(use) > 0:
        use=get_list(use)

    # Create a geodataframe with the point sources
    df = pd.read_csv(fname_points)
    gdf = gpd.GeoDataFrame(df.drop(['lon', 'lat'], axis=1), crs='epsg:4326',
                           geometry=[Point(xy) for xy in zip(df.lon, df.lat)])

    # Read polygons
    polygons_gdf = gpd.read_file(fname_polygons)

    # Iterate over the polygons defining the boundaries of area sources
    for idx, poly in polygons_gdf.iterrows():

        if poly.id in skip:
            continue
            
        if len(use) > 0 and poly.id not in use:
            continue

        # Create a geodataframe with the polygon in question
        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')

        # Find the point sources from the smoothing inside the polygon
        within = gpd.sjoin(gdf, gdf_poly, predicate='within')

        within['lon'] = within['geometry'].x
        within['lat'] = within['geometry'].y

        if isinstance(poly.id, str):
            fout = os.path.join(folder_out, '{:s}.csv'.format(poly.id))
        else:
            fout = os.path.join(folder_out, '{:d}.csv'.format(poly.id))

        if len(within):
            within.to_csv(fout, index=False, columns=['lon', 'lat', 'nocc'])

main.fname_points = ".csv file created by the smoothing code"
main.fname_polygons = "Shapefile with the polygons"
main.folder_out = "The name of the output folder where to save .csv files"
main.skip = "A string containing a list of source IDs"
main.use = "A string containg a list of source IDs to use"

if __name__ == '__main__':
    sap.run(main)
