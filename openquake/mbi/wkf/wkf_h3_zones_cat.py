#!/usr/bin/env python
# coding: utf-8

import os
import h3
import json
import shapely
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from openquake.baselib import sap
from openquake.wkf.utils import get_list

def main(h3_level: str, fname_catalogue : str, fname_out: str, poly_file: str, use: str = []):
    '''
    Construct H3 zones covering area of extent of catalogue. ToDo: Map cells to source zones. 
    
    :param h3_level: 
        resolution to set h3 cells to
    :param fname_catalogue:
        Location of catalogue csv file
    :param fname_out:
        Folder in which to save the output.
    '''
    
    h3_level = int(h3_level)
    
    df = pd.read_csv(fname_catalogue)
    catalogue = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy
                           in zip(df.longitude, df.latitude)])
                           
    # Read polygons from a file where this is provided
    # Best practice would be to provide this! 
    # Yeah this is exactly what Marco's code does you fool!
    if poly_file:
        polygons_gdf = gpd.read_file(poly_file)
    else:
        # Make boundary polygon from catalogue extent                       
        xmin,ymin,xmax,ymax  = catalogue.total_bounds
        polygon = shapely.geometry.Polygon([(xmin, ymin),  (xmax, ymin), (xmax, ymax), (xmin, ymax),  (xmin, ymin)])
        # Make polygon into geopandas dataframe
        gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])
        gdf['id'] = 1
    
    # Only calculate for spatial polygon ids in use list     
    if len(use) > 0: 
        use = get_list(use)
        use = map(int, use)
        polygons_gdf = polygons_gdf[polygons_gdf['id'].isin(use)]
    # Generate h3 hexagons over the area, save in dataframe 
    fout = open(fname_out, 'w')
    print(len(polygons_gdf))
    for index,row in polygons_gdf.iterrows():
        hexagons = h3.polyfill(row['geometry'].__geo_interface__, h3_level, geo_json_conformant=True)
    
        for hxg in hexagons:
            fout.write("{:s},{:d}\n".format(hxg, row['id']))    


main.h3_level = "h3 resolution level"
main.fname_catalogue = "file containing an earthquake catalogue in .csv format"
main.fname_out = "The name of the output folder where to save .csv files"


if __name__ == '__main__':
    sap.run(main)
