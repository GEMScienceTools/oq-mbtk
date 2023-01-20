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

def main(h3_level: str, fname_catalogue : str, fname_out: str):
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
    
    # Make boundary polygon from catalogue extent                       
    xmin,ymin,xmax,ymax  = catalogue.total_bounds
    polygon = shapely.geometry.Polygon([(xmin, ymin),  (xmax, ymin), (xmax, ymax), (xmin, ymax),  (xmin, ymin)])

    gojpol = eval(json.dumps(shapely.geometry.mapping(polygon)))
    
    # Revert the positions of lons and lats
    coo = [[c[1], c[0]] for c in gojpol['coordinates'][0]]
    gojpol['coordinates'] = [coo]
    
    # Generate h3 hexagons over the area, save in dataframe
    hexagons = list(h3.polyfill(gojpol, h3_level))
    
    fout = open(fname_out, 'w')
    for hxg in hexagons:
        # TODO: Update this to take correct source zone names from source zone polygons
        zid = 1
        fout.write("{:s},{:d}\n".format(hxg, zid))

main.h3_level = "h3 resolution level"
main.fname_catalogue = "file containing an earthquake catalogue in .csv format"
main.fname_out = "The name of the output folder where to save .csv files"


if __name__ == '__main__':
    sap.run(main)
