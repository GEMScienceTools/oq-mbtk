#!/usr/bin/env python
# coding: utf-8

import os
import numpy
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from openquake.baselib import sap
from utils import create_folder, _get_src_id
from shapely.geometry import Point
from openquake.hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT


def get_dataframe(fname):
    print(fname)
    parser = ParseNDKtoGCMT(fname)
    cat_gcmt = parser.read_file()
    df = pd.DataFrame({k: cat_gcmt.data[k] for k in cat_gcmt.data.keys()})
    return df


def create_gcmt_files(fname_polygons: str, gcmt_filename: str, folder_out: str,
                      depth_max: float=600.0, depth_min:float=0.0):

    # Create output folder
    create_folder(folder_out)

    # Create geodataframe with the catalogue 
    tmp = get_dataframe(gcmt_filename)
    
    # Filter depths
    df = tmp[(tmp.depth > depth_min) & (tmp.depth <= depth_max)]
    if len(df) < 0:
        return []
    
    # Create geodataframe
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy
                           in zip(df.longitude, df.latitude)])
    
    # Read polygons
    polygons_gdf = gpd.read_file(fname_polygons)
    
    # Iterate over sources
    fnames_list = []
    for idx, poly in polygons_gdf.iterrows():
        
        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')
        within = gpd.sjoin(gdf, gdf_poly, op='within')

        if len(df) < 0:
            continue
        
        # Create output file
        if isinstance(poly.id, int):
            fname = 'subcatalogue_zone_{:d}.csv'.format(poly.id)
        else:
            fname = 'subcatalogue_zone_{:s}.csv'.format(poly.id)
        out_fname = os.path.join(folder_out, fname)
        within.to_csv(out_fname, index=False)
        
        fnames_list.append(out_fname)
        
    return fnames_list


create_gcmt_files.fname_polygons = 'Name of a shapefile with polygons'
create_gcmt_files.fname_gcmt = 'Name of the .csv file with the catalog'
create_gcmt_files.folder_out = 'Name of the output folder'
create_gcmt_files.depth_max = 'Maximum depth [km]'
create_gcmt_files.depth_min = 'Minimum depth [km]'

if __name__ == '__main__':
    sap.run(create_gcmt_files)
