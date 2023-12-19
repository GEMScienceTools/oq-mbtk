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

"""
Module :mod:`openquake.wkf.catalogue` contains tools for working with catalogue
files in the hmtk format
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Type
from shapely.geometry import Point
from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.wkf.utils import create_folder
from openquake.hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT
from openquake.hmtk.seismicity.selector import CatalogueSelector


from openquake.hmtk.parsers.catalogue import CsvCatalogueParser   # Reads an earthquake catalogue from CSV
from openquake.hmtk.parsers.catalogue.csv_catalogue_parser import CsvCatalogueWriter  # Writes an earthquake catalogue to CSV
from openquake.mbt.tools.general import _get_point_list
from openquake.mbt.tools.geo import get_idx_points_inside_polygon
#from openquake.mbt.oqt_project import OQtSource
from osgeo import ogr
from shapely import wkt
from copy import deepcopy
#from openquake.hazardlib.geo.polygon import Polygon

from openquake.mbt.tools.area import load_geometry_from_shapefile, create_catalogue

def extract(fname_in: str, **kwargs) -> pd.DataFrame:
    """
    Creates a copy of a .csv catalogue containing only the events responding
    to the selection criteria. Accepted parameters:
        - Minimum hypocentral depth: min_depth
        - Maximum hypocentral depth: min_depth
        - Minimum magnitude: min_mag
        - Maximum magnitude: max_mag

    :param fname_in:
        Name of the input .csv file with the catalogue
    :param fname_out:
        Name of the output .csv file with the catalogue
    :returns:
        A dataframe with the filtered catalogue
    """

    # Reads the catalogue
    df = pd.read_csv(fname_in, comment="#")

    # Filter the catalogue
    for key in kwargs:
        if key == 'min_depth':
            df.query(f"depth > {kwargs['min_depth']}", inplace=True)
        elif key == 'max_depth':
            df.query(f"depth < {kwargs['max_depth']}", inplace=True)
        elif key == 'min_mag':
            df.query(f"magnitude >= {kwargs['min_mag']}", inplace=True)
        elif key == 'max_mag':
            df.query(f"magnitude < {kwargs['max_mag']}", inplace=True)

    # Return the final catalogue
    df.reset_index()

    if len(df) < 1:
        print('Empty catalogue!')
    else:
        print(f'Catalogue contains {len(df):d} earthquakes')

    return df


def to_df(cat: Type[Catalogue]) -> pd.DataFrame:
    """
    Converts an :class:`openquake.hmtk.seismicity.catalogue.Catalogue` instance
    into a dataframe

    :param cat:
        The catalogue instance
    :returns:
        The dataframe with the catalogue
    """
    df = pd.DataFrame()
    for key in cat.data:
        if key not in ['comment', 'flag'] and len(cat.data[key]):
            df.loc[:, key] = cat.data[key]
    return df


def from_df(df, end_year=None) -> Type[Catalogue]:
    """
    Converts a dataframe into a
    :class:`openquake.hmtk.seismicity.catalogue.Catalogue` instance

    :param df:
        The dataframe with the catalogue
    :returns:
        The catalogue instance
    """
    cat = Catalogue()
    for column in df:
        if (column in Catalogue.FLOAT_ATTRIBUTE_LIST or
                column in Catalogue.INT_ATTRIBUTE_LIST):
            cat.data[column] = df[column].to_numpy()
        else:
            cat.data[column] = df[column]
    cat.end_year = np.max(df.year) if end_year is None else end_year
    return cat


def create_subcatalogues(fname_polygons: str, fname_cat: str, folder_out: str,
                         source_ids: list = []):
    """
    Given a catalogue and a gis-file with polygons (e.g. shapefile or
    .geojson), this code creates for each polygon a subcatalogue with the
    earthquakes with epicenters in the polygon.

    ** Does not work with polygons that cross the international dateline

    :param fname_polygons:
        The name of the gis file containing the polygons.
    :param fname_cat:
        The name of the file with the catalogue (hmtk formatted)
    :param folder_out:
        The name of the output folder where to create the output .csv files
        containing the subcatalogues
    :param source_ids:
        [optional] The list of source ids to be considered. If omitted all the
        polygons will be considered.
    """

    # Create output folder
    create_folder(folder_out)

    # Create geodataframe with the catalogue
    df = pd.read_csv(fname_cat)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy
                           in zip(df.longitude, df.latitude)])

    # Read polygons
    polygons_gdf = gpd.read_file(fname_polygons)
    

    # Select point in polygon
    columns = ['eventID', 'year', 'month', 'day', 'hour', 'minute', 'second', 'magnitude', 'longitude',
               'latitude', 'depth']

    # Iterate over sources
    out_fnames = []
    for idx, poly in polygons_gdf.iterrows():
        if len(source_ids) > 0 and int(poly.id) not in eval(source_ids):
            continue
        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')
        within = gpd.sjoin(gdf, gdf_poly, op='within')
        # Create output file
        if isinstance(poly.id, int):
            fname = f'subcatalogue_zone_{poly.id:d}.csv'
        else:
            fname = f'subcatalogue_zone_{poly.id}.csv'
        out_fname = os.path.join(folder_out, fname)
        out_fnames.append(out_fname)
        within.to_csv(out_fname, index=False, columns=columns)

    return out_fnames

def create_subcatalogues_nongeopd(fname_polygons: str, fname_cat: str, folder_out: str,
                         source_ids: list = []):
    """
    Given a catalogue and a gis-file with polygons (e.g. shapefile or
    .geojson), this code creates for each polygon a subcatalogue with the
    earthquakes with epicenters in the polygon.

    ** Based on old mbtk functions for subcats so no relying on geopandas


    :param fname_polygons:
        The name of the gis file containing the polygons.
    :param fname_cat:
        The name of the file with the catalogue (hmtk formatted)
    :param folder_out:
        The name of the output folder where to create the output .csv files
        containing the subcatalogues
    :param source_ids:
        [optional] The list of source ids to be considered. If omitted all the
        polygons will be considered.
    """

    # Create output folder
    create_folder(folder_out)

    # Create geodataframe with the catalogue
    #df = pd.read_csv(fname_cat)
    #gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=[Point(xy) for xy
                           #in zip(df.longitude, df.latitude)])
    
    parser = CsvCatalogueParser(fname_cat)
    catalogue = parser.read_file() 
    # Read polygons
    #polygons_gdf = gpd.read_file(fname_polygons)
    
    # Process catalogue
    neqk = len(catalogue.data['longitude'])
    
    pnt_idxs = [i for i in range(0, neqk)]

    
    
    idname = 'Id'

    # Set the driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(fname_polygons, 0)
    layer = datasource.GetLayer()

    # Reading sources geometry
    sources = {}
    id_set = set()
    out_fnames = []
    nfeatures = layer.GetFeatureCount()
    #sel_idx = np.full((neqk), False, dtype=bool)
    
    for feature in layer:
        sel_idx = np.full((neqk), False, dtype=bool)
        cat = deepcopy(catalogue)
    
        geom = feature.GetGeometryRef()
        polygon = wkt.loads(geom.ExportToWkt())
        x, y = polygon.exterior.coords.xy
    
        if isinstance(feature.GetField(idname), str):
            id_str = feature.GetField(idname)
        elif isinstance(feature.GetField(idname), int):
            id_str = '%d' % (feature.GetField(idname))
    
        #x, y = poly.exterior.coords.xy
        idxs = get_idx_points_inside_polygon(catalogue.data['longitude'],
                                         catalogue.data['latitude'],
                                         x, y,
                                         pnt_idxs, buff_distance=0.)
        sel_idx[idxs] = True
    
        # Select earthquakes
        selector = CatalogueSelector(cat, create_copy=False)
        selector.select_catalogue(sel_idx)

        fname = f'subcatalogue_zone_{id_str}.csv'
        out_fname = os.path.join(folder_out, fname)
        out_fnames.append(out_fname)
        
        print(out_fname)
        cat_csv = CsvCatalogueWriter(out_fname)
        cat_csv.write_file(cat)
        
    layer.ResetReading()

    return out_fnames


def get_dataframe(fname: str) -> pd.DataFrame:
    """
    Creates a dataframe with the information included in a .ndk formatted
    file. For a description of the .ndk format see: https://www.globalcmt.org/

    :param fname:
        Name of the .ndk file
    :returns:
        A dataframe with the information in the .ndk file
    """
    parser = ParseNDKtoGCMT(fname)
    cat_gcmt = parser.read_file()
    df = pd.DataFrame({k: cat_gcmt.data[k] for k in cat_gcmt.data.keys()})
    return df


def create_gcmt_files(fname_polygons: str, gcmt_filename: str, folder_out: str,
                      depth_max: float = 600.0, depth_min: float = 0.0):

    # Create output folder
    create_folder(folder_out)

    # Create geodataframe with the catalogue
    print(os.path.abspath(gcmt_filename))
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

        if len(df) < 1:
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
