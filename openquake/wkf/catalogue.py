
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Type
from shapely.geometry import Point
from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.wkf.utils import create_folder
from openquake.hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT


def to_df(cat: Type[Catalogue]):
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


def from_df(df, end_year=None):
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
    out_fnames = []
    for idx, poly in polygons_gdf.iterrows():

        if len(source_ids) and poly.id not in source_ids:
            continue

        df = pd.DataFrame({'Name': [poly.id], 'Polygon': [poly.geometry]})
        gdf_poly = gpd.GeoDataFrame(df, geometry='Polygon', crs='epsg:4326')
        within = gpd.sjoin(gdf, gdf_poly, op='within')
        # Create output file
        if isinstance(poly.id, int):
            fname = 'subcatalogue_zone_{:d}.csv'.format(poly.id)
        else:
            fname = 'subcatalogue_zone_{:s}.csv'.format(poly.id)
        out_fname = os.path.join(folder_out, fname)
        out_fnames.append(out_fname)
        within.to_csv(out_fname, index=False, columns=columns)
    return out_fnames


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
