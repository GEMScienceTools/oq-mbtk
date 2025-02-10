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

import pandas as pd
import geopandas as gpd


def to_hmtk_catalogue(cdf: pd.DataFrame, polygon=None):
    """
    Converts a catalogue obtained from the homogenisation into a format
    compatible with the oq-hmtk.

    :param cdf:
        An instance of :class:`pd.DataFrame`
    :param polygon:
        Polygon as shapefile which will be used to clip the catalogue extent 
    :returns:
        An instance of :class:`pd.DataFrame`
    """

    # if there is a polygon clip the catalogue to it
    if polygon:

        # convert df to gdf
        cgdf = pd.DataFrame(cdf)
        tmp = gpd.points_from_xy(cgdf.longitude.values, cgdf.latitude.values)
        cgdf = gpd.GeoDataFrame(cgdf, geometry=tmp, crs="EPSG:4326")

        # Reading shapefile and dissolving polygons into a single one
        boundaries = gpd.read_file(polygon)
        boundaries['dummy'] = 'dummy'
        geom = boundaries.dissolve(by='dummy').geometry[0]

        # clip the catalogue
        tmpgeo = {'geometry': [geom]}
        gdf = gpd.GeoDataFrame(tmpgeo, crs="EPSG:4326")
        cdf = gpd.sjoin(cgdf, gdf, how="inner", op='intersects')

    # Select columns
    # Check if catalogue contains strike/dip/rake and retain if it does
    cdf['Agencies'] = [f'{oA}|{mA}' for oA, mA in zip(cdf.Agency, cdf.magAgency)]
    if 'str1' in cdf.columns:
        col_list = ['eventID', 'Agencies', 'year', 'month', 'day','hour','minute','second', 'longitude',
               'latitude', 'depth', 'magMw', 'sig_tot', 'str1', 'dip1', 'rake1', 'str2', 'dip2', 'rake2']
               #'latitude', 'depth', 'magMw', 'sigma', 'str1', 'dip1', 'rake1', 'str2', 'dip2', 'rake2']
    else:
        col_list = ['eventID', 'Agencies', 'year', 'month', 'day', 'hour','minute','second', 'longitude',
               'latitude', 'depth', 'magMw', 'sig_tot']
               #'latitude', 'depth', 'magMw', 'sigma']
    
    cdf = cdf[col_list]

    # Rename columns
    cdf = cdf.rename(columns={"magMw": "magnitude", "sig_tot": "sigmaMagnitude",
                              "Agencies": "Agency"})

    return cdf


def to_hmtk_catalogue_csv(fname_in: str, fname_out: str):
    """
    Converts a .csv file as obtained from the homogenisation into a .csv file
    woth the oq-hmtk format

    :param cdf:
        Name of the input .csv file
    :returns:
        Name of the output .csv file
    """
    cdf = pd.read_csv(fname_in)
    odf = to_hmtk_catalogue(cdf)
    odf.to_csv(fname_out, index=False)
