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
Module :module:`~openquake.ghm.utils`
"""

import re
import copy
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon


def explode(indf):
    """
    Implements what's suggested here: http://goo.gl/nrRpdV

    :param indf:
        A geodataframe instance
    :returns:
        A geodataframe instance
    """
    indf = pd.DataFrame(indf)

    # Adding column and index
    #indf['idx'] = np.arange(len(indf))
    #indf.set_index('idx')

    # Create output dataframe
    outdf = pd.DataFrame(columns=indf.columns)

    for idx, row in indf.iterrows():

        if type(row.geometry) == Polygon:
            outdf = pd.concat([outdf, row.to_frame()], ignore_index=True)

        if type(row.geometry) == MultiPolygon:

            # Create a DataFrame
            vals = {}
            for key in row.keys():
                vals[key] = row[key]
            multdf = pd.DataFrame(vals)
            #tmp = pd.DataFrame(vals)

            # Duplicate rows
            #recs = len(row.geometry)
            #for i in range(recs-1):
            #    multdf = pd.concat([multdf, tmp], ignore_index=True)
            #breakpoint()

            # Updating geometry column
            #for i in range(recs):
            #    multdf.loc[i, 'geometry'] = row.geometry[i]

            outdf = pd.concat([outdf, multdf], ignore_index=True)

    return gpd.GeoDataFrame(outdf, geometry='geometry')


def read_hazard_map_csv(fname):
    """
    Read the content of a .csv file with the mean hazard map computed for a
    given hazard model.

    :param str fname:
        The name of the file containing the results
    :return:
        A dictionary with key the sting in the header and values the floats
        in the csv file
    """
    data = {}
    for line in open(fname):
        if re.search('^#', line):
            pass
        else:
            if re.search('lon', line):
                labels = re.split('\,', line)
            else:
                aa = re.split('\,', line)
                for l, d in zip(labels, aa):
                    if l in data:
                        data[l].append(float(d))
                    else:
                        data[l] = [float(d)]
    return data


def create_query(inpt, field, labels):
    """
    Creates a query

    :param inpt:
    :param field:
    :param labels:
    :returns:
    """
    sel = None
    for lab in labels:
        if sel is None:
            sel = inpt[field] == lab
        else:
            sel = sel | (inpt[field] == lab)
    return inpt.loc[sel]
