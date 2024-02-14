#!/usr/bin/env python
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

import re
import os
import sys
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from openquake.baselib import sap
from openquake.hazardlib.probability_map import compute_hazard_maps


def read_all_maps(data_path):
    """
    :param data_path: 
        path to map json files

    returns a geodataframe with all the maps joined
    """

    dfm = gpd.GeoDataFrame()

    # read in all files into one dataframe
    for fname in sorted(glob.glob(data_path)):
        print(fname)
        df = gpd.read_file(fname)
        dfm = gpd.GeoDataFrame( pd.concat([df, dfm], ignore_index=True) )
    return dfm


def create_map(path_in, prefix, fnames_out, path_out, imt_str, pexs):
    """
    :param path_in: 
    """
    from time import process_time 
    t1_start = process_time()  

    lons = []
    lats = []
    poes = []

    # set datapath to maps and read them
    data_path = os.path.join(path_in, f'{prefix}*json')
    dfm = read_all_maps(data_path)

    t2_read = process_time()  
    t12 = t2_read - t1_start
    print(f'Time to read: {t12}s')

    
    # get the coordinates 
    lons = [p.x for p in dfm.geometry]
    lats = [p.y for p in dfm.geometry]
    
    # take the columns with the poe data
    poelabs = [k for k in dfm.keys() if 'poe' in k]
    poes = dfm[poelabs]
    poesnp = np.array(poes)

    # get the imls 
    imlsnp = np.array([float(k.replace('poe-','')) for k in poelabs])

    # get map values from curves via engine interpolation
    mapvals = compute_hazard_maps(poesnp, imlsnp, pexs)

    t3_read = process_time()  
    t23 = t3_read - t2_read
    print(f'Time to get maps: {t23}s')

    # write to files
    for i, pex in enumerate(pexs):
        fout = os.path.join(path_out, fnames_out[i])
        dfout = pd.DataFrame({'lon': lons, 'lat': lats, f'{imt_str}-{pex}': mapvals[:,i]})
        dfout.to_csv(fout, index=False, float_format='%.5f')



def map(path_in, prefix, fnames_out, path_out, imt_str, pexs=None):
    """
    Creates a hazard map from a set of hazard curves

    Example use:

    ./create_map_from_curves.py PGA-rock map "['PGA-rock-475.csv', 'PGA-rock-2475.csv']" 
    maps_out PGA "[0.002105, 0.000404]"
    """

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    pexs = eval(pexs)
    fnames_out = eval(fnames_out)
    create_map(path_in, prefix, fnames_out, path_out, imt_str, pexs)


map.path_in = 'Name of the folder with input .json files'
map.prefix = 'Prefix for selecting files'
map.fnames_out = 'List with names of output csv files'
map.path_out = 'Path to the output folder'
map.imt_str = 'String describing the IMT'
map.pexs = 'List with probabilities of exceedance, same lenght as fnames_out'


if __name__ == "__main__":
    sap.run(map)
