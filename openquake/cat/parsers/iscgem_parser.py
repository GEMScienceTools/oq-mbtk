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
Parser for ISC-GEM catalogue
"""
import pandas as pd
import geopandas as gpd
import re

def parse_iscgem(fname_in, fname_out):
    '''
    Function to parse the ISC-GEM catalogue to generic CSV format. 

    :param fname_in:
        location of input ISCGEM file
    :param fname_out:
        name and location for output file
    '''

    # number of rows in header to skip varies by version
    lineNumber = -1
    with open(fname_in, "r") as in_file:
        for i, line in enumerate(in_file, 1):
            if line.startswith("#"):
                lineNumber = lineNumber + 1
    
    df = pd.read_csv(fname_in, low_memory=False, skiprows = lineNumber - 1, delimiter=',')
    print(lineNumber)
    # First strip whitespace 
    df.columns = df.columns.str.replace(' ', '')
    print(df.columns)
    df.columns[0].removeprefix('#')
    df['date'] = pd.to_datetime(df['date'])
    
    # Creating time-date columns
    df['year'] = df['date'].map(lambda x: x.year)
    df['month'] = df['date'].map(lambda x: x.month)
    df['day'] = df['date'].map(lambda x: x.day)
    df['hour'] = df['date'].map(lambda x: x.hour)
    df['minute'] = df['date'].map(lambda x: x.minute)
    df['second'] = df['date'].map(lambda x: x.second)

    # Cleaning and renaming

    ## Multiple columns called 'unc' - rename these
    cols = []
    count = 1
    for column in df.columns:
        if column == 'unc':
            cols.append(f'unc_{count}')
            count+=1
            continue
        cols.append(column)

    df.columns = cols
    
    df = df.drop(columns=['mrr', 'mtt', 'mpp', 'mrt', 'mpr', 'mtp'])

    df['str1'] = df['str1'].str.strip()
    df['rake1'] = df['rake1'].str.strip()
    df['dip1'] = df['dip1'].str.strip()
    df['str2'] = df['str2'].str.strip()
    df['rake2'] = df['rake2'].str.strip()
    df['dip2'] = df['dip2'].str.strip()

    df = df.rename(columns={"eventid":"eventID",
                            "lon":"longitude",
                            "lat":"latitude",
                            "mw":"magnitude", 
                            "unc_2":"sigmaMagnitude",
                            "smajax":"SemiMajor90",
                            "sminax":"SemiMinor90",
                            "unc_1":"depth_error"})
    
    # Saving data
    df.to_csv(fname_out, index=False)
    print(f"Saving results into the file {fname_out}")
