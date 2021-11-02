#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (c) 2019 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import re
import os
import sys
import glob
import numpy as np
import geopandas as gpd
from openquake.baselib import sap
from openquake.man.tools import csv_output as csvt


def read_hazard_curve_files(path_in, prefix=''):
    """
    :param path_in:
    :param pattern:
    """
    data_path = os.path.join(path_in, '{:s}*'.format(prefix))
    lons = []
    lats = []
    poes = []
    for i, fname in enumerate(sorted(glob.glob(data_path))):
        print(fname)
        df = gpd.read_file(fname)
        # Extract IMLs from names
        if i == 0:
            imls = []
            for tmps in list(df.columns):
                m = re.search(r'^poe-(\d*\.\d*)', tmps)
                if m:
                    imls.append(float(m.group(1)))
        # Save coordinates
        for p in df['geometry']:
            lons.append(p.x)
            lats.append(p.y)
        # Save data
        tmp_poes = df.filter(regex=("poe*")).values
        for row in list(tmp_poes):
            poes.append(row)
    return lons, lats, np.array(poes), np.array(imls)


def write_hazard_map(filename, lons, lats, pex, gms, imt):
    fou = open(filename, 'w')
    fou.write('# mean, investigation_time=1.0\n')
    fou.write('lon,lat,{:s}-{:f}\n'.format(imt, pex))
    for lo, la, gm in zip(lons, lats, gms):
        fou.write('{:f},{:f},{:e}\n'.format(lo, la, gm))
    fou.close()
    print('Created:\n{:s}'.format(filename))


def create_map(path_in, prefix, fname_out, path_out, imt_str, pex=None,
               iml=None):
    """
    :param fname:
        Name of the .csv file with the hazard curves
    """
    pex = float(pex)
    lons, lats, poes, imls = read_hazard_curve_files(path_in, prefix)
    # Check is output path exists
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    # Read the file with the hazard curves
    # lons, lats, poes, hea, imls = csvt.read_hazard_curve_csv(fname_csv)
    # Compute the hazard maps
    if pex is not None and iml is None:
        dat = csvt.get_map_from_curves(imls, poes, pex)
    elif pex is None and iml is not None:
        raise ValueError('Not yet supported')
    else:
        raise ValueError('You cannot set both iml and pex')
    # Save the hazard map
    path_out = os.path.join(path_out, fname_out)
    write_hazard_map(path_out, lons, lats, pex, dat, imt_str)


def map(path_in, prefix, fname_out, path_out, imt_str, pex=None, iml=None):
    """
    Creates a hazard map from a set of hazard curves
    """
    create_map(path_in, prefix, fname_out, path_out, imt_str, pex, iml)


map.path_in = 'Name of the file with input .json files'
map.prefix = 'Prefix for selecting files'
map.fname_out = 'Name output csv file'
map.path_out = 'Path to the output folder'
map.imt_str = 'String describing the IMT'
map.pex = 'Probability of exceedance'
map.iml = 'Intensity measure level used for building the maps'


if __name__ == "__main__":
    sap.run(map)
