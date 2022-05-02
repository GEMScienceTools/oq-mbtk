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

import os
import sys
import geopandas as gpd
from pathlib import Path
from openquake.baselib import sap


def parse(json_fname, output_path):
    #
    # Checking output folder
    if output_path is not None:
        assert os.path.exists(output_path)
    else:
        output_path = ''
    #
    # Parsing the .json file
    df = gpd.read_file(json_fname)
    #
    # Set output filename
    p = Path(os.path.basename(json_fname))
    fname = os.path.join(output_path, p.stem+'.xyz')
    #
    # Set the key
    if 'PGA-0.002107' in df:
        imt_key = 'PGA-0.002107'
    elif 'PGA-0.1' in df:
        imt_key = 'PGA-0.1'
    elif 'PGA-0.002105' in df:
        imt_key = 'PGA-0.002105'
    else:
        raise ValueError('The .json file does not contain the required data')
    #
    # Writing output
    fou = open(fname, 'w')
    for p, z in zip(df['geometry'], df[imt_key]):
        fou.write('{:f},{:f},{:f}\n'.format(p.x, p.y, z))
        print('{:f},{:f},{:f}'.format(p.x, p.y, z))
    fou.close()


def main(argv):

    p = sap.Script(parse)
    p.arg(name='json_fname', help='Name of the .json filename')
    p.opt(name='output_path', help='Output path')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
