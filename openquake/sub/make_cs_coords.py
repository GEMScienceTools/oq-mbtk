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
import glob
import pandas as pd

from openquake.baselib import sap
from openquake.hazardlib.geo.geodetic import azimuth

def make_cs_coords(cs_dir, outfi):
    cs_files = glob.glob(f'{cs_dir}/*csv')
    lines = []
    for fi in cs_files:
        sz = os.path.getsize(fi)
        if sz == 0:
            continue

        df = pd.read_csv(fi, sep=' ', names=["lon", "lat", "depth"])
        az = azimuth(df.lon[0], df.lat[0], df.lon.values[-1], df.lat.values[-1])

        csid = fi.split('/')[-1][3:].replace('.csv','')
        line = f'{df.lon[0]} {df.lat[0]} 700.0 300.0 {az} {csid} cs.ini \n'
        lines.append(line) 
    
    os.remove(outfi) if os.path.exists(outfi) else None

    with open(outfi, 'w') as f:
        for line in lines:
            f.write(line)
    print(f'Written to {outfi}')

make_cs_coords.cs_dir = 'directory with cross section coordinates'
make_cs_coords.outfi = 'output filename'

if __name__ == "__main__":
    sap.run(make_cs_coords)