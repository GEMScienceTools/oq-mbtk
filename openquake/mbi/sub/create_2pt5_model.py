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

import sys

from pyproj import Geod
from openquake.baselib import sap
from openquake.sub.create_2pt5_model import create_2pt5_model
from openquake.hazardlib.geo.geodetic import distance


def main(in_path, out_path, *, maximum_sampling_distance=25.)
    """
    From a set of profiles it creates the top surface of the slab
    """

    if in_path == out_path:
        tmps = '\nError: the input folder cannot be also the output one\n'
        tmps += '    input: {0:s}\n'.format(in_path)
        tmps += '    input: {0:s}\n'.format(out_path)
        print(tmps)
        exit(0)

    create_2pt5_model(in_path, out_path, float(maximum_sampling_distance), 
                      start, end)

main.in_path = 'Folder with the profiles'
main.out_path = 'Folder where to store the output'
main.maximum_sampling_distance = 'Sampling distance [km]'

if __name__ == "__main__":
    main(sys.argv[1:])
