#!/usr/bin/env python
# coding: utf-8
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


""" :module:`openquake.mbi.wkf.add_baseline` """

from openquake.baselib import sap
from openquake.wkf.seismicity.baseline import add_baseline_seismicity


def main(folder_name: str, folder_name_out: str, fname_config: str,
         fname_poly: str, *, use: str = [], skip: str = []):
    """
    Add a baseline rate the to the sources modelling distributed seismicity.
    The .toml configuration file contains four parameters defining the
    baseline seismicity in the `[baseline]` section: the `h3_level` (that must
    be consistent with the one used for the smoothing), the `a_value` rate (per
    square km), the `b_value` and the `mmin` values. Moreover, the parameter in
    `add_baseline` must be set to true.

    The output is a .csv file with the same format of the files created by the
    smoothing procedure.
    """
    add_baseline_seismicity(folder_name, folder_name_out, fname_config,
                            fname_poly, skip)


main.folder_name = "The name of the folder with smoothing results per source"
main.folder_name_out = "The name of the folder where to store the results"
main.fname_config = ".toml configuration file"
MSG = "The name of the shapefile with the polygons of the area sources"
main.fname_poly = MSG
main.use = 'A list with the ID of sources that should be considered'
msg = 'A string containing a list of source IDs that will not be considered'
main.skip = msg

if __name__ == '__main__':
    sap.run(main)
