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
from openquake.baselib import sap
from openquake.wkf.catalogue import extract
from openquake.wkf.utils import create_folder


def main(fname_in: str, fname_out: str, *, depth_min: float = 0,
         depth_max: float = 1000, min_mag: float = -1, max_mag: float = 15):
    """
    Extact from a .csv file catalogue the essential information required for
    hazard modelling. Some filtering options are also available.
    """
    folder_out = os.path.dirname(fname_out)
    create_folder(folder_out)
    kwargs = {'depth_min': depth_min, 'depth_max': depth_max,
              'min_mag': min_mag, 'max_mag': max_mag}
    cdf = extract(fname_in, **kwargs)
    cdf.to_csv(fname_out, index=False)


main.fname_in = "The name of the input catalogue"
main.fname_out = "The name of the output catalogue"
main.depth_min = "Minimum depth [km]"
main.depth_max = "Maximum depth [km]"
main.min_mag = "Minimum magnitude (included)"
main.max_mag = "Maximum magnitude (excluded)"

if __name__ == '__main__':
    sap.run(main)
