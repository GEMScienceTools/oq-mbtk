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

import toml
from openquake.baselib import sap
from openquake.wkf.utils import create_folder
from openquake.sub.get_profiles_from_slab2pt0 import get_profiles


def main(fname_conf: str):
    """
    Given the Slab2.0 strike and depth .grd files, this code creates a set
    of profiles describing the surface of the slab.

    :param fname_conf:
        Name of the .toml formatted file. If the configuration file defines
        the `output_folder` variable, the profiles are saved in that folder
        for further use. If the configuration file defines the `fname_fig`
        variable, a figure with the traces is saved.

        Example of .toml file
        ```
        fname_str ='./str_grd/cam_slab2_str_02.24.18.grd'
        fname_dep ='./dep_grd/cam_slab2_dep_02.24.18.grd'
        spacing = 200.0
        folder_out = './cs'
        fname_fig = './cam.pdf'
        ```
    """

    # Read configuration file
    conf = toml.load(fname_conf)

    # Name of the .grd file with the strike values
    fname_dep = conf.get('fname_dep', None)

    # Name of the .grd file with the strike values
    fname_str = conf.get('fname_str', None)

    # Name of the .grd file with the strike values
    spacing = conf.get('spacing', 100.)

    # Name of the folder where to save the profiles
    folder_out = conf.get('folder_out', None)

    # Name of the figure
    fname_fig = conf.get('fname_fig', '')

    # Get profiles
    slb = get_profiles(fname_str, fname_dep, spacing, fname_fig)

    # Save profiles
    if folder_out is not None:
        create_folder(folder_out)
        slb.write_profiles(folder_out)


main.fname_conf = 'Name of .toml file with configuration parameters'

if __name__ == '__main__':
    sap.run(main)
