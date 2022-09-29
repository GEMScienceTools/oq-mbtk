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
import toml

from openquake.baselib import sap
from openquake.ghm.grid.get_sites import _get_sites

def main(model, folder_out, fname_conf, example=False):

    # Prints an example of configuration file
    if example:
        print(EXAMPLE)
        sys.exit(0)

    # Set model key
    conf = toml.load(fname_conf)

    # Getting the coordinates of the sites
    sites, sites_indices, one_polygon, selection = _get_sites(
        model, folder_out, conf, example)

def _get_site_model(sites):
    """
    :param sites:
    """
    sites



main.model = 'Model key e.g. eur'
main.folder_out = 'Name of the output folder'
main.fname_conf = 'Name of the configuration file'
msg = 'Print an example of configuration and exit'
main.example = msg

if __name__ == '__main__':
    sap.run(main)
