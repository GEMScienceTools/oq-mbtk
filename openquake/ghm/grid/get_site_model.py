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
import subprocess
import tempfile
import numpy as np
import toml

from openquake.baselib import sap
from openquake.ghm.grid.get_sites import _get_sites, EXAMPLE

CODE = os.path.join('..', '..', 'mbt', 'tools', 'site',
                    'create_site_models_julia.jl')
PWD = os.path.join(os.path.dirname(__file__))


def main(model, folder_out, fname_conf, *, example=False):
    """
    This code creates a site model given the code of a model in the mosaic.
    """

    model = model.lower()

    # Prints an example of configuration file
    if example:
        print(EXAMPLE)
        sys.exit(0)

    # Set model key
    conf = toml.load(fname_conf)

    # Getting the coordinates of the sites
    sites, _, _, _ = _get_sites(model, folder_out, conf, example)

    # Write sites to a temporary .csv file
    folder_tmp = tempfile.mkdtemp()
    fname_sites = os.path.join(folder_tmp, 'sites.csv')
    np.savetxt(fname_sites, sites, delimiter=",")

    # Create site model
    res = conf['main']['h3_resolution']
    fname_out = os.path.join(folder_out, f'{model}_res{res}.csv')
    code = os.path.join(PWD, CODE)
    cmd = f"julia {code} '{fname_conf}' '{fname_sites}' '{fname_out}'"
    subprocess.call(cmd, shell=True)


main.model = 'Model key e.g. eur'
main.folder_out = 'Name of the output folder'
main.fname_conf = 'Name of the configuration file'
MSG = 'Print an example of configuration and exit'
main.example = MSG

if __name__ == '__main__':
    sap.run(main)
