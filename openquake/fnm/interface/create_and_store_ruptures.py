#!/usr/bin/env python
# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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
import pathlib

from openquake.baselib import sap
from openquake.fnm.datastore import write
from openquake.fnm.importer import kite_surfaces_from_geojson
from openquake.fnm.fault_system import get_rups_fsys


def main(settings_fname: str, **kwargs):

    # Read the configuration
    settings = toml.load(settings_fname)

    for key in kwargs:
        if isinstance(kwargs[key], dict):
            for subkey in kwargs[key]:
                if key not in settings:
                    settings[key] = {}
                settings[key][subkey] = kwargs[key][subkey]
        else:
            print('aa')
            settings[key] = kwargs[key]

    # Path to the settings file. This is the root folder.
    root = pathlib.Path(settings_fname).parent

    # Get the name of the file with faults. This can either formatted as a
    # .geojson or a .shapefile. TODO need to add a check on the attributes.
    fname_faults = str(root / settings['general']['fname_sections'])

    # Create surfaces using the information in the .geojson file
    surfs = kite_surfaces_from_geojson(fname_faults)

    # Create the ruptures
    out = get_rups_fsys(surfs, settings)

    # Save results
    fname = str(root / settings['general']['output_datastore'])
    write(fname, out)


descr = "The name of the file with the settings"
main.settings_fname = descr

if __name__ == "__main__":
    sap.run(main)
