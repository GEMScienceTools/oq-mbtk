#!/usr/bin/env python
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
