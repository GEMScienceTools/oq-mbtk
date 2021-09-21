#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.seismicity.baseline import add_baseline_seismicity


def main(folder_name: str, folder_name_out: str, fname_config: str,
         fname_poly: str, skip=[]):

    add_baseline_seismicity(folder_name, folder_name_out, fname_config,
                            fname_poly, skip)


main.folder_name = "The name of the folder with smoothing results per source"
main.folder_name_out = "The name of the folder where to store the results"
main.fname_config = ".toml configuration file"
main.fname_poly = "The name of the shapefile with the polygons"
main.skip = "A string containing a list of source IDs"

if __name__ == '__main__':
    sap.run(main)
