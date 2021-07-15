#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.compute_gr_params import compute_a_value_from_density


def main(fname_input_pattern: str, fname_config: str, fname_polygons: str,
         *, use: str = '', folder_out_figs: str = None,
         plt_show: bool = False):

    compute_a_value_from_density(
        fname_input_pattern, fname_config, fname_polygons,
        use, folder_out_figs, plt_show)


descr = 'Pattern to select input files or list of files'
main.fname_input_pattern = descr
descr = 'Name of the .toml file with configuration parameters'
main.fname_config = descr
descr = 'Name of the file (.shp, .geojson) with polygons'
main.fname_polygons = descr
main.use = 'A list with the ID of sources that should be considered'
main.folder_out_figs = 'Folder where to store the figures'
main.plt_show = 'Plag controlling the plotting of fiigures on screen'

if __name__ == '__main__':
    sap.run(main)
