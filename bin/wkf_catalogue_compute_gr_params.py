#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.compute_gr_params import weichert_analysis


def main(fname_input_pattern, fname_config, folder_out=None, 
         folder_out_figs=None, *, skip=[], binw=None, plt_show=False):

    weichert_analysis(fname_input_pattern, fname_config, folder_out,
                      folder_out_figs, skip, binw, plt_show)

main.fname_input_pattern = 'Name of a shapefile with polygons'
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the output folder where to store occurrence counts'
main.folder_out = msg
msg = 'Name of the output folder where to store figures'
main.folder_out_figs = msg
msg = 'A list with the ID of sources that should not be considered'
main.skip = msg
main.binw = 'Width of the magnitude bin used in the analysis'
main.plot_show = 'Show figures on screen'

if __name__ == '__main__':
    sap.run(main)
