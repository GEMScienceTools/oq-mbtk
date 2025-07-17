#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.plot_incremental_mfd import plot_incremental_mfds

def main(fname_input_pattern, fname_config,
         folder_out_figs=None, *, skip=[], binw=0.1, plt_show=False):

    plot_incremental_mfds(fname_input_pattern, fname_config,
                      folder_out_figs, skip, binw, plt_show)

msg = 'A string (defining a pattern) or a list of .csv files '
msg += 'with subcatalogues'
main.fname_input_pattern = msg
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the output folder where to store figures'
main.folder_out_figs = msg
msg = 'A list with the ID of sources that should not be considered'
main.skip = msg
msg = 'Width of the magnitude bin used in the analysis'
main.binw = msg
msg = 'Show figures on screen'
main.plot_show = msg

if __name__ == '__main__':
    sap.run(main)