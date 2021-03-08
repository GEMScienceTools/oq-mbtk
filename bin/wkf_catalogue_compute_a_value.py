#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.compute_gr_params import compute_a_value


def main(fname_input_pattern: str, bval: float, fname_config: str,
         folder_out: str, *, use: str = '', folder_out_figs: str = None, 
         plt_show: bool = False):

    compute_a_value(fname_input_pattern, bval, fname_config,
                    folder_out, use, folder_out_figs, plt_show)


descr = 'Pattern to select input files or list of files'
main.fname_input_pattern = descr
main.bval = 'GR b-value'
descr = 'Name of the .toml file with configuration parameters'
main.fname_config = descr
descr = 'Name of the output folder where to store occurrence counts'
main.folder_out = descr
main.use = 'A list with the ID of sources that should be considered'
main.folder_out_figs = 'Folder where to store the figures'
main.plt_show = 'Plag controlling the plotting of fiigures on screen'

if __name__ == '__main__':
    sap.run(main)
