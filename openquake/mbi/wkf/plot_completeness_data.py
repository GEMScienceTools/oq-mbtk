#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.plot.completeness import completeness_plot


def main(fname_input_pattern, fname_config, outdir, *, skip=[], yealim=''):
    """
    Plots completeness table and earthquake occurrence using a 2D matrix
    """
    completeness_plot(fname_input_pattern, fname_config, outdir, skip, yealim)


descr = 'Pattern for the .csv catalogue files'
main.fname_input_pattern = descr
descr = 'Name of the .toml file with configuration parameters'
main.fname_config = descr
main.outdir = 'Name of the output folder'
main.yealim = 'Year range used in the plot'


if __name__ == '__main__':
    sap.run(main)
