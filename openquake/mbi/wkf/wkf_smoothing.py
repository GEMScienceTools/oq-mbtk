#!/usr/bin/env python
# coding: utf-8

import os
from openquake.baselib import sap
from openquake.wkf.utils import create_folder
from openquake.wkf.seismicity.smoothing_utils import smoothing


def main(count: str, config: str, fname_out: str):
    """
    Smooths the results of box counting using a mixture of gaussian kernels
    """
    folder = os.path.dirname(fname_out)
    if len(folder) > 0 and not os.path.isdir(folder):
        create_folder(folder)
    smoothing(count, config, fname_out)


main.count = 'Name of the .csv file produced by box counting'
main.config = 'Name of the .toml configuration file'
main.fname_out = 'Name of the output .csv file'

if __name__ == '__main__':
    sap.run(main)
