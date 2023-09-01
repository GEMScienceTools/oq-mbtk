#!/usr/bin/env python
# coding: utf-8

import os
from openquake.baselib import sap
from openquake.wkf.utils import create_folder
from openquake.wkf.distributed_seismicity import remove_buffer_around_faults


def main(fname: str, path_point_sources: str, out_path: str, dst: float,
         threshold_mag: float=6.5, use: str=''):

    # Create the output folder (if needed)
    create_folder(out_path)

    # Process sources
    remove_buffer_around_faults(fname, path_point_sources, out_path, dst,
                                threshold_mag, use)


main.fname = "Pattern for input .xml file with fault sources"
main.path_point_sources = "Pattern for input .xml files"
main.out_path = "Output folder"
main.dst = "Distance [km] of the buffer around the fault"
main.threshold_mag = "Threshold magnitude"
main.use = 'A list with the ID of sources that should be considered'

if __name__ == '__main__':
    sap.run(main)
