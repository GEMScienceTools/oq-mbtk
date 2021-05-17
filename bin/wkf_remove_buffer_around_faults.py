#!/usr/bin/env python
# coding: utf-8

import os
from openquake.baselib import sap
from openquake.wkf.utils import create_folder
from openquake.wkf.distributed_seismicity import remove_buffer_around_faults


def main(fname: str, path_point_sources: str, out_path: str, dst: float):
    create_folder(out_path)
    remove_buffer_around_faults(fname, path_point_sources, out_path, dst)


main.fname = "Pattern for input .xml file with fault sources"
main.path_point_sources = "Pattern for input .xml files"
main.out_path = "Output folder"
main.dst = "Distance [km] of the buffer around the fault"

if __name__ == '__main__':
    sap.run(main)
