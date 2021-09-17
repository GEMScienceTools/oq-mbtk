#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.seismicity.nodal_plane import process_gcmt_datafames


def main(fname_folder, folder_out):
    process_gcmt_datafames(fname_folder, folder_out)


main.fname_folder = 'Name of the folder with input files'
main.folder_out = 'Name of the output folder'

if __name__ == '__main__':
    sap.run(process_gcmt_datafames)
