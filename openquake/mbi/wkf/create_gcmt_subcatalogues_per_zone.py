#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.catalogue import create_gcmt_files


def main(fname_polygons: str, gcmt_filename: str, folder_out: str,
         depth_max: float=600.0, depth_min:float=0.0):

    create_gcmt_files(fname_polygons, gcmt_filename, folder_out, depth_max,
                      depth_min)


main.fname_polygons = 'Name of a shapefile with polygons'
main.fname_gcmt = 'Name of the .csv file with the catalog'
main.folder_out = 'Name of the output folder'
main.depth_max = 'Maximum depth [km]'
main.depth_min = 'Minimum depth [km]'

if __name__ == '__main__':
    sap.run(main)
