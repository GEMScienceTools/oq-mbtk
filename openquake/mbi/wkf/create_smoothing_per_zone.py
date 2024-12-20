#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.seismicity.smoothing import create_smoothing_per_zone


def main(fname_points: str, fname_polygons: str, folder_out: str='/tmp', 
         skip: str=[], use:str=[]):
    create_smoothing_per_zone(fname_points, fname_polygons, folder_out, skip, use)


main.fname_points = ".csv file created by the smoothing code"
main.fname_polygons = "Shapefile with the polygons"
main.folder_out = "The name of the output folder where to save .csv files"
main.skip = "A string containing a list of source IDs to skip"
main.use = "A string containing a list of source IDs to use"

if __name__ == '__main__':
    sap.run(main)
