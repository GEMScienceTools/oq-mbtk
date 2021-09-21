#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.h3.zones import discretize_zones_with_h3_grid


def main(h3_level: str, fname_poly: str, folder_out: str):
    """
    Given a set of polygons, using H3 this creates grids representing in a
    'discrete' sense the original geometries
    """
    discretize_zones_with_h3_grid(h3_level, fname_poly, folder_out)


descr = 'The level of the H3 grid'
main.h3_level = descr
descr = 'The name of the shapefile with polygons'
main.fname_poly = descr
descr = 'The name of the folder where to save output'
main.folder_out = descr

if __name__ == '__main__':
    sap.run(main)
