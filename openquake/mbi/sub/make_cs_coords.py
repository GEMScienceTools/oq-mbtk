#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.sub.make_cs_coords import make_cs_coords

def main(cs_dir, outfi, ini_fname, cs_length=300., cs_depth=300.):
    """
    Creates file with parameters needed to plot cross sections.
    Output file is a list for each cross section with the format:

        lon lat length depth azimuth id <config>.ini

    Example use: 

    oqm sub make_cs_coords openquake/sub/tests/data/cs_cam cs_profiles.csv 
                example.ini 300 300
    """

    make_cs_coords(cs_dir, outfi, ini_fname, cs_length, cs_depth)


make_cs_coords.cs_dir = 'directory with cross section coordinates'
make_cs_coords.outfi = 'output filename'
make_cs_coords.ini_fname = 'name of ini file specifying data paths'
make_cs_coords.cs_length = 'length of cross sections (default 300)'
make_cs_coords.cs_depth = 'depth extent of cross sections (default 300 km)'

if __name__ == '__main__':
    sap.run(main)
