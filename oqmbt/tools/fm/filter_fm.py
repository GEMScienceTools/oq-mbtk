#!/usr/bin/env python3

import re
import sys

from openquake.baselib import sap
from openquake.hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT


def filter_gcmt(ndf_fname, limits):
    #
    #
    lims = [float(v) for v in re.split('\,', limits)]
    #
    #
    with open(ndf_fname , 'r') as f:
        while True:
            lines = []
            for i in range(5):
                try:
                    lines.append(next(f))
                except StopIteration:
                    return

            aa = re.split('\s+', lines[0])
            lon = float(aa[4])
            lat = float(aa[3])
            # lat, lon
            if (lat > lims[1] and  lat < lims[3] and
                lon > lims[0] and  lon < lims[2]):
                for line in lines:
                    print('{:s}'.format(line.rstrip()))


def main(argv):

    p = sap.Script(filter_gcmt)
    p.arg(name='ndf_fname', help='Name of the file with .ndk format')
    p.arg(name='limits', help='A list with: lomin, lomax, lamin, lamax')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()

if __name__ == '__main__':
    main(sys.argv[1:])
