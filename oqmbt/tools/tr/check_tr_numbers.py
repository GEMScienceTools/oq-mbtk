#!/usr/bin/env python

import sys
import h5py
import numpy as np

from openquake.baselib import sap


def check(fname):
    """
    """

    f = h5py.File(fname, 'r')
    keys = sorted(f.keys())

    for i1, k1 in enumerate(keys):
        for i2 in range(i1, len(keys)):
            k2 = keys[i2]
            if k1 == k2:
                continue
            chk = np.logical_and(f[k1], f[k2])
            if any(chk):
                print('{:s} and {:s} have common eqks'.format(k1, k2))
                print(len(chk))
    #
    # Check total numbers
    for key in f:
        print('{:10s}: {:5d}'.format(key, sum(f[key])))
    #
    # close hdf5
    f.close()


def main(argv):
    """
    """
    p = sap.Script(check)
    p.arg(name='fname', help='TR hdf5 filename')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    main(sys.argv[1:])
