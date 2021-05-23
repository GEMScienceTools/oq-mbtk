#!/usr/bin/env python

import sys
import h5py
import numpy as np

from openquake.baselib import sap


def check(fname):
    """
    Check the numbers of classified earthquakes
    """

    f = h5py.File(fname, 'r')
    keys = sorted(f.keys())
    print('Total number of earthquakes: {:d}'.format(len(f[keys[0]])))

    for i1, k1 in enumerate(keys):
        for i2 in range(i1, len(keys)):
            k2 = keys[i2]
            if k1 == k2:
                continue
            chk = np.logical_and(f[k1], f[k2])
            if any(chk):
                print('{:s} and {:s} have common eqks'.format(k1, k2))
                print(len(chk))

    # Check total numbers
    tot = 0
    for key in f:
        smm = sum(f[key])
        print('{:10s}: {:5d}'.format(key, smm))
        tot += smm
    print(tot, '/', len(f[key]))

    # Close hdf5
    f.close()


check.fname = "Name of the .hdf5 file wiith the catalogue classification"

if __name__ == "__main__":
    sap.run(check)
