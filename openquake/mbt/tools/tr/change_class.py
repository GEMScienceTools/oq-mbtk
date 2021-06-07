#!/usr/bin/env python
import sys
import h5py
import pickle
import re

from openquake.baselib import sap


def change(cat_pickle_filename, treg, eqlist):
    """
    cat_pickle_filename - pickled catalogue
    treg - hdf5 file with classifications; currently
    a new file '<>_up.hdf5' is created rather than overwriting
    eqlist - csv file with two columns format <eventID>,<target label>
    """

    treg2 = treg.replace('.hdf5', '_up.hdf5')
    cat = pickle.load(open(cat_pickle_filename, 'rb'))
    dct = {key: idx for idx, key in enumerate(cat.data['eventID'])}

    idx = []
    targ = []
    for line in open(eqlist, 'r'):
        eqid, target = line.split(',')
        idx.append(dct[eqid])
        targ.append(target)

    f = h5py.File(treg, "r")
    f2 = h5py.File(treg2, "w")
    for key in f.keys():
        tmp = f[key][:]
        for eid, tar in zip(idx, targ):
            if re.search(key, tar):
                tmp[eid] = True
            else:
                tmp[eid] = False
        f2[key] = tmp
    f.close()
    f2.close()

change.cat_pickle_filename = 'pickled catalogue'
change.treg = 'TR hdf5 filename'
msg = 'list of events to change. format <eventID>,<target class>'
change.eqlist = msg


if __name__ == "__main__":
    sap.run(change)
