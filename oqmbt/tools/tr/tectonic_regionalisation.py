"""
:mod:`oqmbt.tools.tr.tectonic_regionalisation`
"""
import re
import numpy as np

from rtree import index


def _generator_function(data):
    for i, dat in enumerate(data):
        yield (i, (dat[0], dat[1], dat[0], dat[1]), None)


def get_crust_model(filename):
    """
    :parameter filename:
    """
    crust = []
    for line in open(filename, 'r'):
        aa = re.split('\s+', re.sub('^\s*', '', line.rstrip()))
        crust.append([float(aa[0]), float(aa[1]), float(aa[2])])
    #
    # create the spatial index
    sidx = index.Index(_generator_function(crust))
    #
    out = np.array(crust)
    out[:, 2] = -1*out[:, 2]
    return out, sidx


def set_crustal(cat, treg, crust, sidx, delta=0):
    """
    :parameter catalogue:
    :parameter treg:
    :parameter crust:
    :parameter sidx:
    """
    data = []
    treg = np.full((len(cat.data['longitude'])), False, dtype=bool)
    for idx, (lon, lat, dep) in enumerate(zip(cat.data['longitude'],
                                              cat.data['latitude'],
                                              cat.data['depth'])):
        #
        # Find the nearest cell
        iii = list(sidx.nearest((lon, lat, lon, lat), 1))
        #
        #
        if crust[iii[0], 2]+float(delta) > dep:
            treg[idx] = True
        data.append([dep, crust[iii[0], 2]])
    return treg, data


def set_firm_depth(cat, treg, firm_depth):
    """
    :parameter catalogue:
    :parameter treg:
    :parameter firm_depth:
    """
    data = []
    treg = np.full((len(cat.data['longitude'])), False, dtype=bool)
    for idx, (lon, lat, dep) in enumerate(zip(cat.data['longitude'],
                                              cat.data['latitude'],
                                              cat.data['depth'])):
        #
        # Find the nearest cell
        if firm_depth > dep:
            treg[idx] = True
        data.append([dep, firm_depth])
    return treg, data


