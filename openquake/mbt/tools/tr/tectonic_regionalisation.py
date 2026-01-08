"""
:mod:`openquake.mbt.tools.tr.tectonic_regionalisation`
"""
import re
import numpy as np
from decimal import Decimal, getcontext

from rtree import index

getcontext().prec = 6 

def _generator_function(data):
    for i, dat in enumerate(data):
        yield (i, (dat[0], dat[1], dat[0], dat[1]), None)


def get_crust_model(filename):
    """
    :parameter filename:
    """
    crust = []
    for line in open(filename, 'r'):
        aa = re.split('\\s+', re.sub('^\\s*', '', line.rstrip()))
        crust.append([float(aa[0]), float(aa[1]), float(aa[2])])
    #
    # create the spatial index
    sidx = index.Index(_generator_function(crust))
    #
    out = np.array(crust)
    out[:, 2] = -1*out[:, 2]
    return out, sidx


def set_crustal(cat, crust, sidx, delta=0, lower_depth=400):
    """
    :parameter catalogue:
        An instance of :class:`openquake.hmtk.seismicity.catalogue.Catalogue`
    :parameter crust:

    :parameter sidx:
    """
    data = []
    treg = np.full((len(cat.data['longitude'])), False, dtype=bool)
    for idx, (lon, lat, dep) in enumerate(zip(cat.data['longitude'],
                                              cat.data['latitude'],
                                              cat.data['depth'])):
        #
        # Find the nearest cell in the crustal model
        iii = list(sidx.nearest((lon, lat, lon, lat), 1))
        #
        # Set the crustal earthquakes
        if Decimal(crust[iii[0], 2]+float(delta)) > Decimal(dep) and (Decimal(dep) <= Decimal(lower_depth)):
            treg[idx] = True
        data.append([dep, crust[iii[0], 2]])
    return treg, data
