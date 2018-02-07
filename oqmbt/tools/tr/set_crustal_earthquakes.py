# coding: utf-8

import h5py
import numpy as np

from oqmbt.tools.tr.catalogue import get_catalogue
from oqmbt.tools.tr.tectonic_regionalisation import (set_crustal,
                                                     get_crust_model)
from openquake.hmtk.seismicity.selector import CatalogueSelector

class SetCrustalEarthquakes():
    """
    :param crust_filename:
    :param catalogue_filename:
    :param treg_filename:
    :param label:
    """

    def __init__(self, crust_filename, catalogue_fname, treg_filename,
                 distance_delta,label='crustal'):
        self.crust_filename = crust_filename
        self.catalogue_fname = catalogue_fname
        self.treg_filename = treg_filename
        self.label = label
        self.delta = distance_delta

    def classify(self, remove_from):
        """
        :param str remove_from:
        """
        #
        # get catalogue
        cat = get_catalogue(self.catalogue_fname)
        #
        # prepare dictionary with classification
        treg = {}
        treg['crustal'] = np.full((len(cat.data['longitude'])), False,
                                  dtype=bool)
        #
        # load the crust model
        crust, sidx = get_crust_model(self.crust_filename)
        #
        # classify earthquakes
        treg, data = set_crustal(cat, treg, crust, sidx, self.delta)
        #
        # storing results in the .hdf5 file
        f = h5py.File(self.treg_filename, "a")
        if len(remove_from):
            iii = np.nonzero(treg)
            for tkey in remove_from:
                print('    Cleaning {:s}'.format(tkey))
                old = f[tkey][:]
                del f[tkey]
                old[iii] = False
                f[tkey] = old
        if self.label in f.keys():
            del f[self.label]
        f[self.label] = treg
        f.close()
