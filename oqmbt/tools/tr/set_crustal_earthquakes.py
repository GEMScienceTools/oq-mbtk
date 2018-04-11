# coding: utf-8

import h5py
import numpy as np
import geopandas as gpd
import logging

from oqmbt.tools.tr.catalogue import get_catalogue
from oqmbt.tools.geo import get_idx_points_inside_polygon
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
                 distance_delta, label, shapefile=None):
        self.crust_filename = crust_filename
        self.catalogue_fname = catalogue_fname
        self.treg_filename = treg_filename
        self.delta = distance_delta
        self.label = label
        self.shapefile = shapefile


    def classify(self, remove_from):
        """
        :param str remove_from:
        """
        #
        # get catalogue
        icat = get_catalogue(self.catalogue_fname)
        #
        # prepare dictionary with classification
        treg = {}
        treg[self.label] = np.full((len(icat.data['longitude'])), False,
                                   dtype=bool)
        #
        # load the crust model
        crust, sidx = get_crust_model(self.crust_filename)
        #
        # classify earthquakes
        treg, data = set_crustal(icat, crust, sidx, self.delta)
        #
        # select eartquakes within the polygon
        if self.shapefile is not None:
            #
            # create an array with the coordinates of the earthquakes in the
            # catalogue
            cp = []
            idxs = []
            for i, (lo, la) in enumerate(zip(icat.data['longitude'],
                                            icat.data['latitude'])):
                cp.append([lo, la])
                idxs.append(i)
            cp = np.array(cp)
            #
            #
            isel = np.full((len(icat.data['longitude'])), False, dtype=bool)
            #
            # read polygon using geopandas - get a geodataframe
            gdf = gpd.read_file(self.shapefile)
            #
            #
            idx_all_sel = []
            for pol in gdf.geometry:
                pcoo = []
                for pt in list(pol.exterior.coords):
                    pcoo.append(pt)
                pcoo = np.array(pcoo)
                sel_idx = get_idx_points_inside_polygon(cp[:, 0], cp[:, 1],
                                                        pcoo[:, 0], pcoo[:, 1],
                                                        idxs)
                idx_all_sel += sel_idx
            #
            # final catalogue
            isel[idx_all_sel] = True
            #
            # final TR
            treg = np.logical_and(treg, isel)
        #
        # storing results in the .hdf5 file
        f = h5py.File(self.treg_filename, "a")
        if len(remove_from):
            iii = np.nonzero(treg)
            for tkey in remove_from:
                logging.info('    Cleaning {:s}'.format(tkey))
                old = f[tkey][:]
                del f[tkey]
                old[iii] = False
                f[tkey] = old
        if self.label in f.keys():
            del f[self.label]
        f[self.label] = treg
        f.close()
