# coding: utf-8

import h5py
import numpy as np
import geopandas as gpd
import logging

from openquake.mbt.tools.tr.catalogue import get_catalogue
from openquake.mbt.tools.geo import get_idx_points_inside_polygon
from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.mbt.tools.tr.tectonic_regionalisation import (
    set_crustal, get_crust_model)


class SetCrustalEarthquakes():
    """
    :param crust_filename:
    :param catalogue_filename:
    :param treg_filename:
    :param label:
    """

    def __init__(self, crust_filename, catalogue_fname, treg_filename,
                 distance_delta, label, lower_depth=400, shapefile=None,
                 log_fname=None):
        self.crust_filename = crust_filename
        self.catalogue_fname = catalogue_fname
        self.treg_filename = treg_filename
        self.delta = distance_delta
        self.label = label
        self.lower_depth = lower_depth
        self.shapefile = shapefile
        self.log_fname = log_fname

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
        # open log file and prepare the group
        flog = h5py.File(self.log_fname, 'a')
        if self.label not in flog.keys():
            grp = flog.create_group('/{:s}'.format(self.label))
        else:
            grp = flog['/{:s}'.format(self.label)]
        #
        # load the crust model
        crust, sidx = get_crust_model(self.crust_filename)
        #
        # classify earthquakes
        treg, data = set_crustal(icat, crust, sidx,
                                 self.delta, self.lower_depth)
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
            # prepare array where to store the classification
            isel = np.full((len(icat.data['longitude'])), False, dtype=bool)
            #
            # read polygon using geopandas - get a geodataframe
            gdf = gpd.read_file(self.shapefile)
            #
            # process the geometry i.e. finds points inside
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

        tl = np.zeros(len(treg),
                      dtype={'names': ('eid', 'lon', 'lat', 'dep', 'moh', 'idx'),
                             'formats': ('S15', 'f8', 'f8', 'f8', 'f8', 'i4')})

        tl['eid'] = icat.data['eventID']
        tl['lon'] = icat.data['longitude']
        tl['lat'] = icat.data['latitude']
        tl['dep'] = icat.data['depth']
        tl['moh'] = np.array(data)[:, 1]
        tl['idx'] = treg
        #
        # store log data
        grp.create_dataset('data', data=np.array(tl))
        #
        # storing results in the .hdf5 file
        f = h5py.File(self.treg_filename, "a")
        if len(remove_from):
            fmt = '    treg: {:d}'
            logging.info(fmt.format(len(treg)))
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
        #
        #
        f.close()
        flog.close()
