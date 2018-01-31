# coding: utf-8

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.interpolate import griddata
from oqmbt.tools.tr.catalogue import get_catalogue
from oqmbt.tools.tr.catalogue_hmtk import (get_rtree_index,
                                           get_distances_from_surface)
from oq.hmtk.subduction.utils import (_read_edges,
                                      build_complex_surface_from_edges,
                                      plot_complex_surface)
from openquake.hmtk.seismicity.selector import CatalogueSelector

#
# selection buffer
DELTA = 0.3
LOWER_DEPTH_FOR_EARTHQUAKES = 400


class SetSubductionEarthquakes:
    """
    :param label:
    :param treg_filename:
    :param distance_folder:
    :param edges_folder:
    :param distance_buffer_below:
    :param distance_buffer_above:
    :param catalogue_filename:
    :param cat_pickle_filename:
    """

    def __init__(self, label, treg_filename, distance_folder,
                 edges_folder, distance_buffer_below,
                 distance_buffer_above, lower_depth,
                 catalogue_filename):

        self.label = label
        self.treg_filename = treg_filename
        self.distance_folder = distance_folder
        self.edges_folder = edges_folder
        self.distance_buffer_below = distance_buffer_below
        self.distance_buffer_above = distance_buffer_above
        self.catalogue_filename = catalogue_filename
        self.lower_depth = lower_depth

    def classify(self, compute_distances, remove_from):
        """
        :param compute_distances:
        :param list remove_from:
            A list of labels identifying TR from where the earthquakes assigned
            to this TR must be removed
        """
        #
        # set parameters
        treg_filename = self.treg_filename
        distance_folder = self.distance_folder
        edges_folder = self.edges_folder
        distance_buffer_below = self.distance_buffer_below
        distance_buffer_above = self.distance_buffer_above
        catalogue_filename = self.catalogue_filename
        lower_depth = self.lower_depth
        #
        # read the catalogue
        catalogue = get_catalogue(catalogue_filename)
        treg = np.full((len(catalogue.data['longitude'])), False, dtype=bool)
        #
        # create the spatial index
        sidx = get_rtree_index(catalogue)
        #
        # build the complex fault surface
        tedges = _read_edges(edges_folder)
        surface = build_complex_surface_from_edges(edges_folder)
        mesh = surface.get_mesh()
        #
        # set variables used in griddata
        data = np.array([mesh.lons.flatten().T, mesh.lats.flatten().T]).T
        values = mesh.depths.flatten().T
        #
        # set bounding box of the subduction surface
        min_lo_sub = np.amin(mesh.lons)
        min_la_sub = np.amin(mesh.lats)
        max_lo_sub = np.amax(mesh.lons)
        max_la_sub = np.amax(mesh.lats)
        #
        # select earthquakes within the bounding box
        idxs = sorted(list(sidx.intersection((min_lo_sub-DELTA,
                                              min_la_sub-DELTA, 0,
                                              max_lo_sub+DELTA,
                                              max_la_sub+DELTA, 500))))
        #
        # prepare array for the selection of the catalogue
        flags = np.full((len(catalogue.data['longitude'])), False, dtype=bool)
        flags[idxs] = True
        #
        # create a selector for the catalogue and select earthquakes within
        # bounding box
        sel = CatalogueSelector(catalogue, create_copy=True)
        cat = sel.select_catalogue(flags)
        cat.sort_catalogue_chronologically()
        self.cat = cat
        #
        # compute distances between the earthquakes in the catalogue and
        # the surface of the fault
        if compute_distances:
            #
            # calculate/load the distance to the subduction surface
            out_filename = os.path.join(distance_folder,
                                        'dist_{:s}.pkl'.format(self.label))
            if not os.path.exists(out_filename):
                surf_dist = get_distances_from_surface(cat, surface)
                pickle.dump(surf_dist, open(out_filename, 'wb'))
            else:
                surf_dist = pickle.load(open(out_filename, 'rb'))
                tmps = 'Loading distances from file: {:s}'
                print(tmps.format(out_filename))
                tmps = '    number of values loaded: {:d}'
                print(tmps.format(len(surf_dist)))
        #
        # info
        tmps = 'Number of eqks in the new catalogue     : {:d}'
        print(tmps.format(len(cat.data['longitude'])))
        #
        # calculate/load the distance to the subduction surface
        tmps = 'dist_{:s}.pkl'.format(self.label)
        out_filename = os.path.join(distance_folder, tmps)
        if not os.path.exists(out_filename):
            surf_dist = get_distances_from_surface(cat, surface)
            pickle.dump(surf_dist, open(out_filename, 'wb'))
        else:
            surf_dist = pickle.load(open(out_filename, 'rb'))
            print('Loading distances from file: {:s}'.format(out_filename))
        #
        # Calculate the depth of the top of the slab for every earthquake
        # location
        points = np.array([[lo, la] for lo, la in zip(cat.data['longitude'],
                                                      cat.data['latitude'])])
        #
        # compute the depth of the top of the slab at every epicenter
        sub_depths = griddata(data, values, (points[:, 0], points[:, 1]),
                              method='cubic')
        #
        # saving the distances to a file
        tmps = 'vert_dist_to_slab_{:s}.pkl'.format(self.label)
        out_filename = os.path.join(distance_folder, tmps)
        if not os.path.exists(out_filename):
            pickle.dump(surf_dist, open(out_filename, 'wb'))
        #
        # Let's find earthquakes close to the top of the slab
        # idxb = np.nonzero(np.isfinite(surf_dist) & np.isfinite(sub_depths))[0]
        """
        idxa = np.nonzero((np.isfinite(surf_dist) & np.isfinite(sub_depths)) &
                          ((surf_dist[idxb] < distance_buffer_below) &
                           (sub_depths[idxb] < cat.data['depth'][idxb])) |
                          ((surf_dist[idxb] < distance_buffer_above) &
                           (sub_depths[idxb] >= cat.data['depth'][idxb])))[0]
        """
        idxa = np.nonzero((np.isfinite(surf_dist) &
                           np.isfinite(sub_depths) &
                           np.isfinite(cat.data['depth'])) &
                          ((surf_dist < distance_buffer_below) &
                           (sub_depths < cat.data['depth'])) |
                          ((surf_dist < distance_buffer_above) &
                           (sub_depths >= cat.data['depth'])))[0]
        #
        #
        self.surf_dist = surf_dist
        self.sub_depths = sub_depths
        self.tedges = tedges
        self.idxa = idxa
        self.treg = treg
        #
        # updating the selection array
        for uuu, iii in enumerate(idxa):
            treg[idxs[iii]] = True
        #
        # storing results in the .hdf5 file
        print('Storing data in:\n', treg_filename)
        f = h5py.File(treg_filename, "a")
        if len(remove_from):
            print('    treg:', len(treg))
            iii = np.nonzero(treg)[0]
            for tkey in remove_from:
                print('    Cleaning {:s}'.format(tkey))
                old = f[tkey][:]
                print('     before:', len(np.nonzero(old)[0]))
                del f[tkey]
                old[iii] = False
                f[tkey] = old
                print('     after:', len(np.nonzero(old)[0]))
        if self.label in f.keys():
            del f[self.label]
        f[self.label] = treg
        f.close()

    def plotting_0(self):
        """
        """
        cat = self.cat
        sub_depths = self.sub_depths
        surf_dist = self.surf_dist

        fig = plt.figure(figsize=(10, 8))
        scat = plt.scatter(cat.data['depth'], sub_depths, c=surf_dist,
                           s=2**cat.data['magnitude'], edgecolor='w', vmin=0,
                           vmax=100)
        idx = np.nonzero(sub_depths < cat.data['depth'])
        plt.ylabel('Top of slab depth [km]')
        plt.xlabel('Hypocentral depth [km]')
        xx = np.arange(10, 300)
        plt.plot(xx, xx, ':r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([10, 300])
        plt.ylim([10, 200])
        plt.grid(axis='both', which='both')
        cb = plt.colorbar(scat, extend='both')
        cb.set_label('Shortest distance [km]')

    def plotting_1(self):
        """
        """
        cat = self.cat
        tedges = self.tedges
        idxa = self.idxa

        fig, ax = plot_complex_surface(tedges)
        ax.plot(cat.data['longitude'],
                cat.data['latitude'],
                cat.data['depth'], '.b', alpha=0.05)
        ax.plot(cat.data['longitude'][idxa],
                cat.data['latitude'][idxa],
                cat.data['depth'][idxa], '.c', alpha=0.2)
        plt.show()
