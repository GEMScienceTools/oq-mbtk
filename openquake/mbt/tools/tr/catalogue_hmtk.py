"""
:mod:`openquake.mbt.tools.tr.catalogue_hmtk`
"""

import numpy as np

from rtree import index

from openquake.hazardlib.geo import Mesh

import multiprocessing as mp


def aaa(data):
    return data[1].get_min_distance(data[0])


def gdfs(catalogue, surface):
    """
    :parameter catalogue:
    :parameter surface:
    """
    nel = len(catalogue.data['longitude'])
    # MN: 'dsts' assigned but never used
    dsts = np.empty((nel))
    delta = 4000
    #
    # preparing the sub meshes, each containing a subset of earthquakes
    inputs = []
    for i in np.arange(0, nel, delta):
        upp = min([i+delta, nel-1])
        mesh = Mesh(catalogue.data['longitude'][i:upp],
                    catalogue.data['latitude'][i:upp],
                    catalogue.data['depth'][i:upp])
        inputs.append([mesh, surface])
    #
    # multiprocessing
    pool = mp.Pool(processes=6)
    # MN: 'results' assigned but never used
    results = pool.map(aaa, inputs)


def get_distances_from_surface(catalogue, surface):
    """
    This computes distances
    """
    nel = len(catalogue.data['longitude'])
    dsts = np.empty((nel))
    delta = 4000
    if nel < delta:
        mesh = Mesh(catalogue.data['longitude'][0:nel],
                    catalogue.data['latitude'][0:nel],
                    catalogue.data['depth'][0:nel])
        dsts = surface.get_min_distance(mesh)
    else:
        i = 0
        upp = 0
        while upp < nel-1:
            upp = min([i+delta, nel-1])
            mesh = Mesh(catalogue.data['longitude'][i:upp],
                        catalogue.data['latitude'][i:upp],
                        catalogue.data['depth'][i:upp])
            tmp = surface.get_min_distance(mesh)
            dsts[i:upp] = tmp
            i = upp + 1
    return dsts


def _generator(cat):
    """
    :parameter cat:
    """
    for i, (lon, lat, dep) in enumerate(zip(cat.data['longitude'],
                                            cat.data['latitude'],
                                            cat.data['depth'])):
        yield (i, (lon, lat, dep, lon, lat, dep), None)


def get_rtree_index(cat):
    """
    :parameter cat:
    """
    #
    # set index properties
    p = index.Property()
    p.dimension = 3
    #
    # creating the index
    sidx = index.Index(_generator(cat), properties=p)
    #
    return sidx
