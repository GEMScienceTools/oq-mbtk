""" :mod:`openquake.man.single.point` module. This module contains functions
for computing general characteristics of gridded seismicity sources.
Assumptions: (1) the nodes of the grid represent centroids of cells (2) the
grid nodes have a constant spacing (either in distance or long/lat). The
spacing can be different along longitude and along latitude """

import logging
import numpy as np

from rtree import index

from openquake.man.mfd import get_rates_within_m_range
from openquake.hazardlib.geo.geodetic import azimuth
from openquake.hazardlib.geo.geodetic import geodetic_distance


def generator_function(data):
    for i in range(0, data.shape[0]):
        yield (i, (data[i, 0], data[i, 1], data[i, 0], data[i, 1]))


def get_spatial_index(points):
    """
    """
    #
    # get point coordinates
    coo = []
    for i, p in enumerate(points):
        lo = p.location.longitude
        la = p.location.latitude
        de = p.location.depth
        coo.append((lo, la, de))
    #
    # checking if the model crosses the IDL
    cooa = np.array(coo)
    if any(cooa[:, 0] > 179) and any(cooa[:, 0] < -179):
        logging.info('The model crosses the IDL. Fixing coordinates')
        i1 = np.count_nonzero(cooa[:, 0] < 180.)
        i2 = np.count_nonzero(cooa[:, 0] > 0.)
        if i1 > i2:
            #
            # there are more points west of the IDL therefore we convert the
            # ones east of the IDL
            idx = np.nonzero((cooa[:, 0] < 90) & (cooa[:, 0] > -180.))
            cooa[idx, 0] = 360.+cooa[idx, 0]
    #
    # creating the spatial index
    # sidx = index.Index(generator_function(cooa))
    sidx = index.Index()
    for i in range(0, cooa.shape[0]):
        sidx.insert(i, (cooa[i, 0], cooa[i, 1], cooa[i, 0], cooa[i, 1]))
    #
    # done!
    return sidx, cooa


def get_rates_density(model, mmint=-11.0, mmaxt=11.0, trt=set()):
    """
    This function computes the rates for each point source included in the
    model (i.e. a list of :class:`openquake.hazardlib.source` instances.

    :parameter model:
        A list of openquake source point instances
    :parameter mmint:
        Minimum magnitude
    :parameter mmaxt:
        Minimum magnitude
    :parameter trt:
        A set of tectonic region keywords
    :returns:
        A (key, value) dictionary, where key is the source ID and value
        corresponds to density of the rate of occurrence [eqks/(yr*km2)]
    """
    dens = []
    #
    # compute the area of each cell of the grid. This is done for all the cells
    # without considering their TR
    area, coloc, coo, sidx = get_cell_areas(model)
    #
    # checking results
    assert len(area) == len(model)
    #
    #
    areas = []
    cidx = []
    coo = []
    #
    #
    for cnt, src in enumerate(model):
        #
        # passes if the set is empty or the TR of the source matches the ones
        # defined
        if not trt == 0 or set(src.tectonic_region_type) & trt:
            #
            # rates for the point source
            trates = get_rates_within_m_range(src.mfd, mmint, mmaxt)
            #
            # find the nearest node
            i = list(sidx.nearest((src.location.longitude,
                                   src.location.latitude,
                                   src.location.longitude,
                                   src.location.latitude), 1))
            #
            # compute the density
            cidx.append(cnt)
            if not np.isnan(area[i[0]]):
                for tple in src.hypocenter_distribution.data:
                    dens.append(trates / area[i[0]] * tple[0])
                    areas.append(area[i[0]])
                    coo.append((src.location.longitude,
                                src.location.latitude,
                                tple[1]))
            else:
                dens.append(0)
                areas.append(0)
                coo.append((src.location.longitude,
                            src.location.latitude,
                            0.0))
    #
    #
    return dens, areas, cidx, coo


def get_cell_areas(points):
    """
    Computes the area of each point source included in the `point` list.

    :parameter points:
        A list of openquake source point instances
    :parameter sidx:
        An rtree spatial index

    TODO: we need to add a test for the international date line since this is
        currently not supported. This might be simply solved by replacing
        the geographic coordinates in the spatial index with projected
        coordinates
    """
    dlt_x = 0.3
    dlt_z = 2.
    #
    # create spatial index
    sidx, coo = get_spatial_index(points)
    #
    # compute the area of each grid cell
    areas = []
    coloc = []
    for i, (lop, lap, dep) in enumerate(list(coo)):
        #
        # get the nearest neighbours
        # import pdb; pdb.set_trace()
        nnidx = list(sidx.intersection((lop-dlt_x, lap-dlt_x,
                                        lop+dlt_x, lap+dlt_x)))
        #
        # filter out points at depths other than the one of the point
        fnnidx = []
        for j in nnidx:
            if abs(dep - coo[j, 2]) < dlt_z:
                fnnidx.append(j)
        #
        # get the area
        area, clc = _get_cell_area(lop, lap, coo, fnnidx)
        areas.append(area)
        coloc.append(clc)
    #
    #
    return areas, coloc, coo, sidx


def _get_cell_area(rlo, rla, coo, nnidx):
    """
    :parameter rlo:
    :parameter rla:
    :parameter coo:
    :parameter nnidx:

    :return:

    """
    alo = [coo[idx][0] for idx in nnidx]
    ala = [coo[idx][1] for idx in nnidx]
    #
    # Computing azimuths and distances
    azis = azimuth(rlo, rla, alo, ala)
    dsts = geodetic_distance(rlo, rla, alo, ala)
    #
    # Processing the selected nodes
    delta = 5.0
    colocated = 0
    nearest_nodes = {}
    for azi, dst, idx in zip(azis, dsts, nnidx):
        if dst < 0.5:
            if (abs(rlo - coo[idx][0]) < 0.005 and
                    abs(rla - coo[idx][1]) < 0.005):
                colocated += 1
                continue
        # East
        if abs(azi-90) < delta:
            if 90 in nearest_nodes:
                if dst < nearest_nodes[90][0]:
                    nearest_nodes[90] = (dst, idx)
            else:
                nearest_nodes[90] = (dst, idx)
        # South
        elif abs(azi-180) < delta:
            if 180 in nearest_nodes:
                if dst < nearest_nodes[180][0]:
                    nearest_nodes[180] = (dst, idx)
            else:
                nearest_nodes[180] = (dst, idx)
        # West
        elif abs(azi-270) < delta:
            if 270 in nearest_nodes:
                if dst < nearest_nodes[270][0]:
                    nearest_nodes[270] = (dst, idx)
            else:
                nearest_nodes[270] = (dst, idx)
        # North
        elif abs(azi-360) < delta or azi < delta:
            if 0 in nearest_nodes:
                if dst < nearest_nodes[0][0]:
                    nearest_nodes[0] = (dst, idx)
            else:
                nearest_nodes[0] = (dst, idx)
        else:
            pass
    #
    # fix missing information
    out = np.nan
    try:
        fdsts = _get_final_dsts(nearest_nodes)
        out = (fdsts[0]+fdsts[2])/2*(fdsts[1]+fdsts[3])/2
    except:
        pass
        logging.debug('Node:', rlo, rla)
        logging.debug('Nearest nodes:', nearest_nodes)
        logging.debug('Queried nodes:')
        for idx in nnidx:
            logging.debug('  ', coo[idx][0], coo[idx][1], coo[idx][2])
    return out, colocated


def _get_final_dsts(nno):
    """
    :parameter nno:
        A dictionary where keys are angles (0, 90, 180 and 270) and values are
        tuples containing a distance and one index
    """
    #
    # array containing the final values of distance along the 4 main directions
    fd = []
    #
    # north
    if 0 in nno:
        fd.append(nno[0][0])
    elif 180 in nno:
        fd.append(nno[180][0])
    else:
        raise ValueError('Cannot define distance toward north')
    #
    # east
    if 90 in nno:
        fd.append(nno[90][0])
    elif 270 in nno:
        fd.append(nno[270][0])
    else:
        raise ValueError('Cannot define distance toward east')
    #
    # south
    if 180 in nno:
        fd.append(nno[180][0])
    elif 0 in nno:
        fd.append(nno[0][0])
    else:
        raise ValueError('Cannot define distance toward south')
    #
    # west
    if 270 in nno:
        fd.append(nno[270][0])
    elif 90 in nno:
        fd.append(nno[90][0])
    else:
        raise ValueError('Cannot define distance toward west')
    #
    # final result
    return fd
