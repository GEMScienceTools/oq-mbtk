import re
import os
import copy
import glob
import numpy
import pickle
import logging
from rtree import index

from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.mfd import TruncatedGRMFD, EvenlyDiscretizedMFD
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.geo.geodetic import distance, azimuth
from openquake.hazardlib.source import (AreaSource,
                                        SimpleFaultSource, ComplexFaultSource,
                                        CharacteristicFaultSource,
                                        NonParametricSeismicSource)

from pyproj import Proj, transform


def getcoo(lon, lat):
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3857')
    xp, yp = transform(inProj, outProj, lon, lat)
    return xp, yp


def storeNew(filename, model, info=None):
    """
    This creates a pickle file containing the list of earthquake sources
    representing an earthquake source model.

    :parameter filename:
        The name of the file where the to store the model
    :parameter model:
        A list of OpenQuake hazardlib source instances
    """
    # Preparing output filenames
    dname = os.path.dirname(filename)
    slist = re.split('\\.', os.path.basename(filename))
    # SIDx
    p = index.Property()
    p.dimension = 3
    sidx = index.Rtree(os.path.join(dname, slist[0]), properties=p)
    #
    cpnt = 0
    l_other = []
    l_points = []
    for src in model:
        if isinstance(src, (AreaSource, SimpleFaultSource, ComplexFaultSource,
                      CharacteristicFaultSource, NonParametricSeismicSource)):
            l_other.append(src)
        else:

            if len(src.hypocenter_distribution.data) == 1:
                srcs = [src]
            else:
                srcs = _split_point_source(src)

            x, y = getcoo(srcs[0].location.longitude,
                          srcs[0].location.latitude)

            for src in srcs:
                l_points.append(src)
                # calculate distances
                z = src.hypocenter_distribution.data[0][1]
                sidx.insert(cpnt, (x, y, z, x, y, z))
                cpnt += 1

    # All the other sources
    fou = open(filename, 'wb')
    pickle.dump(l_other, fou)
    fou.close()
    # Load info
    if info is None:
        info = _get_model_info(model)
    # Points
    fou = open(os.path.join(dname, slist[0]) + '_points.pkl', 'wb')
    pickle.dump(l_points, fou)
    fou.close()
    # Info
    fou = open(os.path.join(dname, slist[0]) + '_info.pkl', 'wb')
    pickle.dump(info, fou)
    fou.close()


def store(filename, model, info=None):
    """
    This creates a pickle file containing the list of earthquake sources
    representing an earthquake source model.

    :parameter filename:
        The name of the file where the to store the model
    :parameter model:
        A list of OpenQuake hazardlib source instances
    """
    # Preparing output filenames
    dname = os.path.dirname(filename)
    slist = re.split('\\.', os.path.basename(filename))
    # SIDx
    p = index.Property()
    p.dimension = 3
    sidx = index.Rtree(os.path.join(dname, slist[0]), properties=p)
    #
    cpnt = 0
    l_other = []
    l_points = []
    for src in model:
        if isinstance(src, (AreaSource, SimpleFaultSource, ComplexFaultSource,
                      CharacteristicFaultSource, NonParametricSeismicSource)):
            l_other.append(src)
        else:

            if len(src.hypocenter_distribution.data) == 1:
                srcs = [src]
            else:
                srcs = _split_point_source(src)

            for src in srcs:
                l_points.append(src)
                # calculate distances
                dst = distance(l_points[0].location.longitude,
                               l_points[0].location.latitude,
                               0.,
                               src.location.longitude,
                               src.location.latitude,
                               0.)
                azi = azimuth(l_points[0].location.longitude,
                              l_points[0].location.latitude,
                              src.location.longitude,
                              src.location.latitude)
                x = numpy.cos(numpy.radians(azi)) * dst
                y = numpy.sin(numpy.radians(azi)) * dst
                # update the spatial index
                z = src.hypocenter_distribution.data[0][1]
                sidx.insert(cpnt, (x, y, z, x, y, z))
                cpnt += 1

    # All the other sources
    fou = open(filename, 'wb')
    pickle.dump(l_other, fou)
    fou.close()
    # Load info
    if info is None:
        info = _get_model_info(model)
    # Points
    fou = open(os.path.join(dname, slist[0]) + '_points.pkl', 'wb')
    pickle.dump(l_points, fou)
    fou.close()
    # Info
    fou = open(os.path.join(dname, slist[0]) + '_info.pkl', 'wb')
    pickle.dump(info, fou)
    fou.close()


def _split_point_source(src):
    """
    Split a point source with multiple hypocentral depths into many point
    sources with a single hypo depth. Note that for the time being a
    evenly discretised MFD is required.
    """
    srcs = []
    for idx, tple in enumerate(src.hypocenter_distribution.data):
        tsrc = copy.deepcopy(src)
        tsrc.source_id = tsrc.source_id + "_{0:d}".format(idx)
        if isinstance(tsrc.mfd, TruncatedGRMFD):
            tsrc.mfd.a_val = numpy.log10(10.**tsrc.mfd.a_val * tple[0])
        elif isinstance(tsrc.mfd, EvenlyDiscretizedMFD):
            print(tple[0])
            occ = [tmp * tple[0] for tmp in tsrc.mfd.occurrence_rates]
            print(occ)
            tsrc.mfd.occurrence_rates = occ
            print(tsrc.mfd.occurrence_rates)
            tsrc.hypocenter_distribution.data = PMF([(1.0, tple[1])])
        srcs.append(tsrc)
    return srcs


def load(filename, what='all'):
    """
    This loads a pickle file containing the list of earthquake sources
    representing an earthquake source model.

    :parameter filename:
        The name of the file where the to store the model
    :parameter what:
        Can be 'other', 'all', 'point'
    :returns:
        A list of source instances
    """
    other = None
    points = None
    # Filename
    dname = os.path.dirname(filename)
    slist = re.split('\\.', os.path.basename(filename))
    # SIDx
    p = index.Property()
    p.dimension = 3
    sidx = index.Rtree(os.path.join(dname, slist[0]), properties=p)
    # Other
    if re.search('other', what) or re.search('all', what):
        fou = open(filename, 'rb')
        other = pickle.load(fou)
        fou.close()
    # Point
    if re.search('points', what) or re.search('all', what):
        fou = open(os.path.join(dname, slist[0]) + '_points.pkl', 'rb')
        points = pickle.load(fou)
        fou.close()
    # Info
    fou = open(os.path.join(dname, slist[0]) + '_info.pkl', 'rb')
    info = pickle.load(fou)
    fou.close()
    return other, points, info, sidx


def load_models(path, modell=None):
    """
    This loads a set of pickled models

    :parameter path:
        The name of the folder containing the .pkl files
    :returns:
        A dictionary with models and corresponding info
    """
    modd = {}
    for fname in glob.glob(path):
        slist = re.split('\\.', os.path.basename(fname))
        if not re.search('info', fname):
            if modell is not None and slist[0] in modell:
                mod, info = load(fname)
                modd[slist[0]] = {'model': mod, 'info': info}
            else:
                mod, info = load(fname)
                modd[slist[0]] = {'model': mod, 'info': info}
    return modd
