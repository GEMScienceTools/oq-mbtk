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
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:3857')
    xp, yp = transform(inProj, outProj, lon, lat)
    return xp, yp


def _get_source_model(source_file, investigation_time=1., 
                      rupture_mesh_spacing=10.0,
                      complex_fault_mesh_spacing=10.0, 
                      width_of_mfd_bin=0.1,
                      area_source_discretization=20.,
                      **kwargs):
    """
    Read and build a source model from an xml file

    :parameter source_file
        The name of a file containing the source model
    :parameter float inv_time:
        A positive float
    :paramater float simple_mesh_spacing:
        A positive float
    :parameter float complex_mesh_spacing:
        A positive float
    :parameter float mfd_spacing:
        A positive float
    :parameter float area_discretisation:
        A positive float
    :returns:
        A list of :class:`~openquake.hazardlib.sourceconverter.SourceGroup`
        instances
    """
    conv = SourceConverter(investigation_time, rupture_mesh_spacing, 
                           complex_fault_mesh_spacing,
                           width_of_mfd_bin, area_source_discretization,
                           **kwargs)
    srcs = to_python(source_file, conv)
    return srcs.src_groups


def read(model_filename, get_info=True, **kwargs):
    """
    This reads the nrml file containing the model

    :parameter model_filename:
        The name (including path) to a nrml formatted earthquake source model
    :return:
        A list of sources and information on the model
    """
    # Analysis
    logging.info('Reading: %s' % (model_filename))
    source_grps = _get_source_model(model_filename, **kwargs)
    source_model = []
    for grp in source_grps:
        for src in grp.sources:
            source_model.append(src)
    logging.info('The model contains %d sources' % (len(source_model)))
    #
    # get info
    info = None
    if get_info:
        info = _get_model_info(source_model)
    return source_model, info


def _get_mmin_mmax_nonpar(src):
    mmin = 1e10
    mmax = -1e10
    for d in src.data:
        mmin = min(mmin, d[0].mag)
        mmax = max(mmax, d[0].mag)
    return mmin, mmax


def _get_model_info(srcl):
    """
    :parameter srcl:
        A list of openquake source instances
    """
    trt_mmax = {}
    trt_mmin = {}
    trt_srcs = {}
    srcs_mmax = {}
    srcs_mmin = {}
    for idx, src in enumerate(srcl):
        trt = src.tectonic_region_type
        typ = type(src).__name__
        if typ == 'NonParametricSeismicSource':
            mmin, mmax = _get_mmin_mmax_nonpar(src)
        else:
            mmin, mmax = src.mfd.get_min_max_mag()
        # Mmax per tectonic region
        if trt in trt_mmax:
            trt_mmax[trt] = max(trt_mmax[trt], mmax)
        else:
            trt_mmax[trt] = mmax
        # Mmin per tectonic region
        if trt in trt_mmin:
            trt_mmin[trt] = min(trt_mmin[trt], mmin)
        else:
            trt_mmin[trt] = mmin
        # Mmax per source type
        if typ in srcs_mmax:
            srcs_mmax[typ] = max(srcs_mmax[typ], mmax)
        else:
            srcs_mmax[typ] = mmax
        # Mmin per source type
        if typ in srcs_mmin:
            srcs_mmin[typ] = min(srcs_mmin[typ], mmin)
        else:
            srcs_mmin[typ] = mmin
        # Src per TRT
        if trt in trt_srcs:
            trt_srcs[trt] = trt_srcs[trt] & set(typ)
        else:
            trt_srcs[trt] = set([typ])

    return {'trt_mmax': trt_mmax,
            'trt_mmin': trt_mmin,
            'trt_srcs': trt_srcs,
            'srcs_mmax': srcs_mmax,
            'srcs_mmin': srcs_mmin,
            }


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
            occ = [tmp * tple[0] for tmp in tsrc.mfd.occurrence_rates]
            tsrc.mfd.occurrence_rates = occ
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
