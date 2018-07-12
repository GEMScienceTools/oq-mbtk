"""
"""

import os
import re
import glob
import numpy as np
from pyproj import Geod

from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.geodetic import (distance, azimuth,
                                              npoints_towards)

TOLERANCE = 0.2


def profiles_depth_alignment(pro1, pro2):
    """
    :param pro1:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :param pro2:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :returns:
        AA
    """
    coo1 = [(pnt.longitude, pnt.latitude, pnt.depth) for pnt in pro1.points]
    coo2 = [(pnt.longitude, pnt.latitude, pnt.depth) for pnt in pro2.points]
    #
    #
    coo1 = np.array(coo1)
    coo2 = np.array(coo2)
    #
    #
    swap = 1
    if coo2.shape[0] < coo1.shape[0]:
        tmp = coo1
        coo1 = coo2
        coo2 = tmp
        swap = -1
    #
    # Creating two arrays of the same lenght
    coo1 = np.array(coo1)
    coo2 = np.array(coo2[:coo1.shape[0]])
    #
    # The two profiles require at least 5 points
    if len(coo1) > 5 and len(coo2) > 5:
        indexes = np.arange(-2, 3)
        dff = np.zeros_like(indexes)
        for i, shf in enumerate(indexes):
            if shf < 0:
                dff[i] = np.mean(abs(coo1[:shf, 2] - coo2[-shf:, 2]))
            elif shf == 0:
                dff[i] = np.mean(abs(coo1[:, 2] - coo2[:, 2]))
            else:
                dff[i] = np.mean(abs(coo1[shf:, 2] - coo2[:-shf, 2]))
        amin = np.amin(dff)
        res = indexes[np.amax(np.nonzero(dff == amin))] * swap
    else:
        res = 0

    return res


def _read_profiles(path, prefix='cs'):
    """
    :param path:
        The path to a folder containing a set of profiles
    """
    path = os.path.join(path, '{:s}*.*'.format(prefix))
    profiles = []
    names = []
    print(path)
    for filename in sorted(glob.glob(path)):
        profiles.append(_read_profile(filename))
        names.append(os.path.basename(filename))
    return profiles, names


def _read_profile(filename):
    """
    :parameter filename:
        The name of the folder file (usually with prefix 'cs_')
        specifing the geometry of the top of the slab

    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    points = []
    for line in open(filename, 'r'):
        aa = re.split('\s+', line)
        points.append(Point(float(aa[0]),
                            float(aa[1]),
                            float(aa[2])))
    return Line(points)


def _resample_profile(line, sampling_dist):
    """
    :parameter line:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :parameter sampling_dist:
        A scalar definining the distance used to sample the profile
    :returns:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    """
    lo = [pnt.longitude for pnt in line.points]
    la = [pnt.latitude for pnt in line.points]
    de = [pnt.depth for pnt in line.points]
    #
    # Add a tolerance length
    g = Geod(ellps='WGS84')
    az12, az21, odist = g.inv(lo[-2], la[-2], lo[-1], la[-1])
    odist /= 1e3
    slope = np.arctan((de[-1] - de[-2]) / odist)
    hdist = TOLERANCE * sampling_dist * np.cos(slope)
    vdist = TOLERANCE * sampling_dist * np.sin(slope)
    endlon, endlat, backaz = g.fwd(lo[-1], la[-1], az12, hdist*1e3)
    lo[-1] = endlon
    la[-1] = endlat
    de[-1] = de[-1] + vdist
    az12, az21, odist = g.inv(lo[-2], la[-2], lo[-1], la[-1])
    #
    # checking
    odist /= 1e3
    slopec = np.arctan((de[-1] - de[-2]) / odist)
    assert abs(slope-slopec) < 1e-3
    #
    # initialise the cumulated distance
    cdist = 0.
    #
    # get the azimuth of the profile
    azim = azimuth(lo[0], la[0], lo[-1], la[-1])
    #
    # initialise the list with the resampled nodes
    idx = 0
    resampled_cs = [Point(lo[idx], la[idx], de[idx])]
    #
    # set the starting point
    slo = lo[idx]
    sla = la[idx]
    sde = de[idx]
    #
    # resampling
    while 1:
        #
        # check loop exit condition
        if idx > len(lo)-2:
            break
        #
        # compute the distance between the starting point and the next point
        # on the profile
        segment_len = distance(slo, sla, sde, lo[idx+1], la[idx+1], de[idx+1])
        #
        # search for the point
        if cdist+segment_len > sampling_dist:
            #
            # this is the lenght of the last segment-fraction needed to
            # obtain the sampling distance
            delta = sampling_dist - cdist
            #
            # compute the slope of the last segment and its horizontal length.
            # We need to manage the case of a vertical segment TODO
            segment_hlen = distance(slo, sla, 0., lo[idx+1], la[idx+1], 0.)
            segment_slope = np.arctan((de[idx+1] - sde) / segment_hlen)
            #
            # horizontal and vertical lenght of delta
            delta_v = delta * np.sin(segment_slope)
            delta_h = delta * np.cos(segment_slope)
            #
            # add a new point to the cross section
            pnts = npoints_towards(slo, sla, sde, azim, delta_h, delta_v, 2)
            #
            # update the starting point
            slo = pnts[0][-1]
            sla = pnts[1][-1]
            sde = pnts[2][-1]
            resampled_cs.append(Point(slo, sla, sde))
            #
            # reset the cumulative distance
            cdist = 0.

        else:
            cdist += segment_len
            idx += 1
            slo = lo[idx]
            sla = la[idx]
            sde = de[idx]
    #
    # check the distances along the profile
    coo = [[pnt.longitude, pnt.latitude, pnt.depth] for pnt in resampled_cs]
    coo = np.array(coo)
    for i in range(0, coo.shape[0]-1):
        dst = distance(coo[i, 0], coo[i, 1], coo[i, 2],
                       coo[i+1, 0], coo[i+1, 1], coo[i+1, 2])
        if abs(dst-sampling_dist) > 0.1*sampling_dist:
            raise ValueError('Wrong distance between points along the profile')
    #
    #
    return Line(resampled_cs)
