# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8


"""
Parser for moment tensor catalogue in GCMT format into a set of GCMT classes
"""

import re
import datetime
import numpy as np
from math import floor, fabs
from linecache import getlines
import openquake.cat.gcmt_utils as utils
from openquake.cat.gcmt_catalogue import (GCMTHypocentre, GCMTCentroid,
                                          GCMTPrincipalAxes, GCMTNodalPlanes,
                                          GCMTMomentTensor, GCMTEvent,
                                          GCMTCatalogue)


def _read_date_from_string(str1):
    """
    Reads the date from a string in the format YYYY/MM/DD and returns
    :class: datetime.date
    """
    full_date = list(map(int, str1.split('/')))
    return datetime.date(full_date[0], full_date[1], full_date[2])


def _read_time_from_string(str1):
    """
    Reads the time from a string in the format HH:MM:SS.S and returns
    :class: datetime.time
    """
    full_time = list(map(float, str1.split(':')))
    hour = int(full_time[0])
    minute = int(full_time[1])
    if full_time[2] > 59.99:
        minute += 1
        second = 0
    else:
        second = int(full_time[2])

    microseconds = int((full_time[2] - floor(full_time[2])) * 1000000)
    return datetime.time(hour, minute, second, microseconds)


def _read_moment_tensor_from_ndk_string(ndk_string, system='USE'):
    """
    Reads the moment tensor from the ndk_string representation
    ndk_string = [Mrr, sigMrr, Mtt, sigMtt, Mpp, sigMpp, Mrt, sigMrt, Mrp,
        sigMrp, Mtp, sigMtp]
    Output tensors should be of format:
        expected = [[Mtt, Mtp, Mtr],
                    [Mtp, Mpp, Mpr],
                    [Mtr, Mpr, Mrr]]
        sigma = [[sigMtt, sigMtp, sigMtr],
                 [sigMtp, sigMpp, sigMpr],
                 [sigMtr, sigMpr, sigMrr]]
    Exponent returned in Nm

    :param str ndk_string:
        String of data in ndk format (line 4 of event)
    :param str system:
        Reference frame of tensor Up, South, East {USE} or North, East, Down
        (NED)
    """
    exponent = float(ndk_string[0:2]) - 7.
    mkr = np.array([2, 9, 15], dtype=int)
    vector = []
    for i in range(0, 6):
        vector.extend([float(ndk_string[mkr[0]:mkr[1]]),
                       float(ndk_string[mkr[1]:mkr[2]])])
        mkr = mkr + 13
    vector = np.array(vector)
    mrr, mtt, mpp, mrt, mrp, mtp = tuple(vector[np.arange(0, 12, 2)])
    sig_mrr, sig_mtt, sig_mpp, sig_mrt, sig_mrp, sig_mtp = \
        tuple(vector[np.arange(1, 13, 2)])

    tensor = utils.COORD_SYSTEM[system](mrr, mtt, mpp, mrt, mrp, mtp)
    tensor = (10. ** exponent) * tensor

    sigma = utils.COORD_SYSTEM[system](sig_mrr, sig_mtt, sig_mpp,
                                       sig_mrt, sig_mrp, sig_mtp)
    sigma = (10. ** exponent) * sigma

    return tensor, sigma, exponent


class ParseNDKtoGCMT(object):
    """
    Implements the parser to read a file in ndk format to the GCMT catalogue
    """
    def __init__(self, filename):
        '''
        :param str filename:
            Name of the catalogue file in ndk format
        '''
        self.filename = filename

    def read_file(self, start_year=None, end_year=None):
        '''
        Reads the file
        '''
        raw_data = getlines(self.filename)
        num_lines = len(raw_data)
        if ((float(num_lines) / 5.) - float(num_lines // 5)) > 1E-9:
            raise IOError('GCMT represented by 5 lines - number in file not'
                          ' a multiple of 5!')
        number_gcmts = num_lines // 5
        # Pre-allocates list
        data_gcmts = [None for i in range(number_gcmts)]
        id0 = 0
        print('Parsing catalogue ...')
        for iloc in range(number_gcmts):
            data_gcmts[iloc] = self.read_ndk_event(raw_data, id0)
            id0 += 5
        print('complete. Contains %s moment tensors' % len(data_gcmts))
        if not start_year:
            start_year = data_gcmts[0].centroid.date.year

        if not end_year:
            end_year = data_gcmts[-1].centroid.date.year

        return GCMTCatalogue(start_year, end_year, data_gcmts)

    def read_ndk_event(self, raw_data, id0):
        """
        Reads a 5-line batch of data into a set of GCMTs
        """
        gcmt = GCMTEvent()
        # Get hypocentre
        ndkstring = raw_data[id0].rstrip('\n')
        gcmt.hypocentre = self._read_hypocentre_from_ndk_string(ndkstring)

        # GCMT metadata
        ndkstring = raw_data[id0 + 1].rstrip('\n')
        gcmt = self._get_metadata_from_ndk_string(gcmt, ndkstring)

        # Get Centroid
        ndkstring = raw_data[id0 + 2].rstrip('\n')
        gcmt.centroid = self._read_centroid_from_ndk_string(ndkstring,
                                                            gcmt.hypocentre)

        # Get Moment Tensor
        ndkstring = raw_data[id0 + 3].rstrip('\n')
        gcmt.moment_tensor = self._get_moment_tensor_from_ndk_string(ndkstring)

        # Get principal axes
        ndkstring = raw_data[id0 + 4].rstrip('\n')
        gcmt.principal_axes = self._get_principal_axes_from_ndk_string(
            ndkstring[3:48],
            exponent=gcmt.moment_tensor.exponent)

        # Get Nodal Planes
        gcmt.nodal_planes = self._get_nodal_planes_from_ndk_string(
            ndkstring[57:])

        # Get Moment and Magnitude
        gcmt.moment, gcmt.version, gcmt.magnitude = \
            self._get_moment_from_ndk_string(ndkstring,
                                             gcmt.moment_tensor.exponent)

        return gcmt

    def _read_hypocentre_from_ndk_string(self, linestring):
        """
        Reads the hypocentre data from the ndk string to return an
        instance of the GCMTHypocentre class
        """
        hypo = GCMTHypocentre()
        hypo.source = linestring[0:4]
        hypo.date = _read_date_from_string(linestring[5:15])
        hypo.time = _read_time_from_string(linestring[16:26])
        hypo.latitude = float(linestring[27:33])
        hypo.longitude = float(linestring[34:41])
        hypo.depth = float(linestring[42:47])
        magnitudes = list(map(float, (linestring[48:55]).split(' ')))
        if magnitudes[0] > 0.:
            hypo.m_b = magnitudes[0]
        if magnitudes[1] > 0.:
            hypo.m_s = magnitudes[1]
        hypo.location = linestring[56:].strip()
        return hypo

    def _get_metadata_from_ndk_string(self, gcmt, ndk_string):
        """
        Reads the GCMT metadata from line 2 of the ndk batch
        """
        gcmt.identifier = ndk_string[:16].strip()
        inversion_data = re.split('[A-Z:]+', ndk_string[17:61])
        gcmt.metadata['BODY'] = list(map(float, inversion_data[1].split()))
        gcmt.metadata['SURFACE'] = list(map(float, inversion_data[2].split()))
        gcmt.metadata['MANTLE'] = list(map(float, inversion_data[3].split()))
        further_meta = re.split('[: ]+', ndk_string[62:])
        gcmt.metadata['CMT'] = int(further_meta[1])
        gcmt.metadata['FUNCTION'] = {'TYPE': further_meta[2],
                                     'DURATION': float(further_meta[3])}
        return gcmt

    def _read_centroid_from_ndk_string(self, ndk_string, hypocentre):
        """
        Reads the centroid data from the ndk string to return an
        instance of the GCMTCentroid class
        :param str ndk_string:
            String of data (line 3 of ndk format)
        :param hypocentre:
            Instance of the GCMTHypocentre class
        """
        centroid = GCMTCentroid(hypocentre.date,
                                hypocentre.time)

        data = ndk_string[:58].split()
        centroid.centroid_type = data[0].rstrip(':')
        data = list(map(float, data[1:]))
        time_diff = data[0]
        if fabs(time_diff) > 1E-6:
            centroid._get_centroid_time(time_diff)
        centroid.time_error = data[1]
        centroid.latitude = data[2]
        centroid.latitude_error = data[3]
        centroid.longitude = data[4]
        centroid.longitude_error = data[5]
        centroid.depth = data[6]
        centroid.depth_error = data[7]
        centroid.depth_type = ndk_string[59:63]
        centroid.centroid_id = ndk_string[64:].strip()
        return centroid

    def _get_moment_tensor_from_ndk_string(self, ndk_string):
        """
        Reads the moment tensor from the ndk_string and returns an instance of
        the GCMTMomentTensor class.
        By default the ndk format uses the Up, South, East (USE) reference
        system.
        """
        moment_tensor = GCMTMomentTensor('USE')
        tensor_data = _read_moment_tensor_from_ndk_string(ndk_string, 'USE')
        moment_tensor.tensor = tensor_data[0]
        moment_tensor.tensor_sigma = tensor_data[1]
        moment_tensor.exponent = tensor_data[2]
        return moment_tensor

    def _get_principal_axes_from_ndk_string(self, ndk_string, exponent):
        """
        Gets the principal axes from the ndk string and returns an instance
        of the GCMTPrincipalAxes class
        """
        axes = GCMTPrincipalAxes()
        # The principal axes is defined in characters 3:48 of the 5th line
        exponent = 10. ** exponent
        axes.t_axis = {'eigenvalue': exponent * float(ndk_string[0:8]),
                       'plunge': float(ndk_string[8:11]),
                       'azimuth': float(ndk_string[11:15])}

        axes.b_axis = {'eigenvalue': exponent * float(ndk_string[15:23]),
                       'plunge': float(ndk_string[23:26]),
                       'azimuth': float(ndk_string[26:30])}

        axes.p_axis = {'eigenvalue': exponent * float(ndk_string[30:38]),
                       'plunge': float(ndk_string[38:41]),
                       'azimuth': float(ndk_string[41:])}
        return axes

    def _get_nodal_planes_from_ndk_string(self, ndk_string):
        '''
        Reads the nodal plane information (represented by 5th line [57:] of the
        tensor representation) and returns an instance of the GCMTNodalPlanes
        class
        '''
        planes = GCMTNodalPlanes()
        planes.nodal_plane_1 = {'strike': float(ndk_string[0:3]),
                                'dip': float(ndk_string[3:6]),
                                'rake': float(ndk_string[6:11])}
        planes.nodal_plane_2 = {'strike': float(ndk_string[11:15]),
                                'dip': float(ndk_string[15:18]),
                                'rake': float(ndk_string[18:])}
        return planes

    def _get_moment_from_ndk_string(self, ndk_string, exponent):
        """
        Gets the moment and the moment magnitude
        """
        moment = float(ndk_string[49:56]) * (10. ** exponent)
        version = ndk_string[:3]
        magnitude = utils.moment_magnitude_scalar(moment)
        return moment, version, magnitude
