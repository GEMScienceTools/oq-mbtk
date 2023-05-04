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
Implements set of classes to represent a GCMT Catalogue
"""
from __future__ import print_function
import csv
import datetime
from math import fabs, floor, sqrt, pi
import numpy as np
import openquake.cat.gcmt_utils as utils
from collections import OrderedDict
# Adding on an exporter to Geojson, but only if geojson package exists
try:
    import geojson
except ImportError:
    print("geojson package not installed - export to geojson not available!")
    HAS_GEOJSON = False
else:
    HAS_GEOJSON = True


def cmp_mat(a, b):
    """
    Sorts two matrices returning a positive or zero value
    """
    c = 0
    for x, y in zip(a.flat, b.flat):
        c = cmp(abs(x), abs(y))
        if c != 0:
            return c
    return c


class GCMTHypocentre(object):
    """
    Simple representation of a hypocentre
    """
    def __init__(self):
        """
        """
        self.source = None
        self.date = None
        self.time = None
        self.longitude = None
        self.latitude = None
        self.depth = None
        self.m_b = None
        self.m_s = None
        self.location = None

    def __repr__(self):
        """
        String representation is bar separated list of attributes
        """
        return "|".join([
            str(getattr(self, val))
            for val in ["date", "time", "longitude", "latitude", "depth"]])


class GCMTCentroid(object):
    """
    Representation of a GCMT centroid
    """
    def __init__(self, reference_date, reference_time):
        """
        :param reference_date:
            Date of hypocentre as instance of :class: datetime.datetime.date
        :param reference_time:
            Time of hypocentre as instance of :class: datetime.datetime.time

        """
        self.centroid_type = None
        self.source = None
        self.time = reference_time
        self.time_error = None
        self.date = reference_date
        self.longitude = None
        self.longitude_error = None
        self.latitude = None
        self.latitude_error = None
        self.depth = None
        self.depth_error = None
        self.depth_type = None
        self.centroid_id = None

    def __repr__(self):
        """
        Returns a basic string representation
        """
        return "|".join([
            str(getattr(self, val))
            for val in ["date", "time", "longitude", "latitude", "depth"]])

    def _get_centroid_time(self, time_diff):
        """
        Generates the centroid time by applying the time difference to the
        hypocentre time
        """
        source_time = datetime.datetime.combine(self.date, self.time)
        second_diff = floor(fabs(time_diff))
        microsecond_diff = int(1.0E6 * (time_diff - second_diff))
        if time_diff < 0.:
            source_time = source_time - datetime.timedelta(
                seconds=int(second_diff), microseconds=microsecond_diff)
        else:
            source_time = source_time + datetime.timedelta(
                seconds=int(second_diff), microseconds=microsecond_diff)
        self.time = source_time.time()
        self.date = source_time.date()


class GCMTPrincipalAxes(object):
    """
    Class to represent the plunge and azimuth of T-, B- and P- plunge axes.
    Each axis is a dictionary containing the attributes: eigenvalue, azimuth
    and plunge. i.e.
    self.t_axis = {"eigenvalue": None, "azimuth": None, "plunge": }
    """
    def __init__(self):
        self.t_axis = None
        self.b_axis = None
        self.p_axis = None

    def get_moment_tensor_from_principal_axes(self):
        """
        Retrieves the moment tensor from the prinicpal axes
        """
        raise NotImplementedError('Moment tensor from principal axes not yet '
                                  'implemented!')

    def get_azimuthal_projection(self, height=1.0):
        """
        Returns the azimuthal projection of the tensor according to the
        method of Frohlich (2001)
        """
        raise NotImplementedError('Get azimuthal projection not yet '
                                  'implemented!')

    def __repr__(self):
        """
        """
        if self.t_axis:
            t_str = "T: L={:.4E}|Az={:.3f}|Pl={:.3f}".format(
                self.t_axis["eigenvalue"], self.t_axis["azimuth"],
                self.t_axis["plunge"])
        else:
            t_str = "T: None"
        if self.b_axis:
            b_str = "N: L={:.4E}|Az={:.3f}|Pl={:.3f}".format(
                self.b_axis["eigenvalue"], self.b_axis["azimuth"],
                self.b_axis["plunge"])
        else:
            b_str = "N: None"
        if self.p_axis:
            p_str = "P: L={:.4E}|Az={:.3f}|Pl={:.3f}".format(
                self.p_axis["eigenvalue"], self.p_axis["azimuth"],
                self.p_axis["plunge"])
        else:
            p_str = "P: None"
        return "{:s}|{:s}|{:s}".format(t_str, b_str, p_str)


class GCMTNodalPlanes(object):
    """
    Class to represent the two nodal planes, each as a dictionary containing
    the attributes: strike, dip and rake. i.e.
    self.nodal_plane_1 = {"strike":, "dip":, "rake":}
    """
    def __init__(self):
        """
        """
        self.nodal_plane_1 = None
        self.nodal_plane_2 = None

    def __repr__(self):
        """
        String rep is just strike/dip/rake e.g. 180/90/0
        """
        if self.nodal_plane_1:
            np1_str = "{:.0f}/{:.0f}/{:.0f}".format(
                self.nodal_plane_1["strike"],
                self.nodal_plane_1["dip"],
                self.nodal_plane_1["rake"])
        else:
            np1_str = "-/-/-"
        if self.nodal_plane_2:
            np2_str = "{:.0f}/{:.0f}/{:.0f}".format(
                self.nodal_plane_2["strike"],
                self.nodal_plane_2["dip"],
                self.nodal_plane_2["rake"])
        else:
            np2_str = "-/-/-"
        return "{:s} {:s}".format(np1_str, np2_str)


class GCMTMomentTensor(object):
    """
    Class to represent a moment tensor
    :param numpy.ndarray tensor:
        Moment tensor as 3 by 3 array
    :param numpy.ndarray tensor_sigma:
        Moment tensor uncertainty as 3 by 3 array
    :param float exponent:
        Exponent of the tensor
    :param str ref_frame:
        Reference frame of the tensor (USE or NED)
    """
    def __init__(self, reference_frame=None):
        self.tensor = None
        self.tensor_sigma = None
        self.exponent = None
        self.eigenvalues = None
        self.eigenvectors = None
        if reference_frame:
            self.ref_frame = reference_frame
        else:
            # Default to USE
            self.ref_frame = 'USE'

    def __repr__(self):
        """
        """
        if self.tensor is not None:
            return "[{:.3E} {:.3E} {:.3E}\n{:.3E} {:.3E} {:.3E}\n{:.3E} {:.3E} {:.3E}]".format(
                self.tensor[0, 0], self.tensor[0, 1], self.tensor[0, 2],
                self.tensor[1, 0], self.tensor[1, 1], self.tensor[1, 2],
                self.tensor[2, 0], self.tensor[2, 1], self.tensor[2, 2])
        else:
            return "[]"

    def normalise_tensor(self):
        """
        Normalise the tensor by dividing it by its norm, defined such that
        np.sqrt(X:X)
        """
        self.tensor, tensor_norm = utils.normalise_tensor(self.tensor)
        return self.tensor / tensor_norm, tensor_norm

    def _to_ned(self):
        """
        Switches the reference frame to NED
        """
        if self.ref_frame == 'USE':
            # Rotate
            return utils.use_to_ned(self.tensor), \
                   utils.use_to_ned(self.tensor_sigma)
        elif self.ref_frame == 'NED':
            # Already NED
            return self.tensor, self.tensor_sigma
        else:
            raise ValueError('Reference frame %s not recognised - cannot '
                             'transform to NED!' % self.ref_frame)

    def _to_use(self):
        '''
        Returns a tensor in the USE reference frame
        '''
        if self.ref_frame == 'NED':
            # Rotate
            return utils.ned_to_use(self.tensor), \
                   utils.ned_to_use(self.tensor_sigma)
        elif self.ref_frame == 'USE':
            # Already USE
            return self.tensor, self.tensor_sigma
        else:
            raise ValueError('Reference frame %s not recognised - cannot '
                             'transform to USE!' % self.ref_frame)

    def _to_6component(self):
        '''
        Returns the unique 6-components of the tensor in USE format
        [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        '''
        return utils.tensor_to_6component(self.tensor, self.ref_frame)

    def eigendecompose(self, normalise=False):
        '''
        Performs and eigendecomposition of the tensor and orders into
        descending eigenvalues
        '''
        self.eigenvalues, self.eigenvectors = utils.eigendecompose(self.tensor,
                                                                   normalise)
        return self.eigenvalues, self.eigenvectors

    def get_nodal_planes(self):
        '''
        Extracts the nodel planes from the tensor
        '''
        # Convert reference frame to NED
        self.tensor, self.tensor_sigma = self._to_ned()
        self.ref_frame = 'NED'
        # Eigenvalue decomposition
        # Tensor
        _, evect = utils.eigendecompose(self.tensor)
        # Rotation matrix
        _, rot_vec = utils.eigendecompose(np.matrix([[0., 0., -1],
                                                    [0., 0., 0.],
                                                    [-1., 0., 0.]]))
        rotation_matrix = (np.matrix(evect * rot_vec.T)).T
        if  np.linalg.det(rotation_matrix) < 0.:
            rotation_matrix *= -1.
        flip_dc = np.matrix([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
        rotation_matrices = sorted(
            [rotation_matrix, flip_dc * rotation_matrix], cmp=cmp_mat)
        nodal_planes = GCMTNodalPlanes()
        dip, strike, rake = [(180. / pi) * angle
            for angle in utils.matrix_to_euler(rotation_matrices[0])]
        # 1st Nodal Plane
        nodal_planes.nodal_plane_1 = {'strike': strike % 360,
                                      'dip': dip,
                                      'rake': -rake}

        # 2nd Nodal Plane
        dip, strike, rake = [(180. / pi) * angle
            for angle in utils.matrix_to_euler(rotation_matrices[1])]
        nodal_planes.nodal_plane_2 = {'strike': strike % 360.,
                                      'dip': dip,
                                      'rake': -rake}
        return nodal_planes


    def get_principal_axes(self):
        '''
        Uses the eigendecomposition to extract the principal axes from the
        moment tensor - returning an instance of the GCMTPrincipalAxes class
        '''
        # Perform eigendecomposition - returns in order P, B, T
        _ = self.eigendecompose(normalise=True)
        principal_axes = GCMTPrincipalAxes()
        # Eigenvalues
        principal_axes.p_axis = {'eigenvalue': self.eigenvalues[0]}
        principal_axes.b_axis = {'eigenvalue': self.eigenvalues[1]}
        principal_axes.t_axis = {'eigenvalue': self.eigenvalues[2]}
        # Eigen vectors
        # 1) P axis
        azim, plun = utils.get_azimuth_plunge(self.eigenvectors[:, 0], True)
        principal_axes.p_axis['azimuth'] = azim
        principal_axes.p_axis['plunge'] = plun
        # 2) B axis
        azim, plun = utils.get_azimuth_plunge(self.eigenvectors[:, 1], True)
        principal_axes.b_axis['azimuth'] = azim
        principal_axes.b_axis['plunge'] = plun
        # 3) T axis
        azim, plun = utils.get_azimuth_plunge(self.eigenvectors[:, 2], True)
        principal_axes.t_axis['azimuth'] = azim
        principal_axes.t_axis['plunge'] = plun
        return principal_axes



class GCMTEvent(object):
    '''
    Basic class representation of a GCMT moment tensor in ndk format
    '''
    def __init__(self):
        '''Instantiate'''

        self.identifier = None
        self.hypocentre = None
        self.centroid = None
        self.magnitude = None
        self.moment = None
        self.metadata = {}
        self.moment_tensor = None
        self.nodal_planes = None
        self.principal_axes = None
        self.f_clvd = None
        self.e_rel = None

    def __repr__(self):
        """
        """
        output_str = "{:s} - {:s} Mw\n".format(self.identifier,
                                               str(self.magnitude))
        return output_str + "\n".join([str(self.hypocentre),
                                       str(self.centroid),
                                       str(self.nodal_planes),
                                       str(self.principal_axes),
                                       str(self.moment_tensor)])

    def get_f_clvd(self):
        '''
        Returns the statistic f_clvd: the signed ratio of the sizes of the
        intermediate and largest principal moments

        f_clvd = -b_axis_eigenvalue /
                  max(|t_axis_eigenvalue|,|p_axis_eigenvalue|)

        '''
        if not self.principal_axes:
            # Principal axes not yet defined for moment tensor - raises error
            raise ValueError('Principal Axes not defined!')

        denominator = np.max(np.array([
            fabs(self.principal_axes.t_axis['eigenvalue']),
            fabs(self.principal_axes.p_axis['eigenvalue'])
            ]))
        self.f_clvd = -self.principal_axes.b_axis['eigenvalue'] / denominator
        return self.f_clvd

    def get_relative_error(self):
        '''
        Returns the relative error statistic (e_rel), defined by Frohlich &
        Davis (1999):
            e_rel = sqrt((U:U) / (M:M))
        where M is the moment tensor, U is the uncertainty tensor and : is the
        tensor dot product
        '''
        if not self.moment_tensor:
            raise ValueError('Moment tensor not defined!')

        numer = np.tensordot(self.moment_tensor.tensor_sigma,
                             self.moment_tensor.tensor_sigma)

        denom = np.tensordot(self.moment_tensor.tensor,
                             self.moment_tensor.tensor)
        self.e_rel = sqrt(numer / denom)
        return self.e_rel

    def get_mechanism_similarity(self, mechanisms):
        '''
        '''
        raise NotImplementedError('Not implemented yet!')

class GCMTCatalogue(object):
    """
    Class to represent a set of moment tensors
    :param list gcmts:
        Moment tensors as list of instances of :class: GCMTEvent
    :param int number_gcmts:
        Number of moment tensors in catalogue
    """
    def __init__(self, start_year=None, end_year=None, gcmts=[]):
        """
        Instantiate catalogue class
        """
        self.gcmts = gcmts
        self.number_gcmts = len(gcmts)
        self.start_year = start_year
        self.end_year = end_year
        self.ids = [gcmt.identifier for gcmt in self.gcmts]

    def number_events(self):
        '''
        Returns number of CMTs - kept for backward compatibility!
        '''
        return len(self.gcmts)

    def __len__(self):
        """
        Returns number of CMTs
        """
        return len(self.gcmts)

    def __getitem__(self, key):
        """
        Returns a specific event by event ID
        """
        if key in self.ids:
            return self.gcmts[self.ids.index(key)]
        else:
            raise KeyError("Event %s not found" % key)

    def __iter__(self):
        """
        Iterates over the GCMTs
        """
        for gcmt in self.gcmts:
            yield gcmt


    def gcmt_to_simple_array(self, centroid_location=True):
        '''
        Converts the GCMT catalogue to a simple array of
        [ID, year, month, day, hour, minute, second, long., lat., depth, Mw,
        strike_1, dip_1, rake_1, strike_2, dip_2, rake_2, b-plunge, b-azimuth,
        p-plunge, p-azimuth, t-plunge, t-azimuth]
        '''
        catalogue = np.zeros([self.number_events(), 26], dtype=float)
        for iloc, tensor in enumerate(self.gcmts):
            catalogue[iloc, 0] = iloc
            if centroid_location:
                catalogue[iloc, 1] = float(tensor.centroid.date.year)
                catalogue[iloc, 2] = float(tensor.centroid.date.month)
                catalogue[iloc, 3] = float(tensor.centroid.date.day)
                catalogue[iloc, 4] = float(tensor.centroid.time.hour)
                catalogue[iloc, 5] = float(tensor.centroid.time.minute)
                catalogue[iloc, 6] = np.round(
                    np.float(tensor.centroid.time.second) +
                    np.float(tensor.centroid.time.microsecond) / 1000000., 2)
                catalogue[iloc, 7] = tensor.centroid.longitude
                catalogue[iloc, 8] = tensor.centroid.latitude
                catalogue[iloc, 9] = tensor.centroid.depth
            else:
                catalogue[iloc, 1] = float(tensor.hypocentre.date.year)
                catalogue[iloc, 2] = float(tensor.hypocentre.date.month)
                catalogue[iloc, 3] = float(tensor.hypocentre.date.day)
                catalogue[iloc, 4] = float(tensor.hypocentre.time.hour)
                catalogue[iloc, 5] = float(tensor.hypocentre.time.minute)
                catalogue[iloc, 6] = np.round(
                    np.float(tensor.centroid.time.second) +
                    np.float(tensor.centroid.time.microsecond) / 1000000., 2)
                catalogue[iloc, 7] = tensor.hypocentre.longitude
                catalogue[iloc, 8] = tensor.hypocentre.latitude
                catalogue[iloc, 9] = tensor.hypocentre.depth
            catalogue[iloc, 10] = tensor.magnitude
            # Nodal planes
            catalogue[iloc, 11] = tensor.nodal_planes.nodal_plane_1['strike']
            catalogue[iloc, 12] = tensor.nodal_planes.nodal_plane_1['dip']
            catalogue[iloc, 13] = tensor.nodal_planes.nodal_plane_1['rake']
            catalogue[iloc, 14] = tensor.nodal_planes.nodal_plane_2['strike']
            catalogue[iloc, 15] = tensor.nodal_planes.nodal_plane_2['dip']
            catalogue[iloc, 16] = tensor.nodal_planes.nodal_plane_2['rake']
            # Principal axes
            catalogue[iloc, 17] = tensor.principal_axes.b_axis['eigenvalue']
            catalogue[iloc, 18] = tensor.principal_axes.b_axis['azimuth']
            catalogue[iloc, 19] = tensor.principal_axes.b_axis['plunge']
            catalogue[iloc, 20] = tensor.principal_axes.p_axis['eigenvalue']
            catalogue[iloc, 21] = tensor.principal_axes.p_axis['azimuth']
            catalogue[iloc, 22] = tensor.principal_axes.p_axis['plunge']
            catalogue[iloc, 23] = tensor.principal_axes.t_axis['eigenvalue']
            catalogue[iloc, 24] = tensor.principal_axes.t_axis['azimuth']
            catalogue[iloc, 25] = tensor.principal_axes.t_axis['plunge']
        return catalogue


    def get_locations(self, use_centroids=True):
        '''
        Function to return the longitude, latitude, depth and corresponding
        uncertainties as a simple numpy arrays
        '''
        location = np.zeros([self.number_events(), 3], dtype=float)
        location_uncertainty = np.zeros([self.number_events(), 3], dtype=float)

        for iloc, tensor in enumerate(self.gcmts):
            if use_centroids:
                # Use centroids
                location[iloc, 0] = tensor.centroid.longitude
                location[iloc, 1] = tensor.centroid.latitude
                location[iloc, 2] = tensor.centroid.depth
                location_uncertainty[iloc, 0] = \
                    tensor.centroid.longitude_error
                location_uncertainty[iloc, 1] = \
                    tensor.centroid.latitude_error
                location_uncertainty[iloc, 2] = \
                    tensor.centroid.depth_error
            else:
                # Use hypocentres
                location[iloc, 0] = tensor.hypocentre.longitude
                location[iloc, 1] = tensor.hypocentre.latitude
                location[iloc, 2] = tensor.hypocentre.depth
                # Uncertainties set to zero

        return location, location_uncertainty

    def serialise_to_hmtk_csv(self, filename, centroid_location=True):
        '''
        Serialise the catalogue to a simple csv format, designed for
        comptibility with the GEM Hazard Modeller's Toolkit
        '''
        header_list = ['eventID', 'Agency', 'year', 'month', 'day', 'hour',
                   'minute', 'second', 'timeError', 'longitude', 'latitude',
                   'SemiMajor90', 'SemiMinor90', 'ErrorStrike', 'depth',
                   'depthError', 'magnitude', 'sigmaMagnitude', 'str1', 'dip1', 'rake1', 'str2', 'dip2', 'rake2']
        with open(filename, 'wt') as fid:
            writer = csv.DictWriter(fid, fieldnames=header_list)
            headers = dict((header, header) for header in header_list)
            writer.writerow(headers)
            print('Writing to simple csv format ...')
            for iloc, tensor in enumerate(self.gcmts):
                # Generic Data
                cmt_dict = {'eventID': iloc + 100000,
                            'Agency': 'GCMT',
                            'SemiMajor90': None,
                            'SemiMinor90': None,
                            'ErrorStrike': None,
                            'magnitude': tensor.magnitude,
                            'sigmaMagnitude': None,
                            'depth': None,
                            'depthError': None,
                            'str1': None,
                            'dip1': None,
                            'rake1': None,
                            'str2': None,
                            'dip2': None,
                            'rake2': None}

                if centroid_location:
                    # Time and location come from centroid
                    cmt_dict['year'] = tensor.centroid.date.year
                    cmt_dict['month'] = tensor.centroid.date.month
                    cmt_dict['day'] = tensor.centroid.date.day
                    cmt_dict['hour'] = tensor.centroid.time.hour
                    cmt_dict['minute'] = tensor.centroid.time.minute
                    cmt_dict['second'] = np.round(
                        np.float(tensor.centroid.time.second) +
                        np.float(tensor.centroid.time.microsecond) / 1000000., 2)
                    cmt_dict['timeError'] = tensor.centroid.time_error
                    cmt_dict['longitude'] = tensor.centroid.longitude
                    cmt_dict['latitude'] = tensor.centroid.latitude
                    cmt_dict['depth'] = tensor.centroid.depth
                    cmt_dict['depthError'] = tensor.centroid.depth_error
                    cmt_dict['str1'] = tensor.nodal_planes.nodal_plane_1['strike']
                    cmt_dict['rake1'] = tensor.nodal_planes.nodal_plane_1['rake']
                    cmt_dict['dip1'] = tensor.nodal_planes.nodal_plane_1['dip']
                    cmt_dict['str2'] = tensor.nodal_planes.nodal_plane_2['strike']
                    cmt_dict['rake2'] = tensor.nodal_planes.nodal_plane_2['rake']
                    cmt_dict['dip2'] = tensor.nodal_planes.nodal_plane_2['dip']
                else:
                    # Time and location come from hypocentre
                    cmt_dict['year'] = tensor.hypocentre.date.year
                    cmt_dict['month'] = tensor.hypocentre.date.month
                    cmt_dict['day'] = tensor.hypocentre.date.day
                    cmt_dict['hour'] = tensor.hypocentre.time.hour
                    cmt_dict['minute'] = tensor.hypocentre.time.minute
                    cmt_dict['second'] = np.round(
                        np.float(tensor.hypocentre.time.second) +
                        np.float(tensor.hypocentre.time.microsecond) / 1000000., 2)
                    cmt_dict['timeError'] = None
                    cmt_dict['longitude'] = tensor.hypocentre.longitude
                    cmt_dict['latitude'] = tensor.hypocentre.latitude
                    cmt_dict['depth'] = tensor.hypocentre.depth
                    cmt_dict['depthError'] = None
                writer.writerow(cmt_dict)
        print('done!')

    def sum_tensor_set(self, selection, weight=None):
        '''
        Function to sum a subset of moment tensors from a list of tensors
        :param list selection:
            Indices of selected tensors from within the list
        '''
        if isinstance(weight, list) or isinstance(weight, np.ndarray):
            assert len(weight) == len(selection)
        else:
            weight = np.ones(len(selection), dtype=float)

        resultant = GCMTEvent()
        resultant.moment_tensor = GCMTMomentTensor()
        resultant.moment_tensor.tensor = 0.
        resultant.centroid = GCMTCentroid(None, None)
        for iloc, locn in enumerate(selection):
            # Normalise input tensor
            target = self.gcmts[locn]
            target = weight[iloc] * \
                (target.moment_tensor.normalise_tensor())[0]
            # Sum tensor
            resultant.moment_tensor.tensor += target

            # Update resultant centroid
            resultant.centroid.longitude += (target.centroid.longitude *
                                             weight[iloc])
            resultant.centroid.latitude += (target.centroid.latitude *
                                            weight[iloc])
            resultant.centroid.depth += (target.centroid.depth * weight[iloc])
        return resultant

    def write_to_gmt_format(self, filename, add_text=False):
        """
        Exports the catalogue to a GMT format (for use with the "Sc" flag).
        :param str filename:
            Name of file

        "Sc" flag requires "Long, Lat, Depth, Stike, Dip, Rake, Strike, Dip,
                            Rake, Mantissa, Exponent, LongPlot, LatPlot, Text"
        """
        with open(filename, "wt") as fid:
            for iloc, gcmt in enumerate(self.gcmts):
                mantissa = gcmt.moment / (10. **
                                          float(gcmt.moment_tensor.exponent))
                exponent = gcmt.moment_tensor.exponent + 7.
                if add_text:
                    print("%9.4f %9.4f %9.4f %6.1f %6.1f %6.1f %6.1f "
                          "%6.1f %6.1f %7.2f %5.1f %9.4f %9.4f %s" % (
                              gcmt.centroid.longitude,
                              gcmt.centroid.latitude,
                              gcmt.centroid.depth,
                              gcmt.nodal_planes.nodal_plane_1['strike'],
                              gcmt.nodal_planes.nodal_plane_1['dip'],
                              gcmt.nodal_planes.nodal_plane_1['rake'],
                              gcmt.nodal_planes.nodal_plane_2['strike'],
                              gcmt.nodal_planes.nodal_plane_2['dip'],
                              gcmt.nodal_planes.nodal_plane_2['rake'],
                              mantissa, exponent, gcmt.centroid.longitude,
                              gcmt.centroid.latitude, gcmt.identifier.strip()),
                          file=fid)
                else:
                    print("%9.4f %9.4f %9.4f %6.1f %6.1f %6.1f %6.1f"
                          "%6.1f %6.1f %7.2f %5.1f %9.4f %9.4f" % (
                              gcmt.centroid.longitude,
                              gcmt.centroid.latitude,
                              gcmt.centroid.depth,
                              gcmt.nodal_planes.nodal_plane_1['strike'],
                              gcmt.nodal_planes.nodal_plane_1['dip'],
                              gcmt.nodal_planes.nodal_plane_1['rake'],
                              gcmt.nodal_planes.nodal_plane_2['strike'],
                              gcmt.nodal_planes.nodal_plane_2['dip'],
                              gcmt.nodal_planes.nodal_plane_2['rake'],
                              mantissa, exponent, gcmt.centroid.longitude,
                              gcmt.centroid.latitude), file=fid)

    def write_to_geojson(self, filename):
        """

        """
        if not HAS_GEOJSON:
            raise NotImplementedError("geojson module not available!")
        feature_set = []
        print("Creating geojson features")
        for i, gcmt in enumerate(self.gcmts):
            # Create Feature  set
            geom = geojson.Point((gcmt.centroid.longitude,
                                  gcmt.centroid.latitude))
            attrs = OrderedDict([
                ("MTID", gcmt.identifier),
                ("Mw", gcmt.magnitude),
                ("Mo", gcmt.moment),
                ("CLong", gcmt.centroid.longitude),
                ("CLat", gcmt.centroid.latitude),
                ("CDepth", gcmt.centroid.depth),
                ("HLong", gcmt.hypocentre.longitude),
                ("HLat", gcmt.hypocentre.latitude),
                ("HDepth", gcmt.hypocentre.depth),
                ("Year", gcmt.centroid.date.year),
                ("Month", gcmt.centroid.date.month),
                ("Day", gcmt.centroid.date.day),
                ("Hour", gcmt.centroid.time.hour),
                ("Minute", gcmt.centroid.time.minute),
                ("Second", gcmt.centroid.time.second)
            ])
            # Nodal planes
            if gcmt.nodal_planes:
                attrs["Strike1"] = gcmt.nodal_planes.nodal_plane_1["strike"]
                attrs["Dip1"] = gcmt.nodal_planes.nodal_plane_1["dip"]
                attrs["Rake1"] = gcmt.nodal_planes.nodal_plane_1["rake"]
                attrs["Strike2"] = gcmt.nodal_planes.nodal_plane_2["strike"]
                attrs["Dip2"] = gcmt.nodal_planes.nodal_plane_2["dip"]
                attrs["Rake2"] = gcmt.nodal_planes.nodal_plane_2["rake"]
            else:
                attrs["Strike1"] = ""
                attrs["Dip1"] = ""
                attrs["Rake1"] = ""
                attrs["Strike2"] = ""
                attrs["Dip2"] = ""
                attrs["Rake2"] = ""
            # Principal axes
            if gcmt.principal_axes:
                attrs["T_Length"] = gcmt.principal_axes.t_axis["eigenvalue"]
                attrs["T_Plunge"] = gcmt.principal_axes.t_axis["plunge"]
                attrs["T_Azimuth"] = gcmt.principal_axes.t_axis["azimuth"]
                attrs["N_Length"] = gcmt.principal_axes.b_axis["eigenvalue"]
                attrs["N_Plunge"] = gcmt.principal_axes.b_axis["plunge"]
                attrs["N_Azimuth"] = gcmt.principal_axes.b_axis["azimuth"]
                attrs["P_Length"] = gcmt.principal_axes.p_axis["eigenvalue"]
                attrs["P_Plunge"] = gcmt.principal_axes.p_axis["plunge"]
                attrs["P_Azimuth"] = gcmt.principal_axes.p_axis["azimuth"]
            else:
                attrs["T_Length"] = ""
                attrs["T_Plunge"] = ""
                attrs["T_Azimuth"] = ""
                attrs["N_Length"] = ""
                attrs["N_Plunge"] = ""
                attrs["N_Azimuth"] = ""
                attrs["P_Length"] = ""
                attrs["P_Plunge"] = ""
                attrs["P_Azimuth"] = ""
            # Moment tensor
            if gcmt.moment_tensor:
                mrr, mtt, mpp, mrt, mrp, mtp =\
                    gcmt.moment_tensor._to_6component()
                attrs["mrr"] = mrr
                attrs["mtt"] = mtt
                attrs["mpp"] = mpp
                attrs["mrt"] = mrt
                attrs["mrp"] = mrp
                attrs["mtp"] = mtp
            else:
                attrs["mrr"] = ""
                attrs["mtt"] = ""
                attrs["mpp"] = ""
                attrs["mrt"] = ""
                attrs["mrp"] = ""
                attrs["mtp"] = ""
            if gcmt.identifier:
                i_d = gcmt.identifier
            else:
                i_d = str(i)
            feature_set.append(geojson.Feature(geometry=geom,
                                               properties=attrs,
                                               id=i_d))
        fcollection = geojson.FeatureCollection(feature_set)
        print("Exporting to file")
        with open(filename, "w") as f:
            geojson.dump(fcollection, f)
        print("Done")
