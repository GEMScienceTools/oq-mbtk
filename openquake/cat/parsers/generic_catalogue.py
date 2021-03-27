# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (c) 2015 GEM Foundation
#
# The Catalogue Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# with this download. If not, see <http://www.gnu.org/licenses/>

"""
Class to hold a general csv formatted catalogue to write to other formats
"""
import datetime
import numpy as np
import openquake.cat.gcmt_utils as utils

from math import floor
from openquake.cat.gcmt_catalogue import (GCMTHypocentre, GCMTCentroid,
                                          GCMTMomentTensor, GCMTEvent,
                                          GCMTCatalogue)
from openquake.cat.isf_catalogue import (Magnitude, Location, Origin,
                                         Event, ISFCatalogue)


class GeneralCsvCatalogue(object):
    """
    Class to parse the ISC GEM file to a complete GCMT catalogue class
    """
    FLOAT_ATTRIBUTE_LIST = ['second', 'timeError', 'longitude', 'latitude',
                            'SemiMajor90', 'SemiMinor90', 'ErrorStrike',
                            'depth', 'depthError', 'magnitude',
                            'sigmaMagnitude', 'moment', 'mpp', 'mpr', 'mrr',
                            'mrt', 'mtp', 'mtt']

    INT_ATTRIBUTE_LIST = ['year', 'month', 'day', 'hour', 'minute',
                          'flag', 'scaling']

    STRING_ATTRIBUTE_LIST = ['eventID', 'Agency', 'magnitudeType', 'comment',
                             'source']

    TOTAL_ATTRIBUTE_LIST = list(
        (set(FLOAT_ATTRIBUTE_LIST).union(
            set(INT_ATTRIBUTE_LIST))).union(
                 set(STRING_ATTRIBUTE_LIST)))

    def __init__(self):
        """
        Initialise the catalogue with an empty data dictionary
        """
        self.data = {}
        for attribute in self.TOTAL_ATTRIBUTE_LIST:
            if attribute in self.FLOAT_ATTRIBUTE_LIST:
                self.data[attribute] = np.array([], dtype=float)
            elif attribute in self.INT_ATTRIBUTE_LIST:
                self.data[attribute] = np.array([], dtype=int)
            else:
                self.data[attribute] = []
        self.number_earthquakes = 0
        self.gcmt_catalogue = GCMTCatalogue()

    def get_number_events(self):
        """
        Returns the number of events
        """
        for key in self.data.keys():
            if len(self.data[key]) > 0:
                return len(self.data[key])
        return 0

    def write_to_gcmt_class(self):
        """
        Exports the catalogue to an instance of the :class:
        eqcat.gcmt_catalogue.GCMTCatalogue
        """
        for iloc in range(0, self.get_number_events()):
            gcmt = GCMTEvent()
            gcmt.identifier = self.data['eventID'][iloc]
            gcmt.magnitude = self.data['magnitude'][iloc]
            # Get moment plus scaling
            if not np.isnan(self.data['moment'][iloc]):
                scaling = float(self.data['scaling'][iloc])
                gcmt.moment = self.data['moment'][iloc] * (10. ** scaling)
            gcmt.metadata = {'Agency': self.data['Agency'][iloc],
                             'source': self.data['source'][iloc]}

            # Get the hypocentre
            gcmt.hypocentre = GCMTHypocentre()
            gcmt.hypocentre.source = self.data['source'][iloc]
            gcmt.hypocentre.date = datetime.date(self.data['year'][iloc],
                                                 self.data['month'][iloc],
                                                 self.data['day'][iloc])
            second = self.data['second'][iloc]
            microseconds = int((second - floor(second)) * 1000000)

            gcmt.hypocentre.time = datetime.time(self.data['hour'][iloc],
                                                 self.data['minute'][iloc],
                                                 int(floor(second)),
                                                 microseconds)
            gcmt.hypocentre.longitude = self.data['longitude'][iloc]
            gcmt.hypocentre.latitude = self.data['latitude'][iloc]

            setattr(gcmt.hypocentre,
                    'semi_major_90',
                    self.data['SemiMajor90'][iloc])

            setattr(gcmt.hypocentre,
                    'semi_minor_90',
                    self.data['SemiMinor90'][iloc])

            setattr(gcmt.hypocentre,
                    'error_strike',
                    self.data['ErrorStrike'][iloc])

            # Get the centroid - basically just copying across the hypocentre
            gcmt.centroid = GCMTCentroid(gcmt.hypocentre.date,
                                         gcmt.hypocentre.time)
            gcmt.centroid.longitude = gcmt.hypocentre.longitude
            gcmt.centroid.latitude = gcmt.hypocentre.latitude
            gcmt.centroid.depth = gcmt.hypocentre.depth
            gcmt.centroid.depth_error = self.data['depthError'][iloc]

            if self._check_moment_tensor_components(iloc):
                # Import tensor components
                gcmt.moment_tensor = GCMTMomentTensor()
                # Check moment tensor has all the components!
                gcmt.moment_tensor.tensor = utils.COORD_SYSTEM['USE'](
                    self.data['mrr'][iloc],
                    self.data['mtt'][iloc],
                    self.data['mpp'][iloc],
                    self.data['mrt'][iloc],
                    self.data['mpr'][iloc],
                    self.data['mtp'][iloc])
                gcmt.moment_tensor.tensor_sigma = np.array([[0., 0., 0.],
                                                            [0., 0., 0.],
                                                            [0., 0., 0.]])
                # Get nodal planes
                gcmt.nodal_planes = gcmt.moment_tensor.get_nodal_planes()
                gcmt.principal_axes = gcmt.moment_tensor.get_principal_axes()

                # Done - append to catalogue
                self.gcmt_catalogue.gcmts.append(gcmt)

        return self.gcmt_catalogue

    def write_to_isf_catalogue(self, catalogue_id, name):
        """
        Exports the catalogue to an instance of the :class:
        eqcat.isf_catalogue.ISFCatalogue
        """
        isf_cat = ISFCatalogue(catalogue_id, name)
        print('write_to', self.get_number_events())

        for iloc in range(0, self.get_number_events()):
            # Origin ID
            if len(self.data['eventID']) > 0:
                event_id = str(self.data['eventID'][iloc])
            else:
                raise ValueError('Unknown key. Line: {:d}'.format(iloc))
            origin_id = event_id
            # Create Magnitude
            sigma_mag = None
            if ('sigmaMagnitude' in self.data and
                    len(self.data['sigmaMagnitude']) > 0):
                sigma_mag = self.data['sigmaMagnitude'][iloc]
            mag = [Magnitude(event_id,
                             origin_id,
                             self.data['magnitude'][iloc],
                             catalogue_id,
                             scale='Mw',
                             sigma=sigma_mag)]
            # Create Moment
            if 'moment' in self.data and len(self.data['moment']):
                if not np.isnan(self.data['moment'][iloc]):
                    moment = self.data['moment'][iloc] *\
                        (10. ** self.data['scaling'][iloc])
                    mag.append(Magnitude(event_id,
                                         origin_id,
                                         moment,
                                         catalogue_id,
                                         scale='Mo'))

            # Create Location
            if len(self.data['SemiMajor90']):
                semimajor90 = self.data['SemiMajor90'][iloc]
            else:
                semimajor90 = np.nan
            if len(self.data['SemiMinor90']):
                semiminor90 = self.data['SemiMinor90'][iloc]
            else:
                semiminor90 = np.nan
            if len(self.data['ErrorStrike']):
                error_strike = self.data['ErrorStrike'][iloc]
            else:
                error_strike = np.nan
            if len(self.data['ErrorStrike']):
                depth_error = self.data['depthError'][iloc]
            else:
                depth_error = np.nan
            #
            if np.isnan(semimajor90):
                semimajor90 = None
            if np.isnan(semiminor90):
                semiminor90 = None
            if np.isnan(error_strike):
                error_strike = None
            if np.isnan(depth_error):
                depth_error = None

            locn = Location(origin_id,
                            self.data['longitude'][iloc],
                            self.data['latitude'][iloc],
                            self.data['depth'][iloc],
                            semimajor90,
                            semiminor90,
                            error_strike,
                            depth_error)

            # Create Origin
            # Date
            if len(self.data['day']) > 1 and self.data['day'][iloc] == 0:
                self.data['day'][iloc] = 1
            if len(self.data['month']) > 1 and self.data['month'][iloc] == 0:
                self.data['month'][iloc] = 1
            try:
                eq_date = datetime.date(self.data['year'][iloc],
                                        self.data['month'][iloc],
                                        self.data['day'][iloc])
            except ValueError:
                print('skipping ',
                      iloc, self.data['year'][iloc], self.data['month'][iloc],
                      self.data['day'][iloc], self.data['magnitude'][iloc])
                continue

            # Time
            secs = self.data['second'][iloc]

            microsecs = int((secs - floor(secs)) * 1E6)
            eq_time = datetime.time(self.data['hour'][iloc],
                                    self.data['minute'][iloc],
                                    int(secs),
                                    microsecs)
            origin = Origin(origin_id, eq_date, eq_time, locn, catalogue_id,
                            is_prime=True)
            origin.magnitudes = mag
            event = Event(event_id, [origin], origin.magnitudes)

            if 'mrr' in self.data and len(self.data['mrr']):
                if self._check_moment_tensor_components(iloc):
                    # If a moment tensor is found then add it to the event
                    moment_tensor = GCMTMomentTensor()
                    scaling = 10. ** self.data['scaling'][iloc]
                    moment_tensor.tensor = scaling * utils.COORD_SYSTEM['USE'](
                        self.data['mrr'][iloc],
                        self.data['mtt'][iloc],
                        self.data['mpp'][iloc],
                        self.data['mrt'][iloc],
                        self.data['mpr'][iloc],
                        self.data['mtp'][iloc])
                    moment_tensor.exponent = self.data['scaling'][iloc]
                    setattr(event, 'tensor', moment_tensor)
            isf_cat.events.append(event)
        return isf_cat

    def _check_moment_tensor_components(self, iloc):
        '''
        Check to see is any tensor components are missing - will be a NaN.
        If it is not possible to construct the full moment tensor then it
        is assumed the tensor does not exist.
        '''
        for component in ['mrr', 'mtt', 'mpp', 'mrt', 'mpr', 'mtp']:
            if np.isnan(self.data[component][iloc]):
                return False
        return True


class MixedMagnitudeCsvCatalogue(GeneralCsvCatalogue):
    """
    """
    def write_to_isf_catalogue(self, catalogue_id, name):
        """
        Exports the catalogue to an instance of the :class:
        eqcat.isf_catalogue.ISFCatalogue
        """
        isf_cat = ISFCatalogue(catalogue_id, name)
        for iloc in range(0, self.get_number_events()):
            # Origin ID
            event_id = str(self.data['eventID'][iloc])
            origin_id = event_id
            # Create Magnitude
            mag = self.data["magnitude"][iloc]
            if not mag or np.isnan(mag):
                # No magnitude - not useful
                continue

            mag = [Magnitude(event_id,
                             origin_id,
                             mag,
                             catalogue_id,
                             scale=self.data["magnitudeType"][iloc],
                             sigma=self.data['sigmaMagnitude'][iloc])]

            # Create Location
            semimajor90 = self.data['SemiMajor90'][iloc]
            semiminor90 = self.data['SemiMinor90'][iloc]
            error_strike = self.data['ErrorStrike'][iloc]
            if np.isnan(semimajor90):
                semimajor90 = None
            if np.isnan(semiminor90):
                semiminor90 = None
            if np.isnan(error_strike):
                error_strike = None
            depth_error = self.data['depthError'][iloc]
            if np.isnan(depth_error):
                depth_error = None
            locn = Location(origin_id,
                            self.data['longitude'][iloc],
                            self.data['latitude'][iloc],
                            self.data['depth'][iloc],
                            semimajor90,
                            semiminor90,
                            error_strike,
                            depth_error)

            # Create Origin
            # Date
            eq_date = datetime.date(self.data['year'][iloc],
                                    self.data['month'][iloc],
                                    self.data['day'][iloc])
            # Time
            secs = self.data['second'][iloc]

            microsecs = int((secs - floor(secs)) * 1E6)
            eq_time = datetime.time(self.data['hour'][iloc],
                                    self.data['minute'][iloc],
                                    int(secs),
                                    microsecs)
            origin = Origin(origin_id, eq_date, eq_time, locn, catalogue_id,
                            is_prime=True)
            origin.magnitudes = mag
            event = Event(event_id, [origin], origin.magnitudes)
            isf_cat.events.append(event)
        return isf_cat
