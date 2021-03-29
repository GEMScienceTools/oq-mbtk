#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) 2015-2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
General class for an earthquake catalogue in ISC (ISF) format
"""

import os
import warnings
import numpy as np
import pandas as pd
import datetime as dt

from math import fabs
from rtree import index
from geojson import LineString, Feature, FeatureCollection, dump

from openquake.cat.utils import decimal_time


DATAMAP = [("eventID", "U20"), ("originID", "U20"), ("Agency", "U14"),
           ("year", "i2"), ("month", "i2"), ("day", "i2"), ("hour", "i2"),
           ("minute", "i2"), ("second", "f2"), ("time_error", "f4"),
           ("longitude", "f4"), ("latitude", "f4"), ("depth", "f4"),
           ("depthSolution", "U1"), ("semimajor90", "f4"),
           ("semiminor90", "f4"), ("error_strike", "f2"),
           ("depth_error", "f4"), ("prime", "i1")]

MAGDATAMAP = [("eventID", "U20"), ("originID", "U20"), ("magnitudeID", "U40"),
              ("value", "f4"), ("sigma", "f4"), ("magType", "U6"),
              ("magAgency", "U14")]


def _generator_function(data):
    for i, tmp in enumerate(data):
        yield (i, (tmp[0], tmp[1], tmp[0], tmp[1]), (tmp[2], tmp[3]))


def datetime_to_decimal_time(date, time):
    '''
    Converts a datetime object to decimal time
    '''
    # Seconds, microseconds to floating seconds
    seconds = np.array(float(time.second))
    microseconds = np.array(float(time.microsecond))
    seconds = seconds + (microseconds / 1.0E6)
    return decimal_time(np.array([date.year]),
                        np.array([date.month]),
                        np.array([date.day]),
                        np.array([time.hour]),
                        np.array([time.minute]),
                        np.array([seconds]))


class Magnitude(object):
    '''
    Stores an instance of a magnitude
    :param str event_id:
        Identifier as Event ID
    :param str origin_id:
        Identifier as Origin ID
    :param float value:
        Magnitude value
    :param str author:
        Magnitude author
    :param str scale:
        Magnitude scale (defaults to UK if not entered)
    :param float sigma:
        Magnitude uncertainty
    :param int stations:
        Number of stations
    '''
    def __init__(self, event_id, origin_id, value, author, scale=None,
                 sigma=None, stations=None):
        """
        """
        self.event_id = event_id
        self.origin_id = origin_id
        self.value = value
        self.author = author
        if scale:
            self.scale = scale
        else:
            self.scale = 'UK'
        self.sigma = sigma
        self.stations = stations
        # Createa ID string from attributes
        if self.value > 10.0:
            # Probably a moment magnitude
            self.magnitude_id = "|".join(["{:s}".format(self.origin_id),
                                          self.author,
                                          "{:.6e}".format(self.value),
                                          self.scale])
        else:
            self.magnitude_id = "|".join(["{:s}".format(self.origin_id),
                                          self.author,
                                          "{:.2f}".format(self.value),
                                          self.scale])

    def compare_magnitude(self, magnitude, tol=1E-3):
        '''
        Compares if a second instance of a magnitude class is the same as the
        current magnitude
        '''
        if ((magnitude.origin_id == self.origin_id) and
                (magnitude.author == self.author) and
                (magnitude.scale == self.scale)):
            if fabs(magnitude.value - self.value) > 0.001:
                print("%s != %s" % (self.__str__(), str(magnitude)))
                raise ValueError('Two magnitudes with same metadata contain '
                                 'different values!')
            return True
        else:
            return False

    def __repr__(self):
        """
        Returns the magnitude identifier
        """
        return self.magnitude_id

    def __eq__(self, eqk, tol=1.0E-3):
        """
        Returns True if the event IDs, magnitudes, scale and author are the
        same, False otherwise
        """
        eq_check = (eqk.event_id == self.event_id) and\
            (eqk.origin_id == self.origin_id) and\
            (fabs(eqk.value - self.value) < tol) and\
            (eqk.scale == self.scale) and\
            (eqk.author == self.author)
        return eq_check


class Location(object):
    '''
    Instance of a magnitude location
    :param int origin_id:
        Identifier as origin ID
    :param float longitude:
        Longitude (decimal degrees)
    :param float latitude:
        Latitude (decimal degrees)
    :param float depth:
        Depth (decimal degrees)
    :param str DepthSolution:
        depthSolution (flag) fixed flag (f = fixed depth station,
                                         d = depth phases,
                                         blank if not a fixed depth)
    :param float semimajor90:
        Semimajor axis of 90 % error ellipse (km)
    :param float semiminor90:
        Semiminor axis of 90 % error ellipse (km)
    :param float error_strike:
        Strike of the semimajor axis of the error ellipse
    :param float depth_error:
        1 s.d. Error on the depth value (km)
    '''
    def __init__(self, origin_id, longitude, latitude, depth,
                 depthSolution=None, semimajor90=None, semiminor90=None,
                 error_strike=None, depth_error=None):
        """
        """
        self.identifier = origin_id
        self.longitude = longitude
        self.latitude = latitude
        self.depth = depth
        self.depthSolution = depthSolution
        self.semimajor90 = semimajor90
        self.semiminor90 = semiminor90
        self.error_strike = error_strike
        self.depth_error = depth_error

    def __str__(self):
        """
        Returns a simple location string that concatenates longitude,
        latitude and depth
        """
        if not self.depth:
            depth_str = ""
        else:
            depth_str = str(self.depth)
        return "%s|%s|%s" % (str(self.longitude),
                             str(self.latitude),
                             depth_str)

    def __eq__(self, loc, tol=1.0E-3):
        """
        Determines if the location is the same
        """
        loc_check = (loc.identifier == self.identifier) and\
            (fabs(loc.longitude - self.longitude) < tol) and\
            (fabs(loc.latitude - self.latitude) < tol) and\
            (fabs(loc.depth - self.depth) < tol)
        return loc_check


class Origin(object):
    """
    In instance of an origin block
    :param int identifier:
        Origin identifier
    :param date:
        Date as instance of datetime.date object
    :param time:
        Time as instance of datetime.time object
    :param location:
        Location as instance of isf_catalogue.Location object
    :param str author:
        Author ID
    :param float time_error:
        Time error (s)
    :param float time_rms:
        Time root-mean-square error (s)
    :param dict metadata:
        Metadata of dictionary including -
        - 'Nphases' - Number of defining phases
        - 'Nstations' - Number of recording stations
        - 'AzimuthGap' - Azimuth Gap of recodring stations
        - 'minDist' - Minimum distance to closest station (degrees)
        - 'maxDist' - Maximum distance to furthest station (degrees)
        - 'FixedTime' - Fixed solution (str)
        - 'DepthSolution' -
        - 'AnalysisType' - Analysis type
        - 'LocationMethod' - Location Method
        - 'EventType' - Event type

    """
    def __init__(self, identifier, date, time, location, author,
                 is_prime=False, is_centroid=False, time_error=None,
                 time_rms=None, metadata=None):
        """
        Instantiates origin
        """
        self.id = identifier
        self.date = date
        self.time = time
        self.location = location
        self.author = author
        self.metadata = metadata
        self.magnitudes = []
        self.is_prime = is_prime
        self.is_centroid = is_centroid
        self.time_error = time_error
        self.time_rms = time_rms
        self.date_time_str = "|".join([str(self.date).replace("-", "|"),
                                       str(self.time).replace(":", "|")])

    def get_number_magnitudes(self):
        """
        Returns the total number of magnitudes associated to the origin
        """
        return len(self.magnitudes)

    def get_magnitude_scales(self):
        """
        Returns the list of magnitude scales associated with the origin
        """
        if self.get_number_magnitudes() == 0:
            return None
        else:
            return [mag.scale for mag in self.magnitudes]

    def get_magnitude_values(self):
        """
        Returns the list of magnitude values associated with the origin
        """
        if self.get_number_magnitudes() == 0:
            return None
        else:
            return [mag.value for mag in self.magnitudes]

    def get_magnitude_tuple(self):
        """
        Returns a list of tuples of (Value, Type) for all magnitudes
        associated with the origin
        """
        if self.get_number_magnitudes() == 0:
            return None
        else:
            return [(mag.value, mag.scale) for mag in self.magnitudes]

    def merge_secondary_magnitudes(self, magnitudes, event_id):
        """
        Merge magnitudes as instances of isf_catalogue.Magnitude into origin
        list.
        """
        if self.get_number_magnitudes() == 0:
            # As no magnitudes currently exist then add all input magnitudes
            # to origin
            for magnitude in magnitudes:
                magnitude.event_id = event_id
                self.magnitudes.append(magnitude)
            return magnitudes
        else:
            new_magnitudes = []
            for magnitude1 in magnitudes:
                if not isinstance(magnitude1, Magnitude):
                    raise ValueError('Secondary magnitude must be instance of '
                                     'isf_catalogue.Magnitude')
                has_magnitude = False
                for magnitude2 in self.magnitudes:
                    # Compare magnitudes
                    has_magnitude = magnitude2.compare_magnitude(magnitude1)
                if not has_magnitude:
                    # Magnitude not in current list - update the event ID
                    # then append
                    magnitude1.event_id = event_id
                    new_magnitudes.append(magnitude1)
                    self.magnitudes.append(magnitude1)
            return new_magnitudes

    def __str__(self):
        """
        Returns an string providing information regarding the origin (namely
        the ID, date, time and location
        """

        return "%s|%s|%s|%s" % (self.id, self.author,
                                self.date_time_str, str(self.location))

    def __eq__(self, orig):
        """
        Determine
        """
        return str(self) == str(orig)


class Event(object):
    '''
    Instance of an event block
    :param int id:
        Event ID
    :param origins:
        List of instances of the Origin class
    :param magnitudes:
        List of instances of the Magnitude class
    :param str description:
        Description string

    '''
    def __init__(self, identifier, origins, magnitudes, description=None):
        """
        Instantiate event object
        """
        self.id = identifier
        self.origins = origins
        self.magnitudes = magnitudes
        self.description = description
        self.comment = ""
        self.induced_flag = ""

    def number_origins(self):
        '''
        Return number of origins associated to event
        '''
        return len(self.origins)

    def get_origin_id_list(self):
        '''
        Return list of origin IDs associated to event
        '''
        return [orig.id for orig in self.origins]

    def get_author_list(self):
        """
        Return list of origin authors associated to event
        """
        return [orig.author for orig in self.origins]

    def number_magnitudes(self):
        """
        Returns number of magnitudes associated to event
        """
        return len(self.magnitudes)

    def magnitude_string(self, delimiter=","):
        """
        Returns the full set of magnitudes as a delimited list of strings
        """
        mag_list = []
        for origin in self.origins:
            for mag in origin.magnitudes:
                mag_list.append(str(mag))
        return delimiter.join(mag_list)

    def assign_magnitudes_to_origins(self):
        """
        Will loop through each origin and assign magnitudes to origin
        """
        if self.number_magnitudes() == 0:
            return ValueError('No magnitudes in event!')
        if self.number_origins() == 0:
            return ValueError('No origins in event!')
        for origin in self.origins:
            for magnitude in self.magnitudes:
                if origin.id == magnitude.origin_id:
                    origin.magnitudes.append(magnitude)

    def merge_secondary_origin(self, origin2set):
        '''
        Merges an instance of an isf_catalogue.Origin class into the set
        of origins.

        :param origin2set:
            An iterable
        '''
        current_id_list = self.get_origin_id_list()
        for origin2 in origin2set:
            if not type(origin2).__name__ == "Origin":
                o_t = type(origin2).__name__
                msg = ('Secondary origins must be instance of ' +
                       'isf_catalogue.Origin class. Found: {:s}'.format(o_t))
                raise ValueError(msg)
            if origin2.id in current_id_list:
                # Origin is already in list - process magnitudes
                location = current_id_list.index(origin2.id)
                origin = self.origins[location]
                new_magnitudes = origin.merge_secondary_magnitudes(
                    origin2.magnitudes, self.id)
                self.magnitudes.extend(new_magnitudes)
                self.origins[location] = origin
            else:
                for magnitude in origin2.magnitudes:
                    magnitude.event_id = self.id
                    self.magnitudes.append(magnitude)
                self.origins.append(origin2)

    def get_origin_mag_vals(self):
        """
        Returns a list of origin and magnitude pairs
        """
        authors = []
        mag_scales = []
        mag_values = []
        mag_sigmas = []
        for origin in self.origins:
            for mag in origin.magnitudes:
                authors.append(mag.author)
                mag_scales.append(mag.scale)
                mag_values.append(mag.value)
                mag_sigmas.append(mag.sigma)
        return authors, mag_scales, mag_values, mag_sigmas

    def __str__(self):
        """
        Return string definition from the ID and description
        """
        return "%s|'%s'" % (str(self.id), self.description)

    def __eq__(self, evnt):
        """
        Compares two events on the basis of ID
        """
        return str(self) == str(evnt)


class ISFCatalogue(object):
    '''
    Instance of an earthquake catalogue
    '''
    def __init__(self, identifier, name, events=None,
                 timezone=dt.timezone(dt.timedelta(hours=0))):
        """
        Instantiate the catalogue with a name and identifier
        """
        self.id = identifier
        self.name = name
        if isinstance(events, list):
            self.events = events
        else:
            self.events = []
        # NOTE we assume that all the origins within a ISFCatalogue instance
        # refer to the same timezone
        self.timezone = timezone
        self.ids = [event.id for event in self.events]

    def __iter__(self):
        """
        If iterable, returns list of events
        """
        for event in self.events:
            yield event

    def __len__(self):
        """
        For len return number of events
        """
        return self.get_number_events()

    def __getitem__(self, key):
        """
        Returns the event corresponding to the specific key
        """
        if not len(self.ids):
            self.ids = [event.id for event in self.events]
        if key in self.ids:
            return self.events[self.ids.index(key)]
        else:
            raise KeyError("Event %s not found" % key)

    def _create_spatial_index(self):
        """
        :return:
            A :class:`rtree.index.Index` instance
        """
        p = index.Property()
        p.dimension = 2
        #
        # Preparing data that will be included in the index
        data = []
        for iloc, event in enumerate(self.events):
            for iori, origin in enumerate(event.origins):
                if not origin.is_prime and len(event.origins) > 1:
                    # Skipping because we have more than one origin and prime
                    # is not defined
                    continue
                else:
                    # Skipping because there is no magnitude defined
                    if len(origin.magnitudes) == 0:
                        continue
                # Saving information regarding the prime origin
                data.append((origin.location.longitude,
                             origin.location.latitude, iloc, iori))
        #
        # Creating the index
        sidx = index.Index(_generator_function(data), properties=p)
        self.sidx = sidx
        self.data = np.array(data)

    # TODO - this does not cope yet with catalogues crossing the international
    # dateline
    def add_external_idf_formatted_catalogue(self, cat, ll_deltas=0.01,
            delta_t=dt.timedelta(seconds=30),
            utc_time_zone=dt.timezone(dt.timedelta(hours=0)),
            buff_t=dt.timedelta(seconds=0), buff_ll=0, use_ids=False,
            logfle=False):
        """
        This merges an external catalogue formatted in the ISF format e.g. a
        catalogue coming from an external agency. Because of this, we assume
        that each event has a single origin.

        :param cat:
            An instance of :class:`ISFCatalogue` i.e. the 'guest' catalogue
        :param ll_deltas:
            A float defining the tolerance in decimal degrees used when looking
            for colocated events
        :param delta_t:
            Tolerance used to find colocated events. It's an instance of
            :class:`datetime.timedelta`
        :param utc_time_zone:
            A :class:`datetime.timezone` instance describing the reference
            timezone for the new catalogue.
        :param buff_t:
            Tolerance used to find events close to the selection threshold.
            It's an instance of :class:`datetime.timedelta`
        :param buff_ll:
            A float defining the tolerance used to find events close to the
            selection threshold.
        :param use_ids:
            A boolean
        :param logfle:
            Name of the file which will contain the log of the processing
        :return:
            - A list with the indexes of the events in the 'guest' catalogue
              added to the 'host' catalogue.
            - A dictionary with doubtful events. The keys in this dictionary
              are the indexes of the events in the 'host' catalogue.
              The values are the indexes of the doubtful events in the 'guest'
              catalogue.
        """
        if logfle:
            fou = open(logfle, 'w')
            fname_geojson = os.path.splitext(logfle)[0]+"_secondary.geojson"

        #
        # This is a dictionary where we store the doubtful events.
        doubts = {}
        #
        # Check if we have a spatial index
        assert 'sidx' in self.__dict__
        #
        # Set delta time thresholds
        if hasattr(delta_t, '__iter__'):
            threshold = np.array([[t[0], t[1].total_seconds()] for t in
                                  delta_t])
        else:
            threshold = np.array([[1000, delta_t.total_seconds()]])
        #
        # Set ll delta thresholds
        if hasattr(ll_deltas, '__iter__'):
            ll_deltas = np.array([d for d in ll_deltas])
        else:
            ll_deltas = np.array([[1000, ll_deltas]])
        #
        # Processing the events in the catalogue 'guest' catalogue
        id_common_events = []
        features = []
        new = 0
        new_old = 0
        common = 0
        common_old = 0
        iloc = 0
        for iloc, event in enumerate(cat.events):

            if logfle:
                msg = 'Index: {:d} Event ID: {:s}\n'.format(iloc, event.id)
                fou.write(msg)

            #
            # Initial settings
            found = False
            before = self.get_number_events()
            #
            # Updating time of the origin to the new timezone
            new_datetime = dt.datetime.combine(event.origins[0].date,
                                               event.origins[0].time,
                                               tzinfo=utc_time_zone)
            new_datetime = new_datetime.astimezone(self.timezone)
            event.origins[0].date = new_datetime.date()
            event.origins[0].time = new_datetime.time()
            #
            # Set the datetime of the event
            dtime_a = dt.datetime.combine(event.origins[0].date,
                                          event.origins[0].time)
            #
            # Take the appropriate value from delta_ll - this is needed in
            # particular when delta_ll varies with time.
            idx_threshold = max(np.argwhere(dtime_a.year > ll_deltas[:, 0]))
            ll_thrs = ll_deltas[idx_threshold, 1]
            #
            # Create selection window
            minlo = event.origins[0].location.longitude - ll_thrs
            minla = event.origins[0].location.latitude - ll_thrs
            maxlo = event.origins[0].location.longitude + ll_thrs
            maxla = event.origins[0].location.latitude + ll_thrs
            #
            # Querying the spatial index
            obj = [n.object for n in self.sidx.intersection((minlo, minla,
                                                             maxlo, maxla),
                                                            objects=True)]
            #
            # This is for checking. We perform the check only if the buffer
            # distance is larger than 0
            obj_e = []
            obj_a = []
            if buff_ll > 0 or buff_t.seconds > 0:
                obj_a = [n.object for n in self.sidx.intersection((
                        minlo-buff_ll, minla-buff_ll, maxlo+buff_ll,
                        maxla+buff_ll), objects=True)]
                obj_b = [n.object for n in self.sidx.intersection((
                        minlo+buff_ll, minla+buff_ll, maxlo-buff_ll,
                        maxla+buff_ll), objects=True)]
                #
                # Find the index of the events in the buffer across the
                # selection window
                obj_e = list(set(obj_a) - set(obj_b))
            #
            # Find the appropriate delta_time
            idx_threshold = max(np.argwhere(dtime_a.year >
                                            threshold[:, 0]))
            sel_thrs = threshold[idx_threshold, 1]

            if logfle:
                msg = '   Selected {:d} events \n'.format(len(obj))
                fou.write(msg)

            if len(obj):
                #
                # Checking the events selected with the spatial index. obj is
                # a list of tuples (event and origin ID) in the host
                # catalogue for the epicenters close to the investigated event
                for i in obj:
                    #
                    # Selecting the origin of the event found in the catalogue
                    i_eve = i[0]
                    i_ori = i[1]
                    orig = self.events[i_eve].origins[i_ori]
                    dtime_b = dt.datetime.combine(orig.date, orig.time)
                    #
                    # Check if time difference is within the threshold value
                    delta = abs((dtime_a - dtime_b).total_seconds())

                    if logfle:
                        eid = self.events[i_eve].id
                        msg = '      Event ID: {:s}\n'.format(eid)
                        msg += '      Delta: {:f}\n'.format(delta)
                        fou.write(msg)

                    if delta < sel_thrs and found is False:

                        # Found an origin in the same space-time window
                        found = True
                        tmp = event.origins

                        # Check this event already contains an origin from
                        # the same agency
                        origins = self.events[i_eve].origins
                        if tmp[0].author in [o.author for o in origins]:

                            fmt = "This event already contains "
                            fmt += " an origin from the same agency: {:s}\n"
                            fmt += " Trying to add evID {:s}\n"
                            msg = fmt.format(tmp[0].author, event.id)
                            warnings.warn(msg)

                            if logfle:
                                fou.write(msg)

                        # Set prime solution is necessary
                        if (len(self.events[i_eve].origins) == 1 and
                                not self.events[i_eve].origins[0].is_prime):
                            tmp[0].is_prime = True
                        else:
                            tmp[0].is_prime = False

                        # Check event ID
                        if use_ids:
                            if event.id != self.events[i_eve].id:
                                fmt = " Trying to add a secondary origin "
                                fmt += " whose ID {:s} differs from the "
                                fmt += " original one. Skipping\n"
                                msg = fmt.format(event.id,
                                                 self.events[i_eve].id)
                                warnings.warn(msg)
                                found = False
                                continue

                        # Check if a secondary solution from the same agency
                        # exists
                        authors = [m.author for m in
                                   self.events[i_eve].magnitudes]
                        if event.magnitudes[0].author in authors:
                            print("Solution already included for this source")
                            print(event.magnitudes[0].origin_id)
                            found = False
                            continue

                        # Info
                        fmt = "Adding to event {:d}\n"
                        msg = fmt.format(i_eve)

                        # Updating the .geojson file
                        if logfle:
                            fou.write(msg)

                            lon1 = self.events[i_eve].origins[0].location.longitude
                            lat1 = self.events[i_eve].origins[0].location.latitude
                            lon2 = tmp[0].location.longitude
                            lat2 = tmp[0].location.latitude
                            line = LineString([(lon1, lat1), (lon2, lat2)])
                            ide = self.events[i_eve].id
                            features.append(Feature(geometry=line,
                                            properties={"originalID": ide}))

                        # Merging a secondary origin
                        self.events[i_eve].merge_secondary_origin(tmp)
                        id_common_events.append(iloc)
                        common += 1

                        break
            #
            # Searching for doubtful events:
            if buff_ll > 1e-10 and buff_t.seconds > 1e-10:
                if len(obj_a) > 0:
                    for i in obj_a:
                        to_add = False
                        #
                        # Selecting origin of the event found in the catalogue
                        i_eve = i[0]
                        i_ori = i[1]
                        orig = self.events[i_eve].origins[i_ori]
                        dtime_b = dt.datetime.combine(orig.date, orig.time)
                        #
                        # Check if time difference within the threshold value
                        tmp_delta = abs(dtime_a - dtime_b).total_seconds()
                        #
                        # Within max distance and across the time buffer
                        tsec = buff_t.total_seconds()
                        if (tmp_delta > (sel_thrs - tsec) and
                                tmp_delta < (sel_thrs + tsec)):
                            to_add = True
                        #
                        # Within max time and within the ll buffer
                        if (not to_add and tmp_delta < (sel_thrs + tsec)):
                            if i in obj_e:
                                to_add = True
                        #
                        # Saving info
                        if to_add:
                            if i[0] in doubts:
                                doubts[i[0]].append(iloc)
                            else:
                                doubts[i[0]] = [iloc]
            #
            # Adding new event
            if not found:
                # Making sure that the ID of the event added does not exist
                # already

                if event.id in set(self.ids):

                    if use_ids:
                        fmt = "Adding a new event whose ID {:s}"
                        fmt += " is already in the DB. Making it secondary."
                        msg = fmt.format(event.id)
                        warnings.warn(msg)

                        if logfle:
                            fou.write(msg)

                        i_eve = np.where(np.array(self.ids) == event.id)
                        tmp = event.origins
                        tmp[0].is_prime = False
                        self.events[i_eve[0][0]].merge_secondary_origin(tmp)
                        found = 1
                        common += 1

                    else:
                        fmt = 'Event ID: {:s} already there. Length ids {:d}'
                        msg = fmt.format(event.id, len(self.ids))
                        raise ValueError(msg)

                else:
                    assert len(event.origins) == 1
                    event.origins[0].is_prime = True
                    self.events.append(event)

                    if logfle:
                        msg = "Adding new event\n"
                        fou.write(msg)

                    self.ids.append(event.id)
                    new += 1
            #
            # Checking
            if (new - new_old) > 0 and (common - common_old > 0):
                msg = '{:d}'.format(iloc)
                raise ValueError(msg)
            elif (new - new_old) > 1:
                msg = 'New increment larger than 1, iloc {:d}'.format(iloc)
                raise ValueError(msg)
            elif (common - common_old) > 1:
                msg = 'Common increment larger than 1, iloc {:d}'.format(iloc)
                raise ValueError(msg)
            else:
                new_old = new
                common_old = common
            #
            #
            after = self.get_number_events()
            #
            #
            # if not iloc % 5000:
            #    idxs, stats = self.get_prime_events_info()
            #    num_primes = [len(stats[k]) for k in stats.keys()]
            #    msg = "{:d}".format(iloc)
            #    assert sum(num_primes) == after, msg

            fmt = 'before {:d} after {:d} iloc {:d} found {:d} loops: {:d}'
            msg = fmt.format(before, after, iloc, found, iloc)
            dlt = 0 if found else 1
            assert before+dlt == after, msg
        #
        # Checking
        fmt = "Wrong budget \n"
        fmt += "Common: {:d} New: {:d} Sum: {:d} Expected: {:d} loops: {:d}\n"
        msg = fmt.format(common, new, common+new, cat.get_number_events(),
                         iloc+1)
        assert (common + new) == cat.get_number_events(), msg
        #
        # Updating the spatial index
        self._create_spatial_index()

        if logfle:
            fou.close()

            feature_collection = FeatureCollection(features)
            with open(fname_geojson, 'w') as f:
                dump(feature_collection, f)

        return id_common_events, doubts

    def get_number_events(self):
        """
        Return number of events
        """
        return len(self.events)

    def get_event_key_list(self):
        """
        Returns list event IDs
        """
        if self.get_number_events() == 0:
            return []
        else:
            return [eq.id for eq in self.events]

    def merge_second_catalogue(self, catalogue):
        '''
        Merge in a second catalogue of the format ISF Catalogue and link via
        Event Keys
        '''
        if not isinstance(catalogue, ISFCatalogue):
            raise ValueError('Input catalogue must be instance of ISF '
                             'Catalogue')

        native_keys = self.get_event_key_list()
        new_keys = catalogue.get_event_key_list()
        for iloc, key in enumerate(new_keys):
            if key in native_keys:
                # Add secondary to primary
                location = native_keys.index(key)
                # Merge origins into catalogue
                event = self.events[location]
                event.merge_secondary_origin(catalogue.events[iloc].origins)
                self.events[location] = event

    def calculate_number_of_unique_events_per_agency(self):
        """
        This method computes the number of unique events provided by each
        agency.

        :returns:
            A dictionary with key the name of the agency and value the number
            of unique events.
        """
        counter = {}
        for iloc, event in enumerate(self.events):
            if len(event.origins) == 1:
                key = event.origins[0].author
                if key in counter:
                    counter[key] += 1
                else:
                    counter[key] = 1
        return counter

    def get_prime_events_info(self):
        """
        :returns:
            This method returns the indexes of the events with more than one
            origin and without the prime origin defined and a dictionary per
            agency with indexes of prime events.
        """
        idxs = []
        stats = {}
        for iloc, event in enumerate(self.events):
            found = False
            for iori, origin in enumerate(event.origins):
                if origin.is_prime or len(event.origins) == 1:
                    key = origin.author
                    if key in stats:
                        stats[key].append((iloc, iori))
                    else:
                        stats[key] = [(iloc, iori)]
                    found = True
                else:
                    continue
            if not found and len(event.origins) > 1:
                idxs.append(iloc)
        return idxs, stats

    def filter_catalogue_by_event_id(self, ids):
        """
        This removes from the catalogue all the events with ID included in the
        list provided.

        :param ids:
            A list of event IDs
        """
        for iloc in sorted(ids, reverse=True):
            del self.events[iloc]

    def get_catalogue_subset(self, ids):
        """
        :param ids:
            An iterable of indexes in [0, |events|-1]
        :returns:
            A new :class:`openquake.cat.isf_catalogue.IDFCatalogue` instance
        """
        newcat = ISFCatalogue(self.id, self.name)
        eids = []
        for iloc in sorted(ids, reverse=True):
            newcat.events.append(self.events[iloc])
            eids.append(self.events[iloc].id)
        # Updating the list of eventss
        newcat.ids = eids
        return newcat

    def get_decimal_dates(self):
        """
        Returns dates and time as a vector of decimal dates
        """
        neq = self.get_number_events()
        year = np.zeros(neq, dtype=int)
        month = np.zeros(neq, dtype=int)
        day = np.zeros(neq, dtype=int)
        hour = np.zeros(neq, dtype=int)
        minute = np.zeros(neq, dtype=int)
        second = np.zeros(neq, dtype=float)
        for iloc, event in enumerate(self.events):
            is_selected = False
            for origin in event.origins:
                if is_selected:
                    continue
                if origin.is_prime:
                    year[iloc] = origin.date.year
                    month[iloc] = origin.date.month
                    day[iloc] = origin.date.day
                    hour[iloc] = origin.time.hour
                    minute[iloc] = origin.time.minute
                    second[iloc] = float(origin.time.second) + \
                        (float(origin.time.microsecond) / 1.0E6)
                    is_selected = True
            if not is_selected:
                # No prime origins - take the first
                year[iloc] = event.origins[0].date.year
                month[iloc] = event.origins[0].date.month
                day[iloc] = event.origins[0].date.day
                hour[iloc] = event.origins[0].time.hour
                minute[iloc] = event.origins[0].time.minute
                second[iloc] = float(event.origins[0].time.second) + \
                    (float(event.origins[0].time.microsecond) / 1.0E6)
        return decimal_time(year, month, day, hour, minute, second)

    def render_to_simple_numpy_array(self):
        """
        Render to a simple array using preferred origin time and magnitude
        :return:
            A :class:`numpy.ndarray` instance
        """
        decimal_time = self.get_decimal_dates()
        decimal_time = decimal_time.tolist()
        simple_array = []
        for iloc, event in enumerate(self.events):
            for origin in event.origins:
                if not origin.is_prime and len(event.origins) > 1:
                    continue
                else:
                    if len(origin.magnitudes) == 0:
                        continue

                    simple_array.append([event.id,
                                         origin.id,
                                         decimal_time[iloc],
                                         origin.location.latitude,
                                         origin.location.longitude,
                                         origin.location.depth,
                                         origin.magnitudes[0].value])
        return np.array(simple_array)

    def get_origin_mag_tables(self):
        """
        Returns the full ISF catalogue as a pair of tables, the first
        containing only the origins, the second containing the
        magnitudes
        """
        # Find out size of tables
        n_origins = 0
        n_mags = 0
        for eq in self.events:
            n_origins += len(eq.origins)
            n_mags += len(eq.magnitudes)
        # Pre-allocate data to zeros
        origin_data = np.zeros((n_origins,), dtype=DATAMAP)
        mag_data = np.zeros((n_mags,), dtype=MAGDATAMAP)
        o_counter = 0
        m_counter = 0
        for eq in self.events:
            for orig in eq.origins:
                # Convert seconds fromd datetime to float
                seconds = float(orig.time.second) +\
                    float(orig.time.microsecond) / 1.0E6
                # Optional defaults
                if orig.time_error:
                    time_error = orig.time_error
                else:
                    time_error = 0.0
                if orig.location.semimajor90:
                    semimajor90 = orig.location.semimajor90
                    semiminor90 = orig.location.semiminor90
                    error_strike = orig.location.error_strike
                else:
                    semimajor90 = 0.0
                    semiminor90 = 0.0
                    error_strike = 0.0

                if orig.location.depth_error:
                    depth_error = orig.location.depth_error
                else:
                    depth_error = 0.0

                if (orig.location.depthSolution == 'None' or
                        orig.location.depthSolution == '' or
                        orig.location.depthSolution is None):
                    depthSolution = np.NaN
                elif orig.location.depthSolution:
                    depthSolution = orig.location.depthSolution
                else:
                    print('Location:', orig.location.depthSolution)
                    raise ValueError("Unsupported case")

                if (orig.location.depth == 'None' or
                        orig.location.depth == '' or
                        orig.location.depth is None):
                    depth = np.NaN
                elif orig.location.depth < 0.01:
                    depth = orig.location.depth
                    fmt = "Warning, depth <= 0.0 (id:{:s})"
                    warnings.warn(fmt.format(eq.id))
                elif orig.location.depth:
                    depth = orig.location.depth
                else:
                    print('Depth:', orig.location.depth)
                    raise ValueError("Unsupported case")

                if orig.is_prime or len(eq.origins) == 1:
                    prime = 1
                else:
                    prime = 0

                origin_data[o_counter] = (eq.id, orig.id, orig.author,
                                          orig.date.year, orig.date.month,
                                          orig.date.day, orig.time.hour,
                                          orig.time.minute, seconds,
                                          time_error, orig.location.longitude,
                                          orig.location.latitude,
                                          depth,
                                          depthSolution,
                                          semimajor90, semiminor90,
                                          error_strike, depth_error, prime)

                o_counter += 1

            for mag in eq.magnitudes:
                if mag.sigma:
                    sigma = mag.sigma
                else:
                    sigma = 0.0
                mag_data[m_counter] = (mag.event_id, mag.origin_id,
                                       mag.magnitude_id, mag.value,
                                       sigma, mag.scale, mag.author)
                m_counter += 1
        return origin_data, mag_data

    def build_dataframe(self, hdf5_file=None):
        """
        Renders the catalogue into two Pandas Dataframe objects, one
        representing the full list of origins, the other the full list
        of magnitudes
        :param str hd5_file:
            Path to the hdf5 for writing
        :returns:
            orig_df - Origin dataframe
            mag_df  - Magnitude dataframe
        """
        origin_data, mag_data = self.get_origin_mag_tables()
        orig_df = pd.DataFrame(origin_data,
                               columns=[val[0] for val in DATAMAP])
        mag_df = pd.DataFrame(mag_data,
                              columns=[val[0] for val in MAGDATAMAP])
        if hdf5_file:
            store = pd.HDFStore(hdf5_file)
            store.append("catalogue/origins", orig_df)
            store.append("catalogue/magnitudes", mag_df)
            store.close()
        return orig_df, mag_df

    def render_to_xyzm(self, filename, frmt='%.3f'):
        '''
        Writes the catalogue to a simple [long, lat, depth, mag] format - for
        use in GMT
        '''
        # Get numpy array
        print('Creating array ...')
        cat_array = self.render_to_simple_numpy_array()
        cat_array = cat_array[:, [4, 3, 5, 6]]
        print('Writing to file ...')
        df = pd.DataFrame(cat_array)
        df.to_csv(filename, index=False, header=False)
        print('done!')

    def quick_export(self, filename, delimiter=","):
        """
        Rapidly exports the catalogue to an ascii format
        """
        with open(filename, "w") as f:
            print("eventID,Description,originID,year,month,day,hour,"
                  "minute,second,longitude,latitude,depth,magOriginID,"
                  "magAgency,magnitude,magScale", file=f)
            for event in self.events:
                base_str = str(event)
                for origin in event.origins:
                    output_strings = [base_str, str(origin)]
                    output_strings.extend([str(m) for m in origin.magnitudes])
                    output_str = "|".join(output_strings)
                    print(output_str.replace("|", delimiter), file=f)
            print("Exported to %s" % filename)
