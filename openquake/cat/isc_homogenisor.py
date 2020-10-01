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
Concept for a simple magnitude homogenisor

Magnitude Conversion rules:

1) If Mw, accept Mw
2) If Ms-ISC, use new ISC-GEM relation
3) If mb-ISC, use new ISC-GEM relation
4) If Ms-PAS, use ????
5) Reject event

Origin conversion

1) EHB location
2) ISC location

"""

from __future__ import print_function
import numpy as np
from scipy.misc import derivative
from datetime import date
from math import exp, sqrt
from openquake.cat.utils import _prepare_coords


def is_GCMTMw(magnitude):
    '''
    '''
    return magnitude


def is_GCMTMw_Sigma(magnitude):
    '''
    '''
    return 0.0


def ISCMs_toGCMTMw(magnitude):
    '''
    Converts an ISC-Ms value to Mw using the ISC-GEM exponential regression
    model
    '''
    return [exp(-0.22 + (0.23 * m)) + 2.86 for m in magnitude]


def ISCMs_toGCMTMw_Sigma(magnitude):
    '''
    '''
    return 0.2


def ISCmb_toGCMTMw(magnitude):
    '''
    Converts an ISC-mb value to Mw using the ISC-GEM exponential regression
    model
    '''
    return [exp(-4.66 + (0.86 * m)) + 4.56 for m in magnitude]


def ISCmb_toGCMTMw_Sigma(magnitude):
    '''
    '''
    return 0.3


def ISCGORMs_toGCMTMw(magnitude):
    '''
    Converts an ISC-Ms value to Mw using the ISC-GEM general orthogonal
    regression model
    '''
    return [0.67 * m + 2.13 if m <= 6.47 else 1.10 * m - 0.67 
            for m in magnitude]


def ISCGORMs_toGCMTMw_Sigma(magnitude):
    '''
    '''
    return 0.2


def ISCGORmb_toGCMTMw(magnitude):
    '''
    Converts an ISC-mb value to Mw using the ISC-GEM general orthogonal
    regression model
    '''
    return [1.38 * m - 1.79 for m in magnitude]


def ISCGORmb_toGCMTMw_Sigma(magnitude):
    '''
    '''
    return 0.3


def PASMs_toGCMTMw(magnitude):
    '''
    Approximate estimator fo convert and Ms from the PAS scale to Moment
    Magnitude Mw
    '''
    return magnitude


def PASMs_toGCMTMw_Sigma(magnitude):
    '''
    '''
    return 0.2


class MagnitudeConversionRule(object):
    '''
    Defines a Rule for converting a magnitude
    '''
    def __init__(self, author, scale, model, sigma_model=None, start_date=None,
                 end_date=None, key=None, model_name=None):
        '''
        Applies to
        '''
        self.author = author
        self.scale = scale
        self.model = model
        if sigma_model:
            self.sigma_model = sigma_model
        else:
            self.sigma_model = None
        self.key = key
        if not model_name:
            self.model_name = self.model.__name__
        else:
            self.model_name = model_name
        if not start_date or isinstance(start_date, date):
            self.start = start_date
        elif isinstance(start_date, str):
            self.start = date(*list(map(int, start_date.split("/"))))
        else:
            raise ValueError("Start date must be instance of datetime.date"
                             " class or string formatted as YYYY/MM/DD")
        if not end_date or isinstance(end_date, date):
            self.finish = end_date
        elif isinstance(end_date, str):
            self.finish = date(*list(map(int, end_date.split("/"))))
        else:
            raise ValueError("End date must be instance of datetime.date"
                             " class or string formatted as YYYY/MM/DD")

    def __str__(self):
        """
        Returns a descriptive string of the rule
        """
        if self.sigma_model:
            return "{:s}-{:s}-{:s}-{:s}".format(self.author,
                                                self.scale,
                                                self.model_name,
                                                self.sigma_model.__name__)
        else:
            return "{:s}-{:s}-{:s}".format(self.author,
                                           self.scale,
                                           self.model_name)

    def convert_value(self, magnitude, sigma):
        '''
        Converts a magnitude and corresponding stda
        '''
        output_mag = self.model(magnitude)
        if sigma:
            output_sigma = self.propagate_sigma(magnitude, sigma)
        else:
            output_sigma = self.sigma_model(magnitude)

        return output_mag, output_sigma

    def propagate_sigma(self, magnitude, sigma):
        '''
        Does simple error propagation
        err_final = sqrt(sigma_model ** 2. + (d(model)/d(mag)) ** 2
        '''
        if self.sigma_model:
            deriv = derivative(self.model, magnitude)
            return sqrt((self.sigma_model(magnitude) ** 2.) + (deriv ** 2.) *
                        (sigma ** 2.))
        else:
            return sigma

    def get_residual(self, input_mag, observed_mag):
        """
        Determines the residual value of a given magnitude based on the
        conversion relation
        """
        expected_mag, sigma = self.convert_value(input_mag, 0)
        return (observed_mag - expected_mag) / sigma, expected_mag, sigma


class OriginRule(object):
    """

    """
    def __init__(self, origin_rule, start_date=None, end_date=None, key=None):
        """

        """
        self.rule = origin_rule
        self.key = key
        if not start_date or isinstance(start_date, date):
            self.start = start_date
        elif isinstance(start_date, str):
            self.start = date(*list(map(int, start_date.split("/"))))
        else:
            raise ValueError("Start date must be instance of datetime.date"
                             " class or string formatted as YYYY/MM/DD")
        if not end_date or isinstance(end_date, date):
            self.finish = end_date
        elif isinstance(end_date, str):
            self.finish = date(*list(map(int, end_date.split("/"))))
        else:
            raise ValueError("End date must be instance of datetime.date"
                             " class or string formatted as YYYY/MM/DD")


MAGNITUDE_RULES = [
    MagnitudeConversionRule('ISC-GEM', 'Mw', is_GCMTMw, is_GCMTMw_Sigma),
    MagnitudeConversionRule('GCMT', 'Mw', is_GCMTMw, is_GCMTMw_Sigma),
    MagnitudeConversionRule('HRVD', 'Mw', is_GCMTMw, is_GCMTMw_Sigma),
    MagnitudeConversionRule('NIED', 'Mw', is_GCMTMw, is_GCMTMw_Sigma),
    MagnitudeConversionRule('ISC', 'Ms', ISCGORMs_toGCMTMw,
                            ISCGORMs_toGCMTMw_Sigma),
    MagnitudeConversionRule('ISC', 'mb', ISCGORmb_toGCMTMw,
                            ISCGORmb_toGCMTMw_Sigma),
    MagnitudeConversionRule('PAS', 'Ms', PASMs_toGCMTMw, PASMs_toGCMTMw_Sigma)]

ORIGIN_RULES = ['ISC-GEM', 'EHB', 'ISC', 'GCMT', 'HRVD', 'GUTE']


def _to_str(value):
    """

    """
    if value:
        return str(value)
    else:
        return ""


class Homogenisor(object):
    '''
    Function to homogenise the ISF Class
    '''
    def __init__(self, catalogue):
        '''

        '''
        self.catalogue = catalogue
        self.mag_rules = None
        self.orig_rules = None

    def homogenise(self, magnitude_rules, origin_rules):
        '''

        '''
        self.mag_rules = magnitude_rules
        self.orig_rules = origin_rules
        for event in self.catalogue.events:
            # Set attribute preferred_solution
            setattr(event, 'preferred', None)
            # Apply origin selection
            pref_origin, author = self._apply_origin_selection(event)

            if pref_origin:
                setattr(pref_origin, 'magnitude', None)
                setattr(pref_origin, 'magnitude_sigma', None)
                setattr(pref_origin, 'record_key', None)
                # Apply magnitude selection
                mag, mag_unc, mag_rec = self._apply_magnitude_selection(event)
                if mag:
                    pref_origin.magnitude = round(mag, 2)
                    if mag_unc:
                        pref_origin.magnitude_sigma = round(mag_unc, 3)
                    else:
                        pref_origin.magnitude_sigma = 0.
                    pref_origin.record_key = "|".join([author, mag_rec])
                    print(event.id, mag_rec, pref_origin.record_key)
                else:
                    # No magnitude can be converted - reject origin
                    pref_origin = None
                    print("% s -- None --  None" % event.id)
                event.preferred = pref_origin
        return self.catalogue

    def _apply_origin_selection(self, event):
        '''
        Checks each agency to see if it is found in the event, returning the
        corresponding origin if so.
        '''
        agencies = event.get_author_list()
        for iloc, author in enumerate(self.orig_rules):
            if author in agencies:
                # Has a solution by the preferred agency
                return event.origins[agencies.index(author)], author
        return False, None

    def _apply_magnitude_selection(self, event):
        '''
        For the preferred origin, select the corresponding magnitude
        '''
        mag_agencies, mag_scales, mag_values, mag_sigmas = \
            event.get_origin_mag_vals()
        # Render all scales to upper
        mag_scales = [mag.upper() for mag in mag_scales]
        for mag_rule in self.mag_rules:
            for iloc in range(len(mag_agencies)):
                if ((mag_rule.author == mag_agencies[iloc]) and
                    (mag_rule.scale.upper() == mag_scales[iloc])):
                    mag_value, mag_sigma = mag_rule.convert_value(
                        mag_values[iloc],
                        mag_sigmas[iloc])
                    return mag_value, mag_sigma, "-".join([mag_rule.author,
                                                           mag_rule.scale])
        return None, None, None

    def export_homogenised_to_csv(self, filename, default_depth=10.0):
        """
        Writes the catalogue to a simple csv format
        [eventID, Agency, OriginID, year, month, day, hour, minute, second,
        timeError, longitude, latitude, SemiMajor90, SemiMinor90, ErrorStrike,
        depth, depthError, magnitude, sigmaMagnitude]

        As some programmes may not be able to process the nans when depth
        is missing they may be unable to treat depth as numerical data
        (... I'm looking at you QGIS!). Set a default numerical value
        for the caes when the depth is missing
        """
        name_list = ['eventID', 'Agency', 'Identifier', 'year', 'month', 'day',
                     'hour', 'minute', 'second', 'timeError', 'longitude',
                     'latitude', 'SemiMajor90', 'SemiMinor90', 'ErrorStrike',
                     'depth', 'depthError', 'magnitude', 'sigmaMagnitude',
                     'Anthropogenic']
        fid = open(filename, "wt")
        # Write header
        print(",".join(name_list), file=fid)
        default_depth = str(default_depth)
        for event in self.catalogue.events:
            if hasattr(event, "preferred") and event.preferred is not None:
                eqk = event.preferred
                if eqk.location.depth:
                    depth_str = str(eqk.location.depth)
                else:
                    depth_str = default_depth
                second = (round(float(eqk.time.second) +
                          float(eqk.time.microsecond) / 1.0E6, 2))
                row_str = ",".join([str(event.id),
                                    eqk.author,
                                    eqk.record_key,
                                    str(eqk.date.year),
                                    str(eqk.date.month),
                                    str(eqk.date.day),
                                    str(eqk.time.hour),
                                    str(eqk.time.minute),
                                    str(second),
                                    _to_str(eqk.time_error),
                                    str(eqk.location.longitude),
                                    str(eqk.location.latitude),
                                    _to_str(eqk.location.semimajor90),
                                    _to_str(eqk.location.semiminor90),
                                    _to_str(eqk.location.error_strike),
                                    str(eqk.location.depth),
                                    _to_str(eqk.location.depth_error),
                                    str(eqk.magnitude),
                                    _to_str(eqk.magnitude_sigma),
                                    str(event.induced_flag),
                                    event.magnitude_string()])
                print(row_str, file=fid)
        fid.close()


def _date_from_string(string, delim="/"):
    """
    Return a date object from YYYY-mm-dd delimited by a specified character
    """
    year, month, day = string.split(delim)
    return date(year, month, day)


class HomogenisorPreprocessor(object):
    """
    Generic pre-processing tool for determining which rules in a set
    should be applied

    e.g. Example Time rules
    [(1900/01/01 - 1990/01/01, [XXX, YYY, ZZZ]),
     (1990/01/01 - 2015/12/31, [YYY, VVV, XXX])]

    e.g. Example key rules
    [(COUNTRY_NAME_1, [XXX, YYY, ZZZ]),
     (COUNTRY_NAME_2, [YYY, VVV, XXX])]

    e.g. Example Depth rules
    [(0.0 - 20.0, [XXX, YYY, ZZZ]),
     (20.0 - 1000.0, [YYY, VVV, XXX])]

    e.g. Example Time + Key rules
    [(1900/01/01 - 1990/01/01 | COUNTRY_NAME_1, [XXX, YYY, ZZZ]),
     (1990/01/01 - 2015/12/31 | COUNTRY_NAME_2, [YYY, VVV, XXX])]

    e.g. Example Depth + Key Rules
    [(0.0 - 20.0 | COUNTRY_NAME_1, [XXX, YYY, ZZZ]),
     (20.0 - 1000.0 | COUNTRY_NAME_2, [YYY, VVV, XXX])]

    e.g. Example Time + Depth Rules
    [(1900/01/01 - 1990/01/01 | 0.0 - 20.0, [XXX, YYY, ZZZ]),
     (1990/01/01 - 2015/12/31 | 20.0 - 1000.0, [YYY, VVV, XXX])]
    """

    def __init__(self, rule_type):
        """

        """
        assert rule_type in ["time", "key", "depth", "time|key", "time|depth",
                             "depth|key"]
        self.rule_type = rule_type
        self.calculation_type = {
            "time": self.time_selection,
            "key": self.key_selection,
            "depth": self.depth_selection,
            "time|key": self.time_key_selection,
            "time|depth": self.time_depth_selection,
            "depth|key": self.depth_key_selection}

    def execute(self, catalogue, origin_rules, magnitude_rules):
        """

        """
        return self.calculation_type[self.rule_type](catalogue,
                                                     origin_rules,
                                                     magnitude_rules)

    def time_selection(self, catalogue, origin_rules, magnitude_rules):
        """
        Define the choice of rule depending on the time
        """
        # Origin rules
        orig_rules = self._build_date_rule_list(origin_rules)
        mag_rules = self._build_date_rule_list(magnitude_rules)
        for event in catalogue.events:
            eq_date = event.origins[0].date
            for iloc, rule in enumerate(orig_rules):
                if eq_date >= rule[0][0] and eq_date <= rule[0][1]:
                    setattr(event, "origin_rule_idx", iloc)
            for iloc, rule in enumerate(mag_rules):
                if eq_date >= rule[0][0] and eq_date <= rule[0][1]:
                    setattr(event, "magnitude_rule_idx", iloc)
        return catalogue

    def key_selection(self, catalogue, origin_rules, magnitude_rules):
        """
        Define choice of rule depending on key
        """
        orig_set = [key for key, rule in origin_rules]
        mag_set = [key for key, rule in magnitude_rules]
        for event in catalogue.events:
            if event.description in orig_set:
                setattr(event, "origin_rule_idx",
                        orig_set.index(event.description))
            if event.description in mag_set:
                setattr(event, "magnitude_rule_idx",
                        mag_set.index(event.description))
        return catalogue

    def depth_selection(self, catalogue, origin_rules, magnitude_rules):
        """
        Defines choice of rule depending on depth
        """
        orig_rules = self._build_float_rule_list(origin_rules)
        mag_rules = self._build_float_rule_list(magnitude_rules)
        for event in catalogue.events:
            eq_depth = None
            for origin in event.origins:
                if not eq_depth and (origin.location.depth is not None):
                    # Take depth as the first origin depth found
                    eq_depth = origin.location.depth
            for iloc, rule in enumerate(orig_rules):
                if eq_depth >= rule[0][0] and eq_depth < rule[0][1]:
                    setattr(event, "origin_rule_idx", iloc)
            for iloc, rule in enumerate(mag_rules):
                if eq_depth >= rule[0][0] and eq_depth < rule[0][1]:
                    setattr(event, "magnitude_rule_idx", iloc)
        return catalogue

    def time_key_selection(self, catalogue, origin_rules, magnitude_rules):
        """
        Defines the choice of rule on the basis of the time and key selection
        """
        orig_rules = self._build_time_key_rule_list(origin_rules)
        mag_rules = self._build_time_key_rule_list(magnitude_rules)
        for event in catalogue.events:
            eq_date = event.origins[0].date
            for iloc, rule in enumerate(orig_rules):
                if eq_date >= rule[0][0] and eq_date <= rule[0][1] and\
                    event.description == rule[0][2]:
                    setattr(event, "origin_rule_idx", iloc)
            for iloc, rule in enumerate(mag_rules):
                if eq_date >= rule[0][0] and eq_date <= rule[0][1] and\
                    event.description == rule[0][2]:
                    setattr(event, "magnitude_rule_idx", iloc)
        return catalogue

    def depth_key_selection(self, catalogue, origin_rules, magnitude_rules):
        """
        Defines the choice of rule on the basis of the depth and key
        """
        orig_rules = self._build_float_key_rule_list(origin_rules)
        mag_rules = self._build_float_key_rule_list(magnitude_rules)
        for event in catalogue.events:
            eq_depth = None
            for origin in event.origins:
                if not depth and (origin.location.depth is not None):
                    # Take depth as the first origin depth found
                    eq_depth = origin.location.depth
            for iloc, rule in enumerate(orig_rules):
                if eq_depth >= rule[0][0] and eq_depth <= rule[0][1] and\
                    event.description == rule[0][2]:
                    setattr(event, "origin_rule_idx", iloc)
            for iloc, rule in enumerate(mag_rules):
                if eq_depth >= rule[0][0] and eq_depth <= rule[0][1] and\
                    event.description == rule[0][2]:
                    setattr(event, "magnitude_rule_idx", iloc)
        return catalogue

    def time_depth_selection(self, catalogue, origin_rules, magnitude_rules):
        """
        Defines the choice of rule on the basis of the time and depth
        """
        orig_rules = self._build_time_float_rule_list(origin_rules)
        mag_rules = self._build_time_float_rule_list(magnitude_rules)
        for event in catalogue.events:
            eq_date = event.origins[0].date
            eq_depth = None
            for origin in event.origins:
                if not depth and (origin.location.depth is not None):
                    # Take depth as the first origin depth found
                    eq_depth = origin.location.depth
            for iloc, rule in enumerate(orig_rules):
                if eq_date >= rule[0][0] and eq_date <= rule[0][1] and\
                    eq_depth >= rule[0][2] and eq_depth < rule[0][3]:
                    setattr(event, "origin_rule_idx", iloc)
            for iloc, rule in enumerate(mag_rules):
                if eq_date >= rule[0][0] and eq_date <= rule[0][1] and\
                    eq_depth >= rule[0][2] and eq_depth < rule[0][3]:
                    setattr(event, "magnitude_rule_idx", iloc)
        return catalogue

    def _build_time_key_rule_list(self, rule_set):
        """
        Parses the rule set from the string-defined rule for a set of
        times and keys
        """
        rule_list = []
        for rule in rule_set:
            date_rule, key_rule = rule[0].split("|")
            start_string, end_string = (date_rule.strip(" ")).split(" - ")
            start_date = date(*list(map(int, start_string.split("/"))))
            end_date = date(*list(map(int, end_string.split("/"))))
            rule_list.append(((start_date, end_date, key_rule.strip(" ")),
                              rule[1]))
        return rule_list

    def _build_float_key_rule_list(self, rule_set):
        """
        Parses the rule set from the string-defined rule for a set of
        floats and keys
        """
        rule_list = []
        for rule in rule_set:
            float_rule, key_rule = rule[0].split("|")
            lower, upper = list(map(float, float_rule.split(" - ")))
            rule_list.append(((lower, upper, key_rule.strip(" ")), rule[1]))
        return rule_list

    def _build_time_float_rule_list(self, rule_set):
        """

        """
        rule_list = []
        for rule in rule_set:
            date_rule, float_rule = rule[0].split("|")
            start_string, end_string = (date_rule.strip(" ")).split(" - ")
            start_date = date(*list(map(int, start_string.split("/"))))
            end_date = date(*list(map(int, end_string.split("/"))))
            lower, upper = list(map(float, float_rule.split(" - ")))
            rule_list.append(((start_date, end_date, lower, upper), rule[1]))
        return rule_list

    def _build_date_rule_list(self, rule_set):
        """

        """
        rule_list = []
        for rule in rule_set:
            start_string, end_string = rule[0].split(" - ")
            start_date = date(*list(map(int, start_string.split("/"))))
            end_date = date(*list(map(int, end_string.split("/"))))
            rule_list.append(((start_date, end_date), rule[1]))
        return rule_list

    def _build_float_rule_list(self, rule_set):
        """
        Builds the rule list tranforming a string into a pair of floats
        """
        rule_list = []
        for rule in rule_set:
            low_limit, high_limit = list(map(float, rule[0].split(" - ")))
            rule_list.append(((low_limit, high_limit), rule[1]))
        return rule_list


class DynamicHomogenisor(Homogenisor):
    """
    Alternative
    """
    def __init__(self, catalogue, logging=False):
        """

        """
        super(DynamicHomogenisor, self).__init__(catalogue)
        if logging:
            self.log = []
        else:
            self.log = None

    def _apply_origin_selection(self, event):
        '''
        Checks each agency to see if it is found in the event, returning the
        corresponding origin if so.
        '''

        if hasattr(event, "origin_rule_idx"):
            # Date is taken from the first origin
            eq_date = event.origins[0].date
            agencies = event.get_author_list()
            event_ori = self.orig_rules[event.origin_rule_idx][1]
            for iloc, author in enumerate(event_ori):
                #self.orig_rules[event.origin_rule_idx][1]):
                if author in agencies:
                    if self.log is not None:
                        self.log.append(
                            ["|".join([author, ";".join(event_ori)]), "NA"]
                            )
                    # Has a solution by the preferred agency
                    return event.origins[agencies.index(author)], author
        if self.log is not None:
            self.log.append(["NA", "NA"])
        return False, None

    def _apply_magnitude_selection(self, event):
        '''
        For the preferred origin, select the corresponding magnitude
        '''

        if hasattr(event, "magnitude_rule_idx"):
            # Date is taken from the first origin
            eq_date = event.origins[0].date
            mag_agencies, mag_scales, mag_values, mag_sigmas = \
                event.get_origin_mag_vals()
            # Render all scales to upper
            mag_scales = [mag.upper() for mag in mag_scales]
            event_mag = self.mag_rules[event.magnitude_rule_idx][1]
            for mag_rule in event_mag:
                #print mag_rule.scale, mag_rule.author
                for iloc in range(len(mag_agencies)):
                    if (mag_rule.author == mag_agencies[iloc]) and\
                        (mag_rule.scale.upper() == mag_scales[iloc]):
                        mag_value, mag_sigma = mag_rule.convert_value(
                            mag_values[iloc],
                            mag_sigmas[iloc])
                        if self.log is not None:
                            mag_log =  "|".join(
                                [str(mag_rule),
                                ";".join([str(rule) for rule in event_mag])]
                                )
                            self.log[-1][1] = mag_log

                        return mag_value, mag_sigma, "-".join([mag_rule.author,
                                                               mag_rule.scale])
        if self.log is not None:
            self.log.append(["NA", "NA"])
        return None, None, None

    def dump_log(self, filename):
        """
        Dumps the catalogue and the contents of the log into a csv file
        """
        if not isinstance(self.log, list) or (len(self.log) == 0):
            raise ValueError("Logging not selected!")
        fid = open(filename, "w")
        for iloc, event in enumerate(self.catalogue.events):
            if hasattr(event, "origin_rule_idx"):
                origin_rule_idx = str(event.origin_rule_idx)
            else:
                origin_rule_idx = ""
            if hasattr(event, "magnitude_rule_idx"):
                magnitude_rule_idx = str(event.magnitude_rule_idx)
            else:
                magnitude_rule_idx = ""
            if "," in event.description:
                descriptor = event.description.replace(",", ";")
            else:
                descriptor = event.description
            if event.preferred:
                print("%s" % ",".join([str(event.id), descriptor,
                    str(event.preferred.date), str(event.preferred.time),
                    str(event.preferred.record_key),
                    str(event.preferred.location.longitude),
                    str(event.preferred.location.latitude),
                    str(event.preferred.location.depth),
                    str(event.preferred.magnitude),
                    str(event.preferred.magnitude_sigma),
                    origin_rule_idx,
                    magnitude_rule_idx,
                    self.log[iloc][0],
                    self.log[iloc][1]]), file=fid)
            else:
                print("%s" % ",".join([str(event.id), descriptor,
                    origin_rule_idx,
                    magnitude_rule_idx,
                    self.log[iloc][0],
                    self.log[iloc][1]]), file=fid)
        fid.close()

#: Earth radius in km.
EARTH_RADIUS = 6371.0


def geodetic_distance_diff(origin1, origin2):
    """
    Calculate the geodetic distance between two points or two collections
    of points.

    Parameters are coordinates in decimal degrees. They could be scalar
    float numbers or numpy arrays, in which case they should "broadcast
    together".

    Implements http://williams.best.vwh.net/avform.htm#Dist

    :returns:
        Distance in km, floating point scalar or numpy array of such.
    """

    lons1 = origin1.location.longitude
    lats1 = origin1.location.latitude
    lons2 = origin2.location.longitude
    lats2 = origin2.location.latitude

    lons1, lats1, lons2, lats2 = _prepare_coords(lons1, lats1, lons2, lats2)
    distance = np.arcsin(np.sqrt(
        np.sin((lats1 - lats2) / 2.0) ** 2.0
        + np.cos(lats1) * np.cos(lats2)
        * np.sin((lons1 - lons2) / 2.0) ** 2.0
    ).clip(-1., 1.))
    return (2.0 * EARTH_RADIUS) * distance


#CF = pi / 180.

#def decimal_degree_diff(origin1, origin2):
#    '''
#    Returns the distance in decimal degrees between two origins
#    '''
#
#    lon1 = origin1.location.longitude * CF
#    lat1 = origin1.location.latitude * CF
#    lon2 = origin2.location.longitude * CF
#    lat2 = origin2.location.latitude * CF
#    dlat = lat2 - lat1
#    dlon = lon2 - lon1
#    aval = (sin(dlat / 2.) ** 2.) + (cos(lat1) * cos(lat2) *
#                                     (sin(dlon / 2.) ** 2.))
#    return atan2(aval, 1. - aval) * (180. / pi)

SECS_PER_YEAR = 365.25 * 24. * 3600.
BREAK_STR = "==============================================="

class DuplicateFinder(object):
    """
    Find duplicate events between two catalogues - adding the origins of
    the second catalogue into the first
    """
    def __init__(self, reference_catalogue, time_window, distance_window,
                 magnitude_window=None, logging=False):
        '''
        :param reference_catalogue:
            Catalogue in ISF Format
        :param float time_window:
            Time window in seconds
        :param float distance_window:
            Distance window in km
        '''
        self.reference = reference_catalogue
        self.time_window = time_window / SECS_PER_YEAR
        self.dist_window = distance_window
        self.mag_window = magnitude_window
        self.logging = logging
        self.merge_log = []

    def merge_catalogue(self, catalogue):
        '''
        Merge a second catalogue in ISFCatalogue format into the reference
        catalogue
        '''
        # Get event key list
        ref_keys = self.reference.get_event_key_list()
        ref_times = self.reference.get_decimal_dates()
        cat_keys = catalogue.get_event_key_list()
        cat_times = catalogue.get_decimal_dates()

        for iloc, event in enumerate(catalogue.events):
            # Check the time difference
            dtime = np.fabs(cat_times[iloc] - ref_times)
            idx = dtime < self.time_window
            if not np.any(idx):
                # No possible duplicates - add to end of event list
                self.reference.events.append(event)
                continue
            else:
                # Possible duplicates
                dup_event, has_dup = self.compare_duplicate_list(event,
                                                                 idx,
                                                                 dtime)
                if has_dup:
                    # Merge origins of new catalogue into origin of reference
                    self.reference.events[dup_event].merge_secondary_origin(
                        event.origins)
                else:
                    self.reference.events.append(event)
                    if self.logging:
                        self.merge_log.append([BREAK_STR])
                        self.merge_log.append(
                            ["Event %s not duplication" % str(event)])

        # Sort reference events
        print("After duplicate finding: %g events (%g)" %\
            (self.reference.get_number_events(), len(self.reference.events)))
        ref_times = self.reference.get_decimal_dates()
        ascend_time = np.argsort(ref_times)
        event_list = [self.reference.events[ascend_time[i]]
                      for i in range(0, self.reference.get_number_events())]
        self.reference.events = event_list
        return self.reference

    def tensor_check(self, event, dup_event):
        '''
        If duplicate event has a tensor - take tensor
        '''
        if not hasattr(event, 'tensor'):
            # No tensor required
            return

        if hasattr(self.reference.events[dup_event], 'tensor'):
            # If the reference event has a tensor then use reference catalogue
            # tensor
            return
        else:
            setattr(self.reference.events[dup_event], 'tensor', event.tensor)

    def compare_duplicate_list(self, event, idx, dtime):
        '''
        Determine if potential duplicates are actual duplicates
        '''
        idx = np.where(idx)[0]
        distance_valid = []
        for iloc in idx:
            # Check if event is within any distance window
            is_in_distance = False
            for origin1 in self.reference.events[iloc].origins:
                for origin2 in event.origins:
                    # compute distance in kms
                    distance = geodetic_distance_diff(origin1, origin2)
                    if distance < self.dist_window:
                        is_in_distance = True
            if is_in_distance:
                distance_valid.append(iloc)
        if len(distance_valid) > 1:
            # Multiple possible duplicates!
            # Assign to nearest event in time
            dtime = dtime[distance_valid]
            nrloc = np.argmin(dtime)
            locn = distance_valid[nrloc]
            if self.logging:
                ref_string = str(self.reference.events[locn]) + "-".join([
                    str(origin) for origin in self.reference.events[locn].origins]
                    )
                event_string = str(event) + "-".join([
                    str(origin) for origin in event.origins])
                self.merge_log.extend([BREAK_STR, ref_string, event_string])
            return locn, True
        elif len(distance_valid) == 1:
            # Single duplicate - add origins from event two to event 1
            locn = distance_valid[0]
            if self.logging:
                ref_string = str(self.reference.events[locn]) + "-".join([
                    str(origin) for origin in self.reference.events[locn].origins]
                    )
                event_string = str(event) + "-".join([
                    str(origin) for origin in event.origins])
                self.merge_log.extend([BREAK_STR, ref_string, event_string])
            return locn, True
        else:
            # Not duplicates
            return None, False
