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


'''
Parser for generic catalogue into either ISFCatalogue format or
GCMT catalogue format
'''
import csv
import pandas as pd
import numpy as np
from openquake.cat.parsers.generic_catalogue import GeneralCsvCatalogue
from openquake.cat.isf_catalogue import (ISFCatalogue, Magnitude,
                                         Origin, Location, Event)
from openquake.cat.parsers.gcmt_ndk_parser import ParseNDKtoGCMT


def _header_check(input_keys, catalogue_keys):
    valid_key_list = []
    for element in input_keys:
        if element in catalogue_keys:
            valid_key_list.append(element)
        else:
            print('Catalogue Attribute %s is not a recognised '
                  'catalogue key' % element)
    return valid_key_list


def _float_check(attribute_array, value):
    """
    Checks if value is valid float, appends to array if valid, appends
    nan if not
    """
    try:
        attribute_array = np.hstack([attribute_array, float(value)])
    except Exception:
        value = value.strip(' ')
        attribute_array = np.hstack([attribute_array, np.nan])
    return attribute_array


def _int_check(attribute_array, value):
    """
    Checks if value is valid integer, appends to array if valid, appends
    nan if not
    """
    try:
        attribute_array = np.hstack([attribute_array, int(value)])
    except Exception:
        value = value.strip(' ')
        attribute_array = np.hstack([attribute_array, np.nan])
    return attribute_array


class GenericCataloguetoISFParser(object):
    '''
    Reads the generic csv catalogue file to return an instance of the
    ISFCatalogue class
    '''
    def __init__(self, filename):
        '''
        '''
        self.filename = filename
        self.catalogue = GeneralCsvCatalogue()

    def parse(self, cat_id, cat_name):
        """
        Parses the file
        """
        self.catalogue.parse_csv(self.filename)
        output_cat = self.export(cat_id, cat_name)

        # Check that the length of the datasets is homogenous
        nels = []
        for k in self.catalogue.data.keys():
            tmplen = len(self.catalogue.data[k])
            if tmplen > 0:
                nels.append(tmplen)
        assert len(list(set(nels))) == 1

        # Printing info

        return output_cat

    def parse_old(self, cat_id, cat_name):
        '''
        Opens the raw file parses the catalogue then exports
        '''
        filedata = open(self.filename, 'rU')
        # Reading the data file
        data = csv.DictReader(filedata)
        # Parsing the data content
        for irow, row in enumerate(data):
            if irow == 0:
                tmp = self.catalogue.TOTAL_ATTRIBUTE_LIST
                valid_key_list = _header_check(list(row.keys()), tmp)
            for key in valid_key_list:
                if key in self.catalogue.FLOAT_ATTRIBUTE_LIST:
                    self.catalogue.data[key] = _float_check(
                        self.catalogue.data[key],
                        row[key])
                elif key in self.catalogue.INT_ATTRIBUTE_LIST:
                    self.catalogue.data[key] = _int_check(
                        self.catalogue.data[key],
                        row[key])
                else:
                    self.catalogue.data[key].append(row[key])
        output_cat = self.export(cat_id, cat_name)
        return output_cat

    def export(self, cat_id=None, cat_name=None):
        """
        Exports the catalogue to ISF Format
        """
        return self.catalogue.write_to_isf_catalogue(cat_id, cat_name)


class GenericCataloguetoGCMT(GenericCataloguetoISFParser):
    '''
    Reads the generic csv catalogue file to return an instance of the
    GCMT class class
    '''
    def export(self, cat_id=None, cat_name=None):
        """
        Exports the catalogue to GCMT format
        """
        return self.write_to_gcmt_class()


class GCMTtoISFParser(object):
    '''
    Read in a file in GCMT NDK format and parse to ISF Catalogue
    '''
    def __init__(self, gcmt_file=None):
        '''
        '''
        if gcmt_file:
            self.filename = gcmt_file
            parser = ParseNDKtoGCMT(self.filename)
            self.catalogue = parser.read_file()
        else:
            self.filename = None
            self.catalogue = None

    @classmethod
    def from_catalogue(cls, catalogue, cat_id, cat_name):
        """
        If a different parser has been used just instantiate the class
        """
        self = cls()
        self.catalogue = catalogue
        if not len(self.catalogue):
            print("No events in catalogue - returning None")
            return None
        return self.parse(cat_id, cat_name)

    def parse(self, cat_id="GCMT", cat_name="GCMT"):
        '''
        Returns the catalogue as an instance of an ISFCatalogue
        An ISF catalogue will have two origins: The hypocentre solution and
        the centroid
        '''
        isf_cat = ISFCatalogue(cat_id, cat_name)
        base_id = cat_id + '_'
        counter = 1
        for gcmt in self.catalogue.gcmts:
            # Get IDs
            event_id = base_id + ("%06d" % counter)
            counter += 1
            origin_id = gcmt.identifier.strip(' ')
            # Two origins - 1 hypocentre (mb, Ms, Mw), 2 - centroid (Mw)
            origin_mags = []
            if gcmt.hypocentre.m_b:
                origin_mags.append(Magnitude(event_id,
                                             origin_id,
                                             gcmt.hypocentre.m_b,
                                             gcmt.hypocentre.source,
                                             scale='mb'))
            if gcmt.hypocentre.m_s:
                origin_mags.append(Magnitude(event_id,
                                             origin_id,
                                             gcmt.hypocentre.m_s,
                                             gcmt.hypocentre.source,
                                             scale='Ms'))
            m_w = Magnitude(event_id,
                            origin_id + "-C",
                            gcmt.magnitude,
                            cat_id,
                            scale='Mw')
            # Get locations
            hypo_loc = Location(origin_id,
                                gcmt.hypocentre.longitude,
                                gcmt.hypocentre.latitude,
                                gcmt.hypocentre.depth)
            centroid_loc = Location(origin_id + "-C",
                                    gcmt.centroid.longitude,
                                    gcmt.centroid.latitude,
                                    gcmt.centroid.depth,
                                    depth_error=gcmt.centroid.depth_error)
            # Get origins
            hypo = Origin(origin_id,
                          gcmt.hypocentre.date,
                          gcmt.hypocentre.time,
                          hypo_loc,
                          gcmt.hypocentre.source,
                          is_prime=True)
            hypo.magnitudes = origin_mags
            # Get centroids
            centroid = Origin(origin_id + "-C",
                              gcmt.centroid.date,
                              gcmt.centroid.time,
                              centroid_loc,
                              cat_id,
                              is_centroid=True,
                              time_error=gcmt.centroid.time_error)
            centroid.magnitudes = [m_w]
            event = Event(event_id, [hypo, centroid],
                          hypo.magnitudes + centroid.magnitudes,
                          gcmt.hypocentre.location)
            setattr(event, 'tensor', gcmt.moment_tensor)
            isf_cat.events.append(event)
        return isf_cat
