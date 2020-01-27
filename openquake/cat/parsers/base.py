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

import abc
import os


def with_metaclass(meta, *bases):
    """
    Returns an instance of meta inheriting from the given bases.
    To be used to replace the __metaclass__ syntax.
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(mcl, name, this_bases, d):
            if this_bases is None:
                return type.__new__(mcl, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})


def _to_int(string):
    """
    Converts a string to an integer, returning none if empty
    """
    string = string.strip(' ')
    if string:
        return int(string)
    else:
        return None

def _to_float(string):
    """
    Converts a string to a float, returning none if empty
    """
    string = string.strip(' ')
    if string:
        return float(string)
    else:
        return None

def _to_str(string):
    """
    Returns the string stripped of whitespace
    """
    return string.strip(' ')


class BaseCatalogueDatabaseReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for reading an earthquake database file
    """
    def __init__(self, filename, selected_origin_agencies=[],
            selected_magnitude_agencies=[]):
        """
        Instantiate the reader
        :param str filename:
            Path to catalogue file
        :param list selected_origin_agencies:
            List of origin agencies to be considered for inclusion 
        :param list selected_magnitude_agencies:
            List of magnitude agencies to be considered for inclusion 
        """
        if not os.path.exists(filename):
            raise IOError("File %s does not exist!" % filename)
        self.filename = filename
        self.catalogue = None
        self.selected_origin_agencies = selected_origin_agencies
        self.selected_magnitude_agencies = selected_magnitude_agencies

    @abc.abstractmethod
    def read_file(self, identifier, name):
        """
        Reads the catalogue from the file and assigning the identifier and name
        """

