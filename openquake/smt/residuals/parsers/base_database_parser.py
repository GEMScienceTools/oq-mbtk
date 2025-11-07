#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation and G. Weatherill
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
Abstract base class for a strong motion database reader
"""
import os
import abc
import pandas as pd

from openquake.baselib.python3compat import with_metaclass


class SMDatabaseReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for strong motion database parser
    """

    def __init__(self, db_id, db_name, input_files):
        """
        Instantiate and conduct folder checks
        """
        self.id = db_id
        self.name = db_name
        self.database = None
        self.input_files = input_files # Can be a single file, or a directory
                                       # depending on the parser (examine the
                                       # one you wish to use to contextually
                                       # understand this input argument more)

    @abc.abstractmethod
    def parse(self):
        """
        Parses the database
        """


class SMTimeSeriesReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for a reader of a ground motion time series. Returns
    a dictionary containing basic time-history information for each component.
    """
    def __init__(self, input_files, units="cm/s/s"):
        """
        Instantiate and conduct folder checks
        """
        self.input_files = []
        for fname in input_files:
            if os.path.exists(fname):
                self.input_files.append(fname)
        self.time_step = None
        self.number_steps = None
        self.units = units
        self.metadata = None

    @abc.abstractmethod
    def parse_record(self):
        """
        Parse the strong motion record
        """


class SMSpectraReader(with_metaclass(abc.ABCMeta)):
    """
    Abstract Base Class for a reader of a ground motion spectra record
    """
    def __init__(self, input_files):
        """
        Intantiate with basic file checks
        """
        self.input_files = []
        for fname in input_files:
            if os.path.exists(fname):
               self.input_files.append(fname)

    @abc.abstractmethod
    def parse_spectra(self):
        """
        Parses the spectra
        """