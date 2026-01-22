# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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

import numpy
from prettytable import PrettyTable


def print_trt_stats_table(model):
    stypes, msrtypes = get_trt_stats(model)
    for trt in stypes:
        print('Tectonic region: {0:s}'.format(trt))
        x = PrettyTable()
        x.add_column('Source type', list(stypes[trt].keys()))
        nums = [stypes[trt][key] for key in stypes[trt].keys()]
        x.add_column('Number of sources', nums)
        print(x)
        # MSR
        s = PrettyTable()
        s.add_column('MSR', list(msrtypes[trt].keys()))
        nums = [msrtypes[trt][key] for key in msrtypes[trt].keys()]
        s.add_column('Number of sources', nums)
        print(s)


def get_trt_stats(model):
    """
    Provide statistics about the sources included in the tectonic regions
    composing the model

    :parameter model:
        A list
    """
    stypes = {}
    msrtypes = {}
    for src in model:

        # Getting parameters
        trt = src.tectonic_region_type
        sty = type(src).__name__
        msr = type(src.magnitude_scaling_relationship).__name__
        
        # Source types
        if trt in stypes:
            if sty in stypes[trt]:
                stypes[trt][sty] += 1
            else:
                stypes[trt][sty] = 1
        else:
            stypes[trt] = {}
            stypes[trt][sty] = 1
        
        # MSR types
        if trt in msrtypes:
            if msr in msrtypes[trt]:
                msrtypes[trt][msr] += 1
            else:
                msrtypes[trt][msr] = 1
        else:
            msrtypes[trt] = {}
            msrtypes[trt][msr] = 1

    return stypes, msrtypes


def get_discrete_mfds(model):
    """
    Get discrete MFDs

    :parameter model:
        A list of hazardlib source instances
    :returns:
        A list of tuples where each tuple is a MFD
    """
    mfds = []
    for src in model:
        out = src.mfd.get_annual_occurrence_rates()
        mfds.append(numpy.array(out))
    return mfds
