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

import sys

from openquake.commonlib import readinput
from openquake.hazardlib.gsim.base import ContextMaker


REQUIRES = ['DISTANCES', 'SITES_PARAMETERS', 'RUPTURE_PARAMETERS']


def main(argv):
    """
    This provides a short summary of the GSIM models used in a hazard model and
    the required parameters necessary to define ruptures, sites and
    rupture-site distances.
    """
    # Get the name of the .ini file
    ini_fname = argv[0]
    
    # Read the content of the configuration file
    oqparam = readinput.get_oqparam(ini_fname)
    
    # MN: 'gmmlt' assigned but never used
    gmmlt = readinput.get_gsim_lt(oqparam)
    gmmlist = set(readinput.get_gsims(oqparam))
    
    # Print results
    print('\nGMPEs:')
    for tstr in gmmlist:
        print('  - ', tstr)
    ctx = ContextMaker(gmmlist)
    
    # Parameters
    print('\nRequired rupture-site distances')
    print('   ', getattr(ctx, 'REQUIRES_DISTANCES'))
    print('Required site parameters')
    print('   ', getattr(ctx, 'REQUIRES_SITES_PARAMETERS'))
    print('Required rupture parameters')
    print('   ', getattr(ctx, 'REQUIRES_RUPTURE_PARAMETERS'))


if __name__ == "__main__":
    main(sys.argv[1:])
