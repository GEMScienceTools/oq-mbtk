

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
    #
    # get the name of the .ini file
    ini_fname = argv[0]
    #
    # read the content of the configuration file
    oqparam = readinput.get_oqparam(ini_fname)
    gmmlt = readinput.get_gsim_lt(oqparam)
    gmmlist = set(readinput.get_gsims(oqparam))
    #
    # print results
    print('\nGMPEs:')
    for tstr in gmmlist:
        print('  - ', tstr)
    ctx = ContextMaker(gmmlist)
    #
    # parameters
    print('\nRequired rupture-site distances')
    print('   ', getattr(ctx,'REQUIRES_DISTANCES'))
    print('Required site parameters')
    print('   ', getattr(ctx,'REQUIRES_SITES_PARAMETERS'))
    print('Required rupture parameters')
    print('   ', getattr(ctx,'REQUIRES_RUPTURE_PARAMETERS'))


if __name__== "__main__":
    main(sys.argv[1:])
