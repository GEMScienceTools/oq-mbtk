#!/usr/bin/env python

import re
import sys
import numpy
import configparser
import pickle
from openquake.sub.cross_sections import CrossSection, Trench
from openquake.sub.create_multiple_cross_sections import plot as pcs


def read_cs(filename):
    """
    """
    cs_dict = {}
    fin = open(filename, 'r')
    for line in fin:
        aa = re.split('\s+', line)
        olo = float(aa[0])
        ola = float(aa[1])
        length = float(aa[3])
        strike = float(aa[4])
        key = aa[5]
        cs = CrossSection(olo, ola, [length], [strike])
        cs_dict[key] = cs
    return cs_dict


def main(argv):
    """
    """

    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(argv[0])
    fname_trench = config['data']['trench_axis_filename']
    fname_eqk_cat = config['data']['catalogue_pickle_filename']
    interdistance = float(config['section']['interdistance'])

    # Load trench axis
    fin = open(fname_trench, 'r')
    lotmp = []; latmp = []
    for line in fin:
        aa = re.split('\s+', re.sub('^\s+', '', line))
        lotmp.append(float(aa[0]))
        latmp.append(float(aa[1]))
    fin.close()
    qual = 0
    if (min(lotmp)/max(lotmp)<0.) & (min(lotmp)<-150.):
        qual = 1
        lotmp = (x+360. if x<0. else x for x in lotmp)
    trench = list(zip(lotmp,latmp))
    trench = Trench(numpy.array(trench))

    # Load catalogue
    cat = pickle.load(open(fname_eqk_cat, 'rb'))

    # Read cs file
    cs_dict = read_cs(argv[1])

    # Plotting
    pcs(trench, cat, cs_dict, interdistance)


if __name__ == "__main__":
    main(sys.argv[1:])
