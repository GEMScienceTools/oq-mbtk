#!/usr/bin/env python

import re
import sys
import numpy
import pickle
import configparser
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from openquake.sub.cross_sections import Trench
from openquake.baselib import sap

def get_cs(trench, ini_filename, cs_len, cs_depth, interdistance, qual,
           fname_out_cs='cs_traces.cs'):
    """
    :parameter trench:
        An instance of the :class:`Trench` class
    :parameter ini_filename:
        The name of the .ini file
    :parameter cs_len:
        Length of the cross-section [km]
    :parameter interdistance:
        Separation distance between cross-sections [km]
    :parameter qual:
        Boolean when true fixes longitudes in proximity of the IDL
    :parameter fname_out_cs:
        Name of the file where we write the traces of the cross sections
    """

    # Plot the traces of cross-sections
    fou = open(fname_out_cs, 'w')

    cs_dict = {}
    for idx, (cs, out) in enumerate(
            trench.iterate_cross_sections(interdistance, cs_len)):

        if cs is not None:
            cs_dict['%s' % idx] = cs

            if qual == 1:
                cs.plo[:] = ([x-360. if x > 180. else x for x in cs.plo[:]])

            # Set the length
            tmp_len = numpy.min([cs_len, out]) if out is not None else cs_len
            tmps = '%f %f %f %f %f %d %s\n' % (cs.plo[0],
                                               cs.pla[0],
                                               cs_depth,
                                               tmp_len,
                                               cs.strike[0],
                                               idx,
                                               ini_filename)
            print(tmps.rstrip())
            fou.write(tmps)
    fou.close()

    return cs_dict


def main(config_fname):
    """
    config_fname is the .ini file
    """

    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(config_fname)
    fname_trench = config['data']['trench_axis_filename']
    fname_eqk_cat = config['data']['catalogue_pickle_filename']
    cs_length = float(config['section']['lenght'])
    cs_depth = float(config['section']['dep_max'])
    interdistance = float(config['section']['interdistance'])

    # Load trench axis
    fin = open(fname_trench, 'r')
    lotmp = []
    latmp = []
    for line in fin:
        aa = re.split('\\s+', re.sub('^\\s+', '', line))
        lotmp.append(float(aa[0]))
        latmp.append(float(aa[1]))
    fin.close()
    qual = 0
    if (min(lotmp)/max(lotmp) < 0.) & (min(lotmp) < -150.):
        qual = 1
        lotmp = (x+360. if x < 0. else x for x in lotmp)
    trench = list(zip(lotmp, latmp))
    trench = Trench(numpy.array(trench))

    # Load catalogue
    with open(fname_eqk_cat, 'rb') as fin:
        cat = pickle.load(fin)

    # Get cross-sections
    cs_dict = get_cs(trench, argv[0], cs_length, cs_depth, interdistance, qual)

    # Plotting
    if False:
        plot(trench, cat, cs_dict, interdistance)

main.config = 'config file for creating cross sections from trench axis'

if __name__ == "__main__":
    sap.run(main)
