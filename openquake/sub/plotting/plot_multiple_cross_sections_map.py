#!/usr/bin/env python

import re
import numpy
import configparser
import pickle

from openquake.baselib import sap
from openquake.sub.cross_sections import CrossSection, Trench


def read_cs(filename):
    """
    """
    cs_dict = {}
    fin = open(filename, 'r')
    for line in fin:
        aa = re.split('\\s+', line)
        olo = float(aa[0])
        ola = float(aa[1])
        length = float(aa[3])
        strike = float(aa[4])
        key = aa[5]
        cs = CrossSection(olo, ola, [length], [strike])
        cs_dict[key] = cs
    return cs_dict


def plot(config_file, cs_file=None):
    """
    """

    # Parse .ini file
    config = configparser.ConfigParser()
    config.read(config_file)

    if cs_file is None:

        fname_trench = config['data']['trench_axis_filename']

        # Load trench axis
        fin = open(fname_trench, 'r')
        lotmp = []
        latmp = []
        for line in fin:
            aa = re.split('\\s+', re.sub('^\\s+', '', line))
            lotmp.append(float(aa[0]))
            latmp.append(float(aa[1]))
        fin.close()
        if (min(lotmp) / max(lotmp) < 0.) & (min(lotmp) < -150.):
            lotmp = (x + 360. if x < 0. else x for x in lotmp)
        trench = list(zip(lotmp, latmp))
        trench = Trench(numpy.array(trench))

    else:

        # Read cs file
        cs_dict = read_cs(cs_file)
        lotmp = [cs_dict[c].olo for c in cs_dict]
        latmp = [cs_dict[c].ola for c in cs_dict]
        trench = list(zip(lotmp, latmp))
        trench = Trench(numpy.array(trench))


plot.config_file = 'config file to datasets'
plot.cs_file = 'existing cross sections details'

if __name__ == "__main__":
    sap.run(plot)
