#!/usr/bin/env python

import re
import os
import sys
import matplotlib.pyplot as plt
from openquake.baselib import sap
from openquake.sub.plotting.plot_cross_section import plt_cs


def pcs(cs_fname, out_folder=None):
    """
    plots cross sections based on lines in cs_fname of format

        lon lat length depth azimuth id <config>.ini

    """
    if out_folder == None:
        out_folder = './'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    fin = open(cs_fname, 'r')
    for line in fin:

        aa = re.split('\\s+', line.rstrip())
        olo = float(aa[0])
        ola = float(aa[1])
        depth = float(aa[2])
        lnght = float(aa[3])
        strike = float(aa[4])
        ids = aa[5]
        ini_fle = aa[6]

        fig = plt_cs(olo, ola, depth, lnght, strike, ids, ini_fle)
        name = 'section_%s.pdf' % (ids)
        path = os.path.join(out_folder, name)
        fig.savefig(path, bbox_inches='tight')
        print('Created %s' % (path))
        plt.close()

pcs.cs_fname = 'file with cross sections details'
pcs.out_folder = 'place to store the pdfs'

if __name__ == "__main__":
    sap.run(pcs)
