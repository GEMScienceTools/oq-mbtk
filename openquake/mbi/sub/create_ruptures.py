#!/usr/bin/env python
# coding: utf-8

import re
import os
import toml
from openquake.baselib import sap
from openquake.sub.slab.rupture import calculate_ruptures


def main(config_fname: str, only_plt: bool = False):
    """
    Creates inslab ruptures
    """

    # Parsing config
    model = toml.load(config_fname)
    path = os.path.dirname(config_fname)

    for key in model['sources']:
        ini_fname = os.path.join(path, model['sources'][key]['ini_fname'])
        agr = model['sources'][key]['agr']
        bgr = model['sources'][key]['bgr']
        mmax = model['sources'][key]['mmax']
        mmin = model['mmin']
        ref_fdr = path
        calculate_ruptures(ini_fname, ref_fdr=ref_fdr, agr=agr, bgr=bgr,
                           mmin=mmin, mmax=mmax)

descr = 'The path to the .toml file containing info to build the ruptures'
main.config_fname = descr

if __name__ == '__main__':
    sap.run(main)
