#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.mbt.tools.tr.classify import classify


def clssfy(ini_fname, compute_distances, rootf):
    """
    Classify an earthquake catalogue
    """
    classify(ini_fname, compute_distances, rootf)


MSG = 'Path to the configuration fname - typically a .ini file for tr'
clssfy.ini_fname = MSG
MSG = 'Flag indicating if distances must be computed'
clssfy.compute_distances = MSG
MSG = 'Root folder (paths are relative to this in the .ini file)'
clssfy.rootf = MSG

if __name__ == "__main__":
    sap.run(clssfy)
