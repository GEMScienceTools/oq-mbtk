#!/usr/bin/env python3

import os
import re
import pandas as pd

from shutil import copyfile
from openquake.baselib import sap


def main(fname_cat, fname_cat_out, fname_csv):
    # :param cat_fname:
    #   Name of the .h5 file with the homogenised catalogue

    if not os.path.exists(fname_cat+'.bak'):
        copyfile(fname_cat, fname_cat+'.bak')
    else:
        raise ValueError("Backup file already exists")

    if not os.path.exists(fname_cat_out+'.bak'):
        copyfile(fname_cat_out, fname_cat_out+'.bak')
    else:
        raise ValueError("Backup file already exists")

    #
    # Read catalogue
    cat = pd.read_hdf(fname_cat)
    cat_out = pd.read_hdf(fname_cat_out)
    print('The catalogue contains {:d} earthquakes'.format(len(cat)))

    #
    # Read file with the list of IDs
    fle = open(fname_csv, mode='r')
    keys = []
    for line in fle:
        aa = line.rstrip().split(',')
        keys.append([re.sub(' ', '', aa[0]), re.sub(' ', '', aa[1])])
    fle.close()

    #
    # Move events
    for key in keys:
        series = cat[cat[key[0]] == key[1]]
        cat_out.append(series)
    cat_out.to_hdf(fname_cat_out, '/events', append=False)

    #
    # Drop events
    for key in keys:
        cat.drop(cat[cat[key[0]] == key[1]].index, inplace=True)
    print('The catalogue contains {:d} earthquakes'.format(len(cat)))
    cat.to_hdf(fname_cat, '/events', append=False)


main.fname_cat = '.h5 file with origins'
main.fname_cat_out = '.h5 file with origins excluded'
main.fname_csv = '.csv file with the list of events ID to purge'

if __name__ == "__main__":
    """
    This removes from the catalogue the events indicated in the the
    `fname_csv` file.
    """
    sap.run(main)
