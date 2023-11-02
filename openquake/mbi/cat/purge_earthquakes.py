#!/usr/bin/env python
# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

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
    print('The catalogue contains {:d} earthquakes'.format(len(cat)))

    #
    # Read file with the list of IDs
    event_df = pd.read_csv(fname_csv)

    #
    # Drop events
    for key in event_df['eventID'].values:
        cat.drop(cat[cat['eventID'] == key].index, inplace=True)
    print('The catalogue contains {:d} earthquakes'.format(len(cat)))
    cat.to_hdf(fname_cat_out, '/events', append=False)


main.fname_cat = '.h5 file with origins'
main.fname_cat_out = '.h5 file with origins excluded'
main.fname_csv = '.csv file with the list of events ID to purge'

if __name__ == "__main__":
    """
    This removes from the catalogue the events indicated in the the
    `fname_csv` file.
    """
    sap.run(main)
