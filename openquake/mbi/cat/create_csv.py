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
import shutil
import pandas as pd
from pathlib import Path
from openquake.baselib import sap
from openquake.cat.hmg.utils import to_hmtk_catalogue


def create_folder(folder: str, clean: bool = False):
    """
    Create a folder. If the folder exists, it's possible to
    clean it.

    :param folder:
        The name of the folder tp be created
    :param clean:
        When true the function removes the content of the folder
    """
    if os.path.exists(folder):
        if clean:
            shutil.rmtree(folder)
    else:
        Path(folder).mkdir(parents=True, exist_ok=True)


def main(cat_fname, fname_out):

    # Read catalogue
    df = pd.read_hdf(cat_fname)

    # Create folder
    create_folder(os.path.dirname(fname_out))

    # Save file
    df.to_csv(fname_out, index=False)

    # Create hmtk file
    odir = os.path.dirname(fname_out)
    ofle = os.path.basename(fname_out)
    tmps = ofle.split('.')
    ofle = f'{tmps[0]}_hmtk.csv'
    odf = to_hmtk_catalogue(df)
    odf.to_csv(os.path.join(odir, ofle), index=False)


main.cat_fname = 'Name of the .hdf5 file containing the homogenized catalogue'
main.cat_fname = 'Name of output .csv that will be created'

if __name__ == "__main__":
    """
    The function creates the .csv file with the events in the homogenised
    catalog
    """
    sap.run(main)
