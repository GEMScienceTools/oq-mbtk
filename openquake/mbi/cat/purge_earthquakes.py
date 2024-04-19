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
from openquake.cat.hmg.purge import purge


def main(fname_cat, fname_cat_out, fname_csv):
    """
    :param fname_cat:
       Name of the .h5 file with the homogenised catalogue
    :param fname_cat_out:
       Name of the .h5 file where the new catalogue will be stored
       once duplicates are removed
    :param fname_csv:
        name of the csv file with one column "eventID" that lists
        duplicates to be removed from the catalogue 
    """

    purge(fname_cat, fname_cat_out, fname_csv) 


main.fname_cat = '.h5 file with origins'
main.fname_cat_out = '.h5 file with origins excluded'
main.fname_csv = '.csv file with the list of events ID to purge'

if __name__ == "__main__":
    """
    This removes from the catalogue the events indicated in the the
    `fname_csv` file.
    """
    sap.run(main)
