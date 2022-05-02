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
from openquake.baselib import sap
from openquake.cat.hmg.hmg import process_dfs


def main(settings, odf_fname, mdf_fname, outfolder='./h5/'):
    """
    The hom function reads the information in the settings file and creates
    a homogenised catalogue following the rules there defined.
    """

    # Homogenise
    save, work = process_dfs(odf_fname, mdf_fname, settings)

    # Outname
    fname = os.path.basename(odf_fname).split('_')[0]

    # Saving results
    fmt = '{:s}_catalogue_homogenised.h5'
    tmp = os.path.join(outfolder, fmt.format(fname))
    save.to_hdf(tmp, '/events', append=False)

    fmt = '{:s}_leftout.h5'
    tmp = os.path.join(outfolder, fmt.format(fname))
    work.to_hdf(tmp, '/events', append=False)


main.settings = '.toml file with the settings'
main.odf_fname = '.h5 file with origins'
main.mdf_fname = '.h5 file with magnitudes'
main.outfolder = 'output folder'

if __name__ == "__main__":
    sap.run(main)
