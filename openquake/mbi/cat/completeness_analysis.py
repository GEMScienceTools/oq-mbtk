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

from openquake.baselib import sap
from openquake.cat.completeness.analysis import completeness_analysis


def main(fname_input_pattern, fname_config, folder_out_figs, folder_in,
         folder_out, *, skip: str = ''):
    """
    Analyses the completeness of a catalogue and saves the information about
    the best fitting GR in the configuration file.
    """

    completeness_analysis(fname_input_pattern, fname_config, folder_out_figs,
                          folder_in, folder_out, skip)


descr = 'Pattern to select input files with subcatalogues'
main.fname_input_pattern = descr
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the folder where to store figures'
main.folder_out_figs = msg
msg = 'Name of the folder where to read candidate completeness tables'
main.folder_in = msg
msg = 'Name of the folder where to store data'
main.folder_out = msg
msg = 'IDs of the sources to skip'
main.skip = msg


if __name__ == '__main__':
    sap.run(main)
