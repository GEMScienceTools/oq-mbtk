#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) 2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.


import subprocess
from openquake.cat.hmg.plot import get_agencies


def write_gmt_file(df, agencies=[], fname_gmt='/tmp/tmp.txt'):
    """
    :param df:
        A dataframe
    :param agencies:
         A list with the names of the  agencies
    :param fname_gmt:
        The name of the output file
    """

    if len(agencies) < 1:
        agencies = get_agencies(df)

    condtxt = ''
    for i, key in enumerate(agencies):
        if i > 0:
            condtxt += ' | '
        condtxt += "(df['magAgency'] == '{:s}')".format(key)
    cond = eval(condtxt)

    rows = df.loc[cond, :]
    rows.to_csv(fname_gmt, sep=" ", columns=['longitude', 'latitude', 'magMw'],
                index=False, header=False)


def plot_catalogue(fname, fname_fig='/tmp/tmp', **kwargs):
    """
    :param fname:
        Name of the file with the catalogue
    :param kwargs:
        Optional parameters:
            - 'extent' - The extent of the plot in the gmt format (i.e.
                         minlo/maxlo/minla/maxla0
            - 'fformat' - The format of the plot created
    """

    cmds = []

    # ARG1=${1:-}
    # ARG2=${2:-R-60/360/-90/90}
    # ARG3=${3:-/tmp/fig}
    # ARG4=${4:-3c}
    # TITLE=${5:-"Catalogue"}
    # EXTENT=-R$ARG2
    # PRO=-Jm$ARG4

    if "extent" in kwargs:
        extent = "-R"+kwargs["extent"]
    else:
        cmds.append("EXTENT=$(gmt info {:s} -I2 -D1)".format(fname))

    cmds.append("gmt set MAP_FRAME_TYPE = PLAIN")
    cmds.append("gmt set MAP_GRID_CROSS_SIZE_PRIMARY = 0.2i")
    cmds.append("gmt set MAP_FRAME_TYPE = PLAIN")
    cmds.append("gmt set FONT_TITLE = 8p")
    cmds.append("gmt set FONT_LABEL = 6p")
    cmds.append("gmt set FONT_ANNOT_PRIMARY = 6p")

    if "fformat" in kwargs:
        fmt = "gmt set GMT_GRAPHICS_FORMAT = {:s}"
        cmds.append(fmt.format(kwargs["fformat"]))
        fformat = kwargs["fformat"].lower()
    else:
        fformat = "pdf"

    cmds.append("gmt begin {:s}".format(fname_fig))

    fmt = "gmt coast {:s} {:s} -Bp5 -N1"
    cmds.append(fmt.format(extent, "-JM10"))
    cmds.append("gmt coast -Ccyan -Scyan")
    cmds.append("gmt plot {:s} -Sc0.1 -W0.5,red -Gpink -t50".format(fname))
    cmds.append("gmt coast -N1 -t50 -B+t{:s}".format("test"))

    # Running
    cmds.append("gmt end")
    for cmd in cmds:
        _ = subprocess.call(cmd, shell=True)

    tmps = "{:s}.{:s}".format(fname_fig, fformat)
    return tmps
