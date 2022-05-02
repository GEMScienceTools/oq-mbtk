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

import subprocess
import pandas as pd
from openquake.cat.hmg.plot import get_agencies


def write_gmt_file(df, agencies=[], fname_gmt='/tmp/tmp.txt', **kwargs):
    """
    :param df:
        A dataframe
    :param agencies:
         A list with the names of the  agencies
    :param fname_gmt:
        The name of the output file
    """

    if "mmin" in kwargs:
        df = df[df["value"] > float(kwargs["mmin"])]

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


def plot_catalogue(fname, fname_fig='/tmp/tmp.pdf', **kwargs):
    """
    :param fname:
        Name of the file with the catalogue
    :param kwargs:
        Optional parameters:
            - 'extent' - The extent of the plot as a list with
                         [minlo, maxlo, minla, maxla]
    """

    try:
        import pygmt
    except:
        print("pygmt is not available")
        return

    lonote = 5
    if "extent" in kwargs:
        extent = "-R"+kwargs["extent"]
    else:
        df = pd.read_csv(fname, sep="\\s+", names=['lon', 'lat', 'mag'])
        factor = 0.02

        # Longitude
        lomn = df.lon.min()
        lomx = df.lon.max()
        lodelta = (lomx - lomn)
        lomn -= lodelta * factor
        lomx += lodelta * factor
        if lodelta < 1:
            lonote = 0.5
        elif lodelta < 5:
            lonote = 2

        # Latitude
        lamn = df.lat.min()
        lamx = df.lat.max()
        ladelta = (lamx - lamn)
        lamn -= ladelta * factor
        lamx += ladelta * factor
        region = [lomn, lomx, lamn, lamx]

    # Get filename
    tmps = fname_fig

    # Plotting
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="wheat", water="skyblue")
    fig.plot(x=df.lon, y=df.lat, size=0.005*2**df.mag, color="white",
             pen="black", style="cc")
    fig.savefig(fname_fig)


def plot_catalogue_old(fname, fname_fig='/tmp/tmp.txt', **kwargs):
    """
    :param fname:
        Name of the file with the catalogue
    :param kwargs:
        Optional parameters:
            - 'extent' - The extent of the plot as a list with
                         [minlo, maxlo, minla, maxla]
    """

    cmds = []

    lonote = 5
    if "extent" in kwargs:
        extent = "-R"+kwargs["extent"]
    else:
        df = pd.read_csv(fname, sep="\\s+")
        factor = 0.2

        lomn = df.iloc[:, 0].min()
        lomx = df.iloc[:, 0].max()
        lodelta = (lomx - lomn)
        lomn -= lodelta * factor
        lomx += lodelta * factor
        if lodelta < 1:
            lonote = 0.5
        elif lodelta < 5:
            lonote = 2

        lamn = df.iloc[:, 1].min()
        lamx = df.iloc[:, 1].max()
        ladelta = (lamx - lamn)
        lamn -= ladelta * factor
        lamx += ladelta * factor
        extent = "-R{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(lomn, lomx, lamn, lamx)

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

    fmt = "gmt coast {:s} {:s} -Bp{:f} -N1"
    cmds.append(fmt.format(extent, "-JM10", lonote))
    cmds.append("gmt coast -Ccyan -Scyan")

    tmp = "gawk '{{print $1, $2, 2.4**$3/800}}' {:s}".format(fname)
    tmp += " | gmt plot -Sc0.1 -W0.5,red -Gpink -t50"

    cmds.append("gmt plot {:s} -Sc0.1 -W0.5,red -Gpink -t50".format(fname))
    cmds.append("gmt coast -N1 -t50 -B+t{:s}".format("Catalogue"))

    # Running
    cmds.append("gmt end")
    for cmd in cmds:
        print(cmd)
        _ = subprocess.call(cmd, shell=True)

    tmps = "{:s}.{:s}".format(fname_fig, fformat)
    return tmps
