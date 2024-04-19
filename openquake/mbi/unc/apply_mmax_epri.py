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
import toml
import numpy as np
import pandas as pd
from pathlib import Path
from openquake.baselib import sap
from openquake.wkf.seismicity.mmax_epri import (
    get_mmax_pmf, get_composite_likelihood, get_xml)


def _main(fname_conf: str, sid: str, fname_cat: str, pri_mean: float,
          pri_std: float, folder_out: str, method: str = "weichert",
          fig_fname: str = None, bsid: str = 'bs0', ):

    # Parse config file
    model = toml.load(fname_conf)

    # Get info for the selected source
    info = model["sources"][sid]
    bgr = info[f"bgr_{method:s}"]
    mmaxobs = info["mmax_obs"]
    ccomp = info["completeness_table"]
    ccomp = [[float(c[0]), float(c[1])] for c in ccomp]

    # Read catalogue as a dataframe
    dfc = pd.read_csv(fname_cat)

    # Get likelihood
    mupp, lkl = get_composite_likelihood(dfc, ccomp, bgr)

    # Create folder for output xml if needed
    if not os.path.exists(folder_out):
        Path(folder_out).mkdir(parents=True, exist_ok=True)

    # Create folder for figure if needed
    if fig_fname is not None:
        ffold = os.path.dirname(fig_fname)
        if not os.path.exists(ffold):
            Path(ffold).mkdir(parents=True, exist_ok=True)

    # Get PMF for mmax
    wdt = 0.5
    mag0 = np.ceil(np.min(dfc.magnitude)/0.1)*0.1
    bins = np.arange(mag0, np.ceil(mmaxobs/wdt)*wdt+3, wdt)
    weis, mags = get_mmax_pmf(pri_mean, pri_std, bins, mupp=mupp,
                              likelihood=lkl, fig_name=fig_fname,
                              sid=sid)

    # Get XML
    xmlstr = get_xml(mags, weis, sid, bsid)

    # Write XML
    fname = os.path.join(folder_out, f"ssclt_{sid}.xml")
    with open(fname, "a", encoding="utf8") as fou:
        fou.write(xmlstr)


def main(fname_conf: str, sid: str,  fname_cat: str, pri_mean: float,
         pri_std: float, folder_out: str, *, method: str = 'weichert',
         fig_fname: str = None):
    """ Create xml describing epistemic uncertainty on Mmax """
    _main(fname_conf, sid, fname_cat, pri_mean, pri_std, folder_out, method,
          fig_fname)


MSG = "Name of configuration file"
main.fname_conf = MSG
MSG = "Source ID"
main.sid = MSG
MSG = "Mean magnitude of the prior"
main.pri_mean = MSG
MSG = "Std of the prior"
main.pri_std = MSG
MSG = "Folder where to store the xml"
main.folder_out = MSG
MSG = "Method for computing bGR"
main.method = MSG
MSG = "Name of the figure"
main.fig_fname = MSG

if __name__ == '__main__':
    sap.run(main)
