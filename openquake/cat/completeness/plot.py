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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from openquake.baselib import sap
from openquake.cat.completeness.analysis import clean_completeness


def get_xy(ctab, ymax=2015, rndx=0.0, rndy=0.0):

    # Adding random value to y
    fct = np.random.rand(1)-0.5
    if rndx > 0:
        rndx = fct[0]*rndx
    if rndy > 0:
        rndy = fct[0]*rndy
    # Create x, y
    x = []
    y = []
    # Starting from the less recent year
    x.append(ymax+rndx)
    y.append(ctab[0, 1]+rndy)
    for i in range(0, ctab.shape[0]-1):
        # Flat
        x.append(ctab[i, 0]+rndx)
        y.append(ctab[i, 1]+rndy)
        # Step
        x.append(ctab[i, 0]+rndx)
        y.append(ctab[i+1, 1]+rndy)
    # Flat
    x.append(ctab[-1, 0]+rndx)
    y.append(ctab[-1, 1]+rndy)
    return np.array([(a, b) for a, b in zip(x, y)])


def plot_completeness(
    perms, mags, years, ymin=1900, ymax=2020, mmin=4.0, mmax=8.0, fname='',
    apriori={}, apriori_in={}):
    """
    Plots the set of completeness windows defined in by the `perms`, `mags`,
    `years` triple.
    """

    fig, ax = plt.subplots()
    lines = []
    colors = np.random.rand(len(perms), 3)
    labls = []
    for i, prm in enumerate(perms):
        idx = prm.astype(int)
        tmp = np.array([(y, m) for y, m in zip(years, mags[idx])])
        ctab = clean_completeness(tmp)
        coo = get_xy(ctab, rndx=0.0, rndy=0.1)
        lines.append(tuple([(x, y) for x, y in zip(coo[:, 0], coo[:, 1])]))
        labls.append(f'{i}')

    for key in apriori:
        mag = float(apriori[key])
        yea = float(key)
        plt.plot(yea, mag, 'o', color='red')

    for key in apriori_in:
        mag = float(apriori_in[key])
        yea = float(key)
        print(mag, yea)
        plt.plot(yea, mag, 'o', markersize=10., color='green')

    coll = matplotlib.collections.LineCollection(lines, colors=colors)
    ax.add_collection(coll)

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Magnitude [yr]')
    ax.grid(which='both')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(mmin, mmax)

    if len(fname):
        plt.savefig(fname)
    plt.show()


def main(folder, ymin=1900, ymax=2020, mmin=4.0, mmax=8.0):
    """
    Plots the set of completeness windows defined by the `perms`, `mags`,
    `years` triple files in a given folder.
    """

    perms = np.load(os.path.join(folder, 'dispositions.npy'))
    mags = np.load(os.path.join(folder, 'mags.npy'))
    years = np.load(os.path.join(folder, 'years.npy'))

    fname = 'completeness.png'
    plot_completeness(perms, mags, years, ymin, ymax, mmin, mmax, fname)


main.folder = 'Folder containing the completeness files'
main.ymin = 'Minimum year in the plot'
main.ymax = 'Maximum year in the plot'
main.mmin = 'Minimum year in the plot'
main.mmax = 'Maximum year in the plot'

if __name__ == '__main__':
    sap.run(main)
