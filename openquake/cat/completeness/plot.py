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
from completeness_analysis import clean_completeness


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


def main(folder, ymin=1900, ymax=2020, mmin=4.0, mmax=8.0):

    perms = np.load(os.path.join(folder, 'dispositions.npy'))
    mags = np.load(os.path.join(folder, 'mags.npy'))
    years = np.load(os.path.join(folder, 'years.npy'))

    fig, ax = plt.subplots()
    lines = []
    colors = np.random.rand(len(perms), 3)

    for i, prm in enumerate(perms):
        idx = prm.astype(int)
        tmp = np.array([(y, m) for y, m in zip(years, mags[idx])])
        ctab = clean_completeness(tmp)
        coo = get_xy(ctab, rndx=0.0, rndy=0.1)
        # ax.plot(coo[:, 0], coo[:, 1], color=colors[i, :])
        lines.append(tuple([(x, y) for x, y in zip(coo[:, 0], coo[:, 1])]))

    coll = matplotlib.collections.LineCollection(lines, colors=colors)
    ax.add_collection(coll)

    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Magnitude [yr]')
    ax.grid(which='both')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(mmin, mmax)

    plt.savefig('completeness.png')
    plt.show()


main.folder = 'Folder containing the completeness files'
main.ymin = 'Minimum year in the plot'
main.ymax = 'Maximum year in the plot'
main.mmin = 'Minimum year in the plot'
main.mmax = 'Maximum year in the plot'

if __name__ == '__main__':
    sap.run(main)
