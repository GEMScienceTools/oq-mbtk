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

import numpy as np
import multiprocessing
from itertools import product


def mm(a):
    return a


def main():
    """
    Creates three .npz files with all the completeness windows admitted by
    the combination of years and magnitudes provided.
    """

    tmp = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 1995,
           2000, 2005, 2010]
    # tmp = [1900, 1930, 1960, 1970, 1980]
    # tmp = [1900, 1930, 1960, 1970]
    years = np.array(tmp)
    mags = np.array([4.0, 4.5, 5, 5.5, 6, 6.5, 7])
    #mags = np.array([4.0, 5.0, 6.0, 7.0])
    idxs = np.arange(len(mags))
    idxs[::-1].sort()
    max_first_idx = 5

    print('Total number of combinations: {:,d}'.format(len(mags)**len(years)))

    step = 4
    perms = []
    for y in [years[i:min(i+step, len(years))] for i in range(0, len(years), step)]:
        with multiprocessing.Pool(processes=8) as pool:
            p = pool.map(mm, product(idxs, repeat=len(y)))
            p = np.array(p)
            p = p[np.diff(p, axis=1).min(axis=1) >= 0, :]
            if len(perms):
                new = []
                for x in perms:
                    for y in p:
                        new.append(list(x)+list(y))
                perms = new
            else:
                perms = p
        p = np.array(perms)
        p = p[np.diff(p, axis=1).min(axis=1) >= -1e-10, :]
        p = p[p[:, 0] <= max_first_idx]
        perms = p

    np.save('dispositions.npy', perms)
    np.save('mags.npy', mags)
    np.save('years.npy', years)


if __name__ == '__main__':
    main()
