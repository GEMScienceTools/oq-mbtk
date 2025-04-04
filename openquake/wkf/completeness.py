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

import matplotlib.pyplot as plt


def _plot_ctab(ctab, label='', xlim=None, ylim=None, color='red', ls='-',
               marker=''):
    """
    Plots completeness

    :param ctab:
        A :class:`np.ndarray` instance containing the completeness table
    :param label:
        A string
    :param xlim:
        A tuple with the min and max values on the abscissa
    :param ylim:
        A tuple with the min and max values on the ordinate
    :param color:
        Color to be used for plotting the line
    :param ls:
        Line style
    """
    n = ctab.shape[0]
    if n > 1:
        for i in range(0, n-1):
            plt.plot([ctab[i, 0], ctab[i, 0]], [ctab[i, 1],
                     ctab[i+1, 1]], color=color, ls=ls, marker=marker)
            plt.plot([ctab[i, 0], ctab[i+1, 0]], [ctab[i+1, 1],
                     ctab[i+1, 1]], color=color, ls=ls, marker=marker)
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()

    if xlim is None:
        xlim = [1900, 2020]

    if ylim is None:
        ylim = [4.5, 7.0]

    plt.plot([ctab[n-1, 0], ctab[n-1, 0]], [ylim[1], ctab[n-1, 1]],
             color=color, ls=ls, marker=marker)
    plt.plot([ctab[0, 0], xlim[1]], [ctab[0, 1], ctab[0, 1]],
             label=label, color=color, ls=ls, marker=marker)
