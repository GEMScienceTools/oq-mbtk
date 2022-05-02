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


from prettytable import PrettyTable


def get_table_agency_mag(work, mthresh=5.0, nthresh=0):
    """
    :param work:
        A :class:`pandas.DataFrame` instance
    :param mthresh:
        Minimum magnitude (original scale) threshold
    :param nthresh:
        Number of values threshold
    :return:
        An instance of :class:`prettytable.PrettyTable` and a list of list
        that can be used with the package tabulate.
    """

    x = PrettyTable()
    tbl = []
    x.field_names = ["Magnitude Type", "Agency", "Number of events"]
    other = 0
    total = 0

    prev = ""
    yyy = work[(work["value"] > mthresh)].groupby(["magAgency", "magType"])
    for name in yyy.groups:
        if len(yyy.groups[name]) > nthresh:
            total += len(yyy.groups[name])
            tmps = "" if prev == name[0] else name[0]
            prev = name[0]
            tmp = [tmps, name[1], len(yyy.groups[name])]
            x.add_row(tmp)
            tbl.append(tmp)
        else:
            other += len(yyy.groups[name])
    x.add_row(["TOTAL", "", "{:d}".format(total)])

    return x, tbl


def get_table_mag_agency(work, mthresh=5.0, nthresh=0):
    """
    :param work:
        A :class:`pandas.DataFrame` instance
    :param mthresh:
        Minimum magnitude (original scale) threshold
    :param nthresh:
        Number of values threshold
    :return:
        An instance of :class:`prettytable.PrettyTable` and a list of list
        that can be used with the package tabulate.
    """

    x = PrettyTable()
    tbl = []
    x.field_names = ["Magnitude Type", "Agency", "Number of events"]
    other = 0
    total = 0

    prev = ""
    yyy = work[(work["value"] > mthresh)].groupby(["magType", "magAgency"])
    for name in yyy.groups:
        if len(yyy.groups[name]) > nthresh:
            total += len(yyy.groups[name])
            tmps = "" if prev == name[0] else name[0]
            prev = name[0]
            tmp = [tmps, name[1], len(yyy.groups[name])]
            x.add_row(tmp)
            tbl.append(tmp)
        else:
            other += len(yyy.groups[name])
    x.add_row(["TOTAL", "", "{:d}".format(total)])

    return x, tbl


def get_number_unique_events_per_agency(work, mthresh=5.0):
    """
    :param work:
        A :class:`pandas.DataFrame` instance
    :param mthresh:
        Minimum magnitude (original scale) threshold
    """
    yyy = work[(work["value"] > mthresh)]
