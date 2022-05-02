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
import shutil
from pathlib import Path


def create_folder(folder: str, clean: bool = False):
    """
    Create a folder. If the folder exists, it's possible to
    clean it.

    :param folder:
        The name of the folder tp be created
    :param clean:
        When true the function removes the content of the folder
    """
    if os.path.exists(folder):
        if clean:
            shutil.rmtree(folder)
    else:
        Path(folder).mkdir(parents=True, exist_ok=True)


def _get_src_id(fpath: str) -> str:
    """
    Returns the ID of the source included in a string with the
    format `whatever_<source_id>.csv`

    :param fpath:
        The string containign the source ID
    :returns:
        The source ID
    """
    fname = os.path.basename(fpath)
    if '_' in fname:
        pattern = '.*_+(\\S*)\\..*'
        # pattern = '.*[_|\\.]([\\w|-]*)\\.'
    else:
        pattern = '(\\S*)\\..*'
    mtch = re.search(pattern, fname)

    try:
        return mtch.group(1)
    except:
        fmt = 'Name {:s} does not comply with standards'
        raise ValueError(fmt.format(fname))


def get_list(tmps, sep=','):
    """
    Given a string of elements separated by a `separator` returns a list of
    elements.

    :param tmps:
        The string to be parsed
    :param sep:
        The separator character
    """
    tml = re.split('\\{:s}'.format(sep), tmps)
    # Cleaning
    tml = [re.sub('(^\\s*|\\s^)', '', a) for a in tml]
    return tml
