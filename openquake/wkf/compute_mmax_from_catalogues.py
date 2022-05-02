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

import toml
import numpy as np
from glob import glob
from openquake.wkf.utils import _get_src_id
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue


def compute_mmax(fname_input_pattern: str, fname_config: str, label: str):
    """
    This function assignes an mmax value to each source with a catalogue
    file as selected by the provided `fname_input_pattern`.
    """

    if isinstance(fname_input_pattern, str):
        fname_list = glob(fname_input_pattern)
    else:
        fname_list = fname_input_pattern

    # Parsing config
    model = toml.load(fname_config)

    # Processing files
    for fname in sorted(fname_list):

        src_id = _get_src_id(fname)

        # Processing catalogue
        tcat = _load_catalogue(fname)

        if tcat is None or len(tcat.data['magnitude']) < 2:
            continue

        tmp = "{:.5e}".format(np.max(tcat.data['magnitude']))
        model['sources'][src_id]['mmax_{:s}'.format(label)] = float(tmp)

    # Saving results into the config file
    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(model))
        print('Updated {:s}'.format(fname_config))
