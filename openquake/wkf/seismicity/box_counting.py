#!/usr/bin/env python
# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation
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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8
"""
Box counting of earthquakes on an H3 grid. Python port of the
`boxcounting` function (and its helpers) in the Julia PSHAModelBuilder
package.
"""

import os
import h3
import numpy as np
import pandas as pd

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None
    import toml


def _load_toml(fname: str) -> dict:
    # tomllib follows TOML 1.0 (e.g. accepts arrays mixing int and float
    # like [1980, 4.0]), as the TOML parser used in the Julia version
    if tomllib is not None:
        with open(fname, 'rb') as fin:
            return tomllib.load(fin)
    return toml.load(fname)


def get_gr_params(config: dict, source_id: str):
    """
    Returns the Gutenberg-Richter a and b values of a source.

    :param config:
        A dictionary with the configuration (as read from the .toml file)
    :param source_id:
        The ID of the source
    :returns:
        A tuple with agr and bgr
    """
    sources = config.get('sources', {})
    if source_id not in sources:
        raise ValueError(f'Source {source_id} is not in the configuration')
    src = sources[source_id]
    if 'agr' not in src:
        raise ValueError(f'Source {source_id} does not have the agr attribute')
    if 'bgr' not in src:
        raise ValueError(f'Source {source_id} does not have the bgr attribute')
    return src['agr'], src['bgr']


def get_rate_double_truncated_gr(agr: float, bgr: float, mag: float,
                                 binw: float = 0.1) -> float:
    """
    Using a double truncated GR relationship, computes the rate of occurrence
    of the magnitude bin (with width `binw`) containing `mag`.

    :param agr:
        The GR a-value
    :param bgr:
        The GR b-value
    :param mag:
        The magnitude
    :param binw:
        The width of the magnitude bin
    :returns:
        The rate of occurrence
    """
    lowm = np.floor(mag / binw) * binw
    uppm = lowm + binw
    lorate = 10**(agr - bgr * lowm)
    uprate = 10**(agr - bgr * uppm)
    return lorate - uprate


def get_completeness_table(config: dict, source_id: str):
    """
    Given a dictionary `config` and a source ID, provides the corresponding
    completeness table. When a source does not have a completeness table,
    the default one is used.

    :param config:
        A dictionary with the configuration (as read from the .toml file)
    :param source_id:
        The ID of the source
    :returns:
        A list of [year, magnitude] pairs
    """
    src = config.get('sources', {}).get(source_id, {})
    if 'completeness_table' in src:
        return src['completeness_table']
    return config['default']['completeness_table']


def get_rate_from_completeness(compl, mag: float, year: float,
                               year_end: float) -> float:
    """
    Returns the weight of an earthquake given a completeness table. The
    weight is 1/(duration of the completeness window) for the first window
    (starting from the last row of the table) that the earthquake belongs to,
    0 otherwise.

    :param compl:
        A list of [year, magnitude] pairs
    :param mag:
        The magnitude of the earthquake
    :param year:
        The year of the earthquake
    :param year_end:
        The last year of the catalogue
    :returns:
        The rate
    """
    for cyear, cmag in reversed(compl):
        if mag >= cmag and year >= cyear:
            return 1.0 / (year_end - cyear)
    return 0.0


def read_h3_to_zone_mapping(fname: str) -> dict:
    """
    Reads the .csv file (no header) with the mapping between h3 cells and
    zones.

    :param fname:
        The name of the file
    :returns:
        A dictionary with key the h3 index (str) and value the zone ID (str)
    """
    mapping = {}
    with open(fname, 'r') as fin:
        for row in fin:
            row = row.strip()
            if not row:
                continue
            tmp = row.split(',')
            mapping[tmp[0].strip().lower()] = tmp[1].strip()
    return mapping


def boxcounting(fname: str, h3res: int, fname_h3_to_zone: str = '',
                fname_config: str = '', folder_out: str = '',
                max_year: float = 3000, weighting: str = 'one'):
    """
    Using H3 as a reference, counts the number of earthquakes in each cell.

    :param fname:
        The name of the .csv file with the catalogue. It must contain the
        columns `longitude`, `latitude`, `year` and `magnitude`
    :param h3res:
        The H3 resolution
    :param fname_h3_to_zone:
        The name of the .csv file with the mapping between h3 cells and zones
    :param fname_config:
        The name of the .toml configuration file
    :param folder_out:
        The folder where to write the output. If empty, nothing is written
    :param max_year:
        The last year of the catalogue (used with `completeness` weighting)
    :param weighting:
        One of `one`, `completeness` or `mfd`
    :returns:
        A :class:`pandas.DataFrame` with columns h3idx (int), lon, lat, count
    """
    if weighting not in ('one', 'completeness', 'mfd'):
        raise ValueError(f'Unknown weighting option: {weighting}')

    # Read the mapping between h3 cells and zones
    mapping = {}
    if len(fname_h3_to_zone) > 0:
        mapping = read_h3_to_zone_mapping(fname_h3_to_zone)

    # Read the configuration
    config = None
    if len(fname_config) > 0:
        assert len(fname_h3_to_zone) > 0
        config = _load_toml(fname_config)
    if weighting in ('completeness', 'mfd'):
        assert config is not None, f'Weighting {weighting} requires a config'

    # Load the catalogue
    df = pd.read_csv(fname)

    # Loop over the earthquakes in the catalogue
    count = {}
    binw = 0.1
    for lon, lat, year, mag in zip(df.longitude, df.latitude, df.year,
                                   df.magnitude):

        base = h3.latlng_to_cell(lat, lon, h3res)

        if weighting == 'one':
            rate = 1.0
        elif weighting == 'completeness':
            if base not in mapping:
                continue
            compl = get_completeness_table(config, mapping[base])
            rate = get_rate_from_completeness(compl, mag, year, max_year)
        else:
            if base not in mapping:
                print(f'The h3 to src mapping does not contain this ID: '
                      f'{base}')
                continue
            agr, bgr = get_gr_params(config, mapping[base])
            rate = get_rate_double_truncated_gr(agr, bgr, mag, binw)

        if rate < 1e-20:
            continue

        count[base] = count.get(base, 0.0) + rate

    # Create the output dataframe
    keys = list(count.keys())
    coos = [h3.cell_to_latlng(k) for k in keys]
    coo = pd.DataFrame({
        'h3idx': np.array([h3.str_to_int(k) for k in keys], dtype=np.uint64),
        'lon': np.array([c[1] for c in coos], dtype=float),
        'lat': np.array([c[0] for c in coos], dtype=float),
        'count': np.array([count[k] for k in keys], dtype=float)})

    # Write output .csv files
    if len(folder_out) > 0:
        bname = os.path.basename(fname)
        fname_out = os.path.join(folder_out, f'box_counting_{bname}')
        fname_out_h3 = os.path.join(folder_out, f'box_counting_h3_{bname}')
        coo.to_csv(fname_out, columns=['lon', 'lat', 'count'], index=False)
        coo.to_csv(fname_out_h3, columns=['h3idx', 'lon', 'lat', 'count'],
                   index=False)
        print('Writing ', fname_out)
        print('Writing ', fname_out_h3)

    return coo
