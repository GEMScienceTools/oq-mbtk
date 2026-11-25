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
Smoothing of earthquake counts on an H3 grid using a mixture of gaussian
kernels. Python port of the `smoothing` functions in the Julia
PSHAModelBuilder package (src/seismicity/smoothing.jl).
"""

import os
import h3
import numpy as np
import pandas as pd

from openquake.wkf.seismicity.box_counting import _load_toml


def get_avg_edge_length_km(h3res: int) -> float:
    """
    :param h3res:
        The H3 resolution
    :returns:
        The average length [km] of the edge of an hexagon at resolution
        `h3res`
    """
    return h3.average_hexagon_edge_length(h3res, unit='km')


def normal_pdf(x, std: float, mean: float = 0.0):
    """
    Probability density function of a normal distribution.

    :param x:
        A scalar or an array with the values where to compute the pdf
    :param std:
        The standard deviation
    :param mean:
        The mean
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))


def get_kernel_weights(dsts, smoothing_sigmas) -> np.ndarray:
    """
    Computes the normalised weights of a mixture of gaussian kernels.

    :param dsts:
        An array with distances [km]
    :param smoothing_sigmas:
        A list of (weight, std) tuples where std is a standard deviation [km]
    :returns:
        An array with the same length of `dsts` with normalised weights
        (i.e. summing to 1)
    """
    dsts = np.asarray(dsts, dtype=float)
    wei = np.zeros_like(dsts)
    for weight, std in smoothing_sigmas:
        wei += normal_pdf(dsts, std) * weight
    wei /= np.sum(wei)
    if abs(np.sum(wei) - 1.0) > 1e-5:
        raise ValueError('Weights are wrong')
    return wei


def get_cell_distances(base: str, idxs, edge_length: float,
                       default_distance: int = 20) -> np.ndarray:
    """
    Computes the distance [km] between the cell `base` and the cells in
    `idxs` as the grid distance multiplied by the average edge length.

    :param base:
        The H3 index of the reference cell
    :param idxs:
        A list of H3 indexes
    :param edge_length:
        The average edge length [km] at the resolution of the cells
    :param default_distance:
        The grid distance used when H3 fails to compute it
    :returns:
        An array of distances [km]. Distances lower than 1 km are set to 1 km
    """
    dsts = np.zeros(len(idxs))
    for i, idx in enumerate(idxs):
        if base == idx:
            d = 0.0
        else:
            try:
                d = h3.grid_distance(base, idx)
            except Exception:
                print(f'failed: {base}, {idx}. Using default '
                      f'{default_distance}')
                d = default_distance
        dsts[i] = max(d * edge_length, 1.0)
    return dsts


def smooth_counts(fname_count: str, smoothing_sigmas, maxdistkm: float, *,
                  default_distance: int = 20) -> pd.DataFrame:
    """
    The `fname_count` is a .csv file containing for each row a
    (h3idx, lon, lat, count) tuple. This function smooths the count in this
    file using the gaussian kernels defined in the array `smoothing_sigmas`;
    this array contains N (weight, std) tuples where std is a standard
    deviation [km]. The sum of weights in `smoothing_sigmas` must be equal
    to 1. The `maxdistkm` is the maximum distance [km] used to perform the
    smoothing.

    :param fname_count:
        Name of the .csv file produced by box counting (it must include the
        `h3idx`, `lon`, `lat` and `count` columns)
    :param smoothing_sigmas:
        A list of (weight, std) tuples
    :param maxdistkm:
        Maximum smoothing distance [km]
    :param default_distance:
        Grid distance used when H3 fails to compute the distance between
        two cells
    :returns:
        A :class:`pandas.DataFrame` with columns `lon`, `lat` and `nocc`
    """
    df = pd.read_csv(fname_count)
    h3idxs = df.h3idx.astype(np.uint64)

    # Find the resolution and the corresponding edge length
    h3res = h3.get_resolution(h3.int_to_str(int(h3idxs.iloc[0])))
    edge_length = get_avg_edge_length_km(h3res)
    print(f'Edge resolution : {h3res:d}')
    print(f'Edge length     : {edge_length:.3f} km')

    maxdistk = int(np.ceil(maxdistkm / edge_length))
    print(f'Max dist k      : {maxdistk:d}')
    print(f'Number nodes    : {len(df)}')

    nocc = {}
    for lon, lat, count in zip(df.lon, df.lat, df['count']):

        # Cell index to which the current point (i.e. cell center) belongs
        base = h3.latlng_to_cell(lat, lon, h3res)

        # Indexes of cells within 'maxdistk' from the cell with index equal
        # to 'base'. 'maxdistk' is the distance in terms of the cell size
        idxs = [i for i in h3.grid_disk(base, maxdistk)
                if h3.is_valid_cell(i)]

        # Distances and gaussian weights
        dsts = get_cell_distances(base, idxs, edge_length, default_distance)
        wei = get_kernel_weights(dsts, smoothing_sigmas)

        # Updating the number of occurrences
        for idx, w in zip(idxs, wei):
            nocc[idx] = nocc.get(idx, 0.0) + count * w

    keys = list(nocc.keys())
    coos = [h3.cell_to_latlng(k) for k in keys]
    smooth = pd.DataFrame({
        'lon': np.array([c[1] for c in coos], dtype=float),
        'lat': np.array([c[0] for c in coos], dtype=float),
        'nocc': np.array([nocc[k] for k in keys], dtype=float)})
    return smooth


def smoothing(fname_count: str, fname_config: str,
              fname_out: str = '') -> pd.DataFrame:
    """
    Smooths the results of box counting using the kernel parameters in the
    configuration file and saves the results in a .csv file.

    :param fname_count:
        Name of the .csv file produced by box counting
    :param fname_config:
        Name of the .toml configuration file. It must contain the
        `kernel_maximum_distance` [km] and the `kernel_smoothing` (a list of
        [weight, std] pairs) parameters
    :param fname_out:
        Name of the output .csv file. When empty, the output is saved in
        `./tmp/<basename>_smooth.csv`
    :returns:
        A :class:`pandas.DataFrame` with columns `lon`, `lat` and `nocc`
    """
    model = _load_toml(fname_config)
    maxdistkm = model['kernel_maximum_distance']
    smoothing_sigmas = model['kernel_smoothing']
    smooth = smooth_counts(fname_count, smoothing_sigmas, maxdistkm)

    if len(fname_out) == 0:
        tmp = os.path.basename(fname_count).split('.')[0]
        fname_out = os.path.join('./tmp', f'{tmp}_smooth.csv')

    smooth.to_csv(fname_out, columns=['lon', 'lat', 'nocc'], index=False)
    print(f'Created         : {fname_out}')
    return smooth
