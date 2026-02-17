#!/usr/bin/env python3
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
import glob
import math
import logging
import argparse

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import toml as tomllib  # pip install toml

from openquake.baselib import sap


import numpy as np
import pandas as pd


def _distribute_total_rates(
    aGR: float, bGR: float, fname_in: str, fname_out: str, fraction_flat: float
):
    """
    Distributes the seismicity specified by the aGR and bGR parameters
    over an irregular grid and writes a .csv with the coordinates of each
    point with the associated a- and b-values

    :param aGR:
        Gutenberg–Richter a-value
    :param bGR:
        Gutenberg–Richter b-value
    :param fname_in:
        Input CSV file with columns: lon, lat, nocc
    :param fname_out:
        Output CSV file with columns: lon, lat, agr, bgr
    :param fraction_flat:
        The fraction of seismicity to be distributed evenly
    """

    # Fractions
    msg = 'The flat fraction of seismicity must be lower than 1.0'
    assert fraction_flat < 1.0 and fraction_flat >= 0.0, msg
    fraction_smooth = 1.0 - fraction_flat

    # Load the points
    points_df = pd.read_csv(fname_in)

    # Normalize the weights
    normalizing_factor = points_df["nocc"].sum()
    weights = points_df["nocc"] / normalizing_factor

    assert abs(1.0 - weights.sum()) < 1e-10

    # Total activity rate
    total_activity_rate = 10.0**aGR

    # Flat contribution
    flat_contr = np.full_like(
        weights.values,
        total_activity_rate * fraction_flat / len(weights.values),
        dtype=float,
    )

    # Compute the smoothed seismicity component and add the flat part
    aGR_points = np.log10(
        total_activity_rate * weights.values * fraction_smooth + flat_contr
    )
    bGR_points = np.full_like(aGR_points, bGR, dtype=float)

    # Creating output DataFrame
    outdf = pd.DataFrame(
        {
            "lon": points_df["lon"],
            "lat": points_df["lat"],
            "agr": aGR_points,
            "bgr": bGR_points,
        }
    )

    # Write output
    outdf.to_csv(fname_out, index=False)


def _distribute_rates(
    folder_smooth: str,
    fname_config: str,
    folder_out: str,
    eps_b: float = 0.0,
    eps_rate: float = 0.0,
    fraction_flat: float = 0.0,
):
    """

    :param folder_smooth:
        Folder containing CSV files from smoothing algorithm
    :param fname_config:
        TOML configuration file
    :param folder_out:
        Output folder
    :param eps_b:
        Epsilon for bgr
    :param eps_rate:
        Epsilon for rate above rmag
    :returns:

    """

    # Parse configuration file
    with open(fname_config, "rb") as f:
        config = tomllib.load(f)

    # Loop over CSV files
    for tmps in glob.glob(os.path.join(folder_smooth, "*.csv")):

        fname = os.path.basename(tmps)
        source_id = fname.split(".")[0]
        src_cfg = config["sources"][source_id]

        # bgr
        sigma_b = src_cfg.get("bgr_sig", 0.0)
        try:
            bgr = src_cfg["bgr"] + sigma_b * eps_b
        except KeyError:  # not defined perhaps due to low seismicity
            logging.warning(f"{source_id} does not have bgr defined")
            continue

        # If rmag exists → support uncertainty on rate
        if "rmag" in src_cfg:
            rmag = src_cfg["rmag"]
            lambda_rmag = src_cfg["rmag_rate"]
            lambda_rmag_sig = src_cfg["rmag_rate_sig"]
            agr = math.log10(
                (lambda_rmag + eps_rate * lambda_rmag_sig)
                / (10 ** (-bgr * rmag))
            )
        else:
            # No uncertainty supported
            if eps_rate != 0.0:
                raise ValueError(
                    "eps_rate must be equal to 0 since rmag is not defined"
                )

            agr = src_cfg["agr"]

        # Output file name
        fname_out = os.path.join(folder_out, f"{source_id}.csv")

        # Distribute the seismicity and write the output file
        _distribute_total_rates(agr, bgr, tmps, fname_out, fraction_flat)
