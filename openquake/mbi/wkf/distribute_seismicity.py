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
from openquake.baselib import sap
from openquake.wkf.seismicity.distribute import _distribute_rates


def main(
        smooth_folder,
        config,
        folder_out,
        eps_b=0.0,
        eps_rate=0.0,
        fraction_flat=0.0
    ):
    """
    Distributes the rates using the output of a seismicity smoothing function.
    """

    # Create output folder if needed
    os.makedirs(folder_out, exist_ok=True)

    _distribute_rates(
        smooth_folder,
        config,
        folder_out,
        eps_b,
        eps_rate,
        fraction_flat
    )

main.smooth_folder = "Folder containing CSV files from smoothing algorithm"
main.config = "TOML configuration file"
main.folder_out = "Output folder"
main.eps_b = "Epsilon for bgr"
main.eps_rate = "Epsilon for rate above rmag"
main.fraction_flat = "The fractional part to be distributed evenly"

if __name__ == "__main__":
    sap.run(main)
