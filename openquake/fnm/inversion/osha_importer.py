# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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

import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd

from openquake.hazardlib.geo import Line, Point

from .utils import SHEAR_MODULUS, get_rupture_displacement


def opensha_sections_to_faults(opensha_sections: gpd.GeoDataFrame) -> list:
    def row_to_fault(row):
        fault = {
            "id": row["FaultID"],
            "slip_rate": row["SlipRate"],
            "slip_rate_err": row["SlipRateStdDev"],
            "trace": list(zip(*row["geometry"].coords.xy)),
            "dip": row["DipDeg"],
            "usd": row["UpDepth"],
            "lsd": row["LowDepth"],
            "rake": row["Rake"],
        }
        oq_trace = Line([Point(*c) for c in fault["trace"]])

        fault["area"] = (
            oq_trace.get_length()
            * (row["LowDepth"] - row["UpDepth"])
            / np.sin(np.radians(row["DipDeg"]))
        )

        return fault

    faults = [row_to_fault(row) for _, row in opensha_sections.iterrows()]
    return faults


def read_opensha_rup_indices(filepath: str) -> Dict[int, List[int]]:
    rup_inds = {}

    with open(filepath, "r") as ff:
        lines = ff.readlines()

    # first line is a sort of header, can skip
    def parse_line(line):
        parts = line.split(",")

        rup_id = int(parts[0])
        # num_sections = int(parts[1])  # just FYI - commented out not used
        fault_sections = [int(p) for p in parts[2:]]

        return {rup_id: fault_sections}

    for line in lines[1:]:
        rup_inds.update(parse_line(line))

    return rup_inds


def read_opensha_ruptures(
    fault_sections_file=None,
    rup_indices_file=None,
    rup_properties_file=None,
    plausibility_file=None,  # not used yet
    shear_modulus=SHEAR_MODULUS,
    min_mag=None,
    max_mag=None,
):
    logging.info("Reading OpenSHA rupture files")
    fault_sections_df = gpd.read_file(fault_sections_file)
    faults = opensha_sections_to_faults(fault_sections_df)
    rup_indices = read_opensha_rup_indices(rup_indices_file)
    rup_properties = pd.read_csv(rup_properties_file, index_col=0)

    # TODO: add plausibility values

    logging.info("Building ruptures")
    ruptures = [
        get_opensha_rup_info(
            rup_idx,
            fault_sections=faults,
            rup_indices=rup_indices,
            rup_properties=rup_properties,
            shear_modulus=shear_modulus,
        )
        for rup_idx in rup_properties.index
    ]

    if min_mag is not None:
        logging.info(f"Filtering ruptures with M >= {min_mag}")
        ruptures = [r for r in ruptures if r["M"] >= min_mag]
    if max_mag is not None:
        logging.info(f"Filtering ruptures with M <= {max_mag}")
        ruptures = [r for r in ruptures if r["M"] <= max_mag]

    logging.info(
        f"Dataset contains {len(ruptures)} ruptures and {len(faults)} faults"
    )

    logging.info("Calculating fault adjacence")
    fault_adjacences = get_fault_adjacence(faults)

    return faults, ruptures, fault_adjacences


def get_fault_adjacence(faults):
    fault_adjacence = {}

    fault_sets = {i: set(fault["trace"]) for i, fault in enumerate(faults)}

    for i in fault_sets.keys():
        for j in fault_sets.keys():
            if i == j:
                continue
            if not fault_sets[i].isdisjoint(fault_sets[j]):
                fault_adjacence.setdefault(i, []).append(j)

    return fault_adjacence


def construct_traces(section_idxs, faults):
    # not sure how this handles branching traces...
    master_trace = []
    this_trace = []
    for section_idx in section_idxs:
        section_coords = faults[section_idx]["trace"]
        if this_trace == [] or section_coords[0] == this_trace[-1]:
            this_trace.extend(section_coords[1:])
        else:
            master_trace.append(this_trace)
            this_trace = section_coords

    master_trace.append(this_trace)
    return master_trace


def get_opensha_rup_info(
    rup_idx,
    fault_sections=None,
    rup_indices=None,
    rup_properties=None,
    shear_modulus=SHEAR_MODULUS,
):
    rd = {
        "rup_id": rup_idx,
        "M": np.round(rup_properties.loc[rup_idx, "Magnitude"], 1),
        "area": rup_properties.loc[rup_idx, "Area (m^2)"]
        * 1e-6,  # store in km^2
        "faults": rup_indices[rup_idx],
        "mean_rake": rup_properties.loc[rup_idx, "Average Rake (degrees)"],
        "plausibility": None,  # will address later
    }

    rd["D"] = get_rupture_displacement(rd["M"], rd["area"], shear_modulus)
    try:
        rd["trace"] = construct_traces(rd["faults"], fault_sections)
    except IndexError:
        print(rd["faults"])

    return rd
