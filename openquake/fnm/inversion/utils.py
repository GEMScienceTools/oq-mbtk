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

import numpy as np
import pandas as pd
import pyproj as pj
import geopandas as gpd

from math import prod

from shapely.ops import transform
from shapely.geometry import Point, LineString

from openquake.baselib.general import AccumDict
from openquake.hazardlib.mfd.tapered_gr_mfd import mag_to_mo

from openquake.fnm.constants import SHEAR_MODULUS


def geom_from_fault_trace(fault_trace):
    return LineString([Point(*c) for c in fault_trace])


def project_faults_and_polies(faults, polies: gpd.GeoDataFrame):
    lines = []
    # lines = [geom_from_fault_trace(fault["trace"]) for fault in faults]
    for i, fault in enumerate(faults):
        try:
            lines.append(geom_from_fault_trace(fault["trace"]))
        except:
            print(i, fault['trace'])
            raise

    trans = pj.Transformer.from_crs(4326, "ESRI:102016", always_xy=True)

    lines_proj = [transform(trans.transform, line) for line in lines]
    polies_proj = polies.to_crs("ESRI:102016")

    return lines_proj, polies_proj


def lines_in_polygon(faults, region_polies: gpd.GeoDataFrame):
    lines_proj, polies_proj = project_faults_and_polies(faults, region_polies)

    lines_in_polies = {
        rp["id"]: [
            faults[i]
            for i, line in enumerate(lines_proj)
            if rp.geometry.contains(line)
        ]
        for j, rp in polies_proj.iterrows()
    }

    return lines_in_polies


def get_rupture_displacement(
    rup_magnitude, rup_area, shear_modulus=SHEAR_MODULUS
):
    return mag_to_mo(rup_magnitude) / (rup_area * 1e6 * shear_modulus)


def weighted_mean(values, fracs):
    return sum(prod(vals) for vals in zip(values, fracs)) / sum(fracs)


def slip_vector_azimuth(strike, dip, rake):
    # Convert degrees to radians
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    rake_rad = np.radians(rake) * -1

    # Calculate the 3D Cartesian coordinates of the slip vector
    slip_x = -np.sin(rake_rad) * np.sin(strike_rad) - np.cos(
        rake_rad
    ) * np.sin(dip_rad) * np.cos(strike_rad)
    slip_y = np.sin(rake_rad) * np.cos(strike_rad) - np.cos(rake_rad) * np.sin(
        dip_rad
    ) * np.sin(strike_rad)

    # Calculate the azimuth of the slip vector
    azimuth = np.degrees(np.arctan2(slip_y, slip_x))

    # Ensure the azimuth is between 0 and 360 degrees
    if azimuth < 0:
        azimuth += 360
    if azimuth >= 360.0:
        azimuth -= 360.0

    return azimuth


def check_fault_in_poly(fault, polies, id_key='id'):
    poly_membership = []

    for i, p in polies.iterrows():
        if p.geometry.contains(fault):
            poly_membership.append(p[id_key])
            break

        elif p.geometry.intersects(fault):
            poly_membership.append(p[id_key])

    return poly_membership


def faults_in_polies(faults, polies, id_key='id'):
    if isinstance(faults, pd.DataFrame):
        faults_ = subsection_df_to_fault_dicts(faults)
    else:
        faults_ = faults
    traces_proj, polies_proj = project_faults_and_polies(faults_, polies)
    fault_poly_membership = {
        faults_[i]["id"]: check_fault_in_poly(
            trace, polies_proj, id_key=id_key
        )
        for i, trace in enumerate(traces_proj)
    }

    return fault_poly_membership


def get_rup_poly_fracs(rup, fpm):
    rpf = AccumDict()

    for sec in rup["faults"]:
        polies = fpm[sec]
        if len(polies) > 0:
            poly_fracs = {p: 1 / len(polies) for p in polies}
            rpf += poly_fracs

    tot = sum(rpf.values())

    rpf = {k: v / tot for k, v in rpf.items()}

    return rpf


def rup_df_to_rupture_dicts(
    rup_df, mag_col='M', displacement_col='D', subfaults_col='subfaults'
):
    rupture_dicts = []
    for i, rup in rup_df.iterrows():
        rupture_dicts.append(
            {
                "idx": i,
                "M": rup[mag_col],
                "D": rup[displacement_col],
                "faults": rup[subfaults_col],
            }
        )
    return rupture_dicts


def subsection_df_to_fault_dicts(
    subsection_df,
    slip_rate_col='slip_rate',
    slip_rate_err_col='slip_rate_err',
):
    fault_dicts = []
    for i, fault in subsection_df.iterrows():
        fault_dicts.append(
            {
                "id": i,
                "slip_rate": fault[slip_rate_col],
                "slip_rate_err": fault[slip_rate_err_col],
                "trace": fault["trace"],
                "area": fault["area"],
            }
        )
    return fault_dicts


def get_rupture_regions(
    rup_df: pd.DataFrame,
    subsection_df: pd.DataFrame,
    seis_regions: gpd.GeoDataFrame,
    id_key='id',
):
    fault_poly_membership = faults_in_polies(subsection_df, seis_regions)
    rup_region_fracs = [
        get_rup_poly_fracs(r, fault_poly_membership)
        for i, r in rup_df.iterrows()
    ]

    regional_rup_fractions = {}

    for rowid, row in seis_regions.iterrows():
        i = row[id_key]
        regional_rup_fractions[i] = {'rups': [], 'fracs': []}
        for j, rup in enumerate(rup_region_fracs):
            if i in rup:
                regional_rup_fractions[i]['rups'].append(j)
                regional_rup_fractions[i]['fracs'].append(rup[i])
