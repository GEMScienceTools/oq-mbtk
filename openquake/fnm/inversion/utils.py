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

from openquake.hazardlib.mfd import (
    TruncatedGRMFD,
    TaperedGRMFD,
    YoungsCoppersmith1985MFD,
)
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
    rup_df,
    mag_col='mag',
    displacement_col='displacement',
    subfaults_col='subfaults',
    faults_col='faults',
    fault_fracs_col='fault_frac_area',
):
    rupture_dicts = []
    for i, rup in rup_df.iterrows():
        rupture_dicts.append(
            {
                "idx": i,
                "M": rup[mag_col],
                "D": rup[displacement_col],
                "faults": rup[subfaults_col],
                "faults_orig": {
                    f: rup[fault_fracs_col][i]
                    for i, f in enumerate(rup[faults_col])
                },
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


def _nearest(val, vals):
    return vals[np.argmin(np.abs(vals - val))]


def make_fault_mfd(
    fault,
    mfd_type='TruncatedGRMFD',
    b_val=1.0,
    seismic_fraction=1.0,
    min_mag=5.0,
    max_mag=8.0,
    bin_width=0.1,
):
    moment_rate = (
        fault['surface'].get_area()
        * fault['net_slip_rate']
        * SHEAR_MODULUS
        * 1e3
        * seismic_fraction
    )

    if mfd_type == 'TruncatedGRMFD':
        mfd = TruncatedGRMFD.from_moment(
            min_mag=min_mag,
            max_mag=max_mag,
            bin_width=bin_width,
            b_val=b_val,
            moment_rate=moment_rate,
        )
    elif mfd_type == 'TaperedGRMFD':
        raise NotImplementedError("only truncated grmfd for now")
    elif mfd_type == 'YoungsCoppersmith1985MFD':
        if min_mag >= (max_mag - 0.5):
            raise ValueError(
                f"fault {fault['fid']} has min mag {min_mag} and max mag {max_mag}"
            )
        mfd = YoungsCoppersmith1985MFD.from_total_moment_rate(
            min_mag=min_mag,
            b_val=b_val,
            char_mag=max_mag - 0.25,
            total_moment_rate=moment_rate,
            bin_width=bin_width,
        )
    else:
        raise NotImplementedError("only truncated grmfd for now")

    return mfd


def get_mag_counts(rups, key="M"):
    mag_counts = {}
    for rup in rups:
        if rup[key] in mag_counts:
            mag_counts[rup[key]] += 1
        else:
            mag_counts[rup[key]] = 1

    return mag_counts


def get_mfd_occurrence_rates(mfd, mag_decimals=1):
    if hasattr(mfd, "get_annual_occurrence_rates"):
        mfd_occ_rates = {
            np.round(r[0], mag_decimals): r[1]
            for r in mfd.get_annual_occurrence_rates()
        }
    elif isinstance(mfd, dict):
        mfd_occ_rates = {
            np.round(M, mag_decimals): rate for M, rate in mfd.items()
        }
    else:
        raise ValueError("mfd must be a dictionary or an MFD object")

    return mfd_occ_rates


def set_single_fault_rupture_rates_by_mfd(
    ruptures, mfd, mag_decimals=1, scale_moment=True
):
    mfd_rates = get_mfd_occurrence_rates(mfd, mag_decimals=mag_decimals)
    mfd_mags = np.array(list(mfd_rates.keys()))

    for rup in ruptures:
        rup['M_mfd'] = _nearest(rup['M'], mfd_mags)

    mag_counts = get_mag_counts(ruptures, key='M_mfd')

    # getting MFD rates per mag, only for magnitude bins w/ ruptures
    mfd_rup_rates = {
        mag: mfd_rates[mag] / count for mag, count in mag_counts.items()
    }

    rup_rates = [mfd_rup_rates[rup['M_mfd']] for rup in ruptures]
    rup_rates = pd.Series(
        data=rup_rates, index=[rup['idx'] for rup in ruptures]
    )

    if scale_moment is True:
        # check that this is the same as what is passed to the MFD!!
        mag_moment = sum(
            [mag_to_mo(mag) * rate for mag, rate in mfd_rates.items()]
        )
        fault = None
        for rup in ruptures:
            if len(rup['faults_orig']) == 1:
                fault = list(rup['faults_orig'].keys())[0]
                break
        if fault is None:
            raise ValueError("cannot determine fault")

        all_rup_moment = sum(
            [
                mag_to_mo(rup['M'])
                * rup_rates[rup['idx']]
                * rup['faults_orig'][fault]
                for rup in ruptures
            ]
        )

        rup_freq_adjust = mag_moment / all_rup_moment
        rup_rates *= rup_freq_adjust

    return rup_rates


def set_single_fault_rup_rates(
    fault_id,
    fault_network,
    mfd=None,
    b_val=1.0,
    seismic_fraction=1.0,
    rup_df='rupture_df',
    mfd_type='TruncatedGRMFD',
):
    fault = _get_fault_by_id(fault_id, fault_network['faults'])
    fault_rup_df = get_ruptures_on_fault(fault_id, fault_network[rup_df])
    rups = rup_df_to_rupture_dicts(
        fault_rup_df, mag_col='mag', displacement_col='displacement'
    )

    if mfd is None:
        mfd = make_fault_mfd(
            fault,
            max_mag=fault_rup_df.mag.max(),
            min_mag=4.0,
            seismic_fraction=seismic_fraction,
            mfd_type=mfd_type,
            b_val=b_val,
        )
    rup_rates = set_single_fault_rupture_rates_by_mfd(rups, mfd)

    return rup_rates


def _get_fault_by_id(fault_id, faults):
    for flt in faults:
        if flt['fid'] == fault_id:
            fault = flt
            break
    else:
        fault = None

    if fault is None:
        raise ValueError(f"fault {fault_id} not found in fault network")

    return fault


def get_ruptures_on_fault(fault_id, rupture_df):
    return rupture_df[rupture_df['faults'].apply(lambda x: fault_id in x)]


def get_rup_rates_from_fault_slip_rates(
    fault_network,
    b_val=1.0,
    mfd_type='TruncatedGRMFD',
    plot_fault_moment_rates=False,
    seismic_fraction=1.0,
    rupture_set_for_rates_from_slip_rates='filtered',
    **kwargs,
):
    """
    Estimates rupture rates from fault slip rates by fitting a magnitude-
    frequency distribution to each fault, from the given parameters and
    a moment rate calculated from the fault slip rate and area.

    Parameters
    ----------
    fault_network : dict
        Fault network dictionary.
    b_val : float
        b-value for magnitude-frequency distribution.
    mfd_type : str
        Magnitude-frequency distribution type. Options are 'TruncatedGRMFD',
        'TaperedGRMFD', and 'YoungsCoppersmith1985MFD'.
    plot_fault_moment_rates : bool
        Whether to plot a comparison of fault moment rates from slip rates
        and rupture rates.
    seismic_fraction : float
        Fraction of slip that is seismic.
    rupture_set_for_rates_from_slip_rates : str
        Which rupture set to use for calculating rupture rates from slip rates.
        Options are 'filtered' and 'all'.
    **kwargs
        Additional keyword arguments to pass to make_fault_mfd.

    Returns
    -------
    final_rup_rates : pd.Series
        Rupture rates indexed by rupture index.
    """

    if rupture_set_for_rates_from_slip_rates == 'filtered':
        rup_df_key = 'rupture_df_keep'
    elif rupture_set_for_rates_from_slip_rates == 'all':
        rup_df_key = 'rupture_df'

    fault_mfds = {}
    for fault in fault_network['faults']:
        fault_mfds[fault['fid']] = make_fault_mfd(
            fault,
            max_mag=get_ruptures_on_fault(
                fault['fid'], fault_network[rup_df_key]
            ).mag.max(),
            mfd_type=mfd_type,
            b_val=b_val,
            seismic_fraction=seismic_fraction,
            **kwargs,
        )

    all_rup_rates = {
        fault['fid']: set_single_fault_rup_rates(
            fault['fid'],
            fault_network,
            mfd=fault_mfds[fault['fid']],
            rup_df=rup_df_key,
            b_val=b_val,
            mfd_type=mfd_type,
            seismic_fraction=seismic_fraction,
            **kwargs,
        )
        for fault in fault_network['faults']
    }

    sf_inds = fault_network['single_rup_df'].index

    final_rup_rates = {}
    mf_rates = {}
    for fault, rates in all_rup_rates.items():
        for idx, rate in rates.items():
            if idx in sf_inds:
                if idx not in final_rup_rates:
                    final_rup_rates[idx] = rate
                else:
                    print(f"{idx} already found")
            else:
                if idx not in mf_rates.keys():
                    mf_rates[idx] = {fault: rate}
                else:
                    mf_rates[idx][fault] = rate

    mf_rup_rates = {}
    for rup, rates in mf_rates.items():
        faults, fault_fracs = fault_network['rupture_df'].loc[
            rup, ['faults', 'fault_frac_area']
        ]
        fault_weights = fault_fracs
        fault_rates = [rates[flt] for flt in faults]
        weighted_mean_rate = weighted_mean(fault_rates, fault_weights)
        mf_rup_rates[rup] = weighted_mean_rate

    final_rup_rates = pd.concat(
        (pd.Series(final_rup_rates), pd.Series(mf_rup_rates))
    )

    # just to check moment rates for faults
    if plot_fault_moment_rates:
        import matplotlib.pyplot as plt

        rups = rup_df_to_rupture_dicts(fault_network['rupture_df'])
        fault_moment_rates_rup = {}
        for rup in rups:
            for fault in rup['faults_orig']:
                rup_moment_rate = (
                    rup['faults_orig'][fault]
                    * mag_to_mo(rup['M'])
                    * final_rup_rates[rup['idx']]
                )
                if fault not in fault_moment_rates_rup:
                    fault_moment_rates_rup[fault] = rup_moment_rate
                else:
                    fault_moment_rates_rup[fault] += rup_moment_rate

        fault_moment_rates_slip = {}
        for fault in fault_network['faults']:
            # moment_rate = (
            #    fault['surface'].get_area()
            #    * fault['net_slip_rate']
            #    * SHEAR_MODULUS
            #    * 1e3
            #    * seismic_fraction
            # )
            moment_rate = sum(
                [
                    mag_to_mo(mag) * rate
                    for mag, rate in get_mfd_occurrence_rates(
                        fault_mfds[fault['fid']]
                    ).items()
                ]
            )
            fault_moment_rates_slip[fault['fid']] = moment_rate

        plt.plot(
            [0, max(fault_moment_rates_slip.values())],
            [0, max(fault_moment_rates_slip.values())],
            '--',
            lw=0.25,
        )
        plt.plot(
            fault_moment_rates_slip.values(),
            [
                fault_moment_rates_rup[fault]
                for fault in fault_moment_rates_slip.keys()
            ],
            '.',
        )
        plt.xlabel(
            "Fault moment rate from slip rates\n"
            + "(corrected for moment release below M_min)"
        )
        plt.ylabel("Fault moment rate from ruptures")
        plt.title("Fault moment rates from slip rate and rupture rates")
        plt.show()

    return final_rup_rates
