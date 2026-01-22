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
from math import prod
from typing import Optional

import numpy as np
import pandas as pd
import pyproj as pj
import geopandas as gpd
import scipy.sparse as ssp

from shapely.ops import transform
from shapely.geometry import Point, LineString

try:
    from ipdb import set_trace

    breakpoint = set_trace
except ImportError:
    breakpoint = breakpoint

from openquake.hazardlib.geo.mesh import Mesh
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


def b_mle(mags, min_mag=4.0):
    mags_include = mags[mags >= min_mag]

    beta = 1 / (mags_include.mean() - min_mag)
    b = np.log10(np.e) * beta
    return b


def get_a_b(mags, min_mag=4.0, cat_duration=40.0, b=None):
    if b is None:
        b = b_mle(mags, min_mag)

    N = len(mags[mags >= min_mag]) / cat_duration
    a = np.log10(N) + b * min_mag
    return a, b


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


def faults_in_polies(
    faults,
    polies,
    fault_id_key='id',
    poly_id_key='id',
    slip_rate_col='net_slip_rate',
    slip_rate_err_col='net_slip_rate_err',
):
    if isinstance(faults, pd.DataFrame):
        faults_ = subsection_df_to_fault_dicts(
            faults,
            slip_rate_col=slip_rate_col,
            slip_rate_err_col=slip_rate_err_col,
        )
    else:
        faults_ = faults
    traces_proj, polies_proj = project_faults_and_polies(faults_, polies)
    fault_poly_membership = {
        faults_[i][fault_id_key]: check_fault_in_poly(
            trace, polies_proj, id_key=poly_id_key
        )
        for i, trace in enumerate(traces_proj)
    }

    return fault_poly_membership


def get_rup_poly_fracs(rup, fpm, fault_key='faults'):
    rpf = AccumDict()

    for sec in rup[fault_key]:
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
    subfault_fracs_col='frac_area',
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
                "subfault_fracs": {
                    f: rup[subfault_fracs_col][i]
                    for i, f in enumerate(rup[subfaults_col])
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
    fault_key='faults',
):
    fault_poly_membership = faults_in_polies(subsection_df, seis_regions)
    rup_region_fracs = [
        get_rup_poly_fracs(r, fault_poly_membership, fault_key=fault_key)
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

    return regional_rup_fractions


def _nearest(val, vals):
    vals = np.asarray(vals)
    return vals[np.argmin(np.abs(vals - val))]


def make_fault_mfd(
    fault,
    mfd_type='TruncatedGRMFD',
    b_val=1.0,
    seismic_fraction=1.0,
    min_mag=5.0,
    max_mag=8.0,
    bin_width=0.1,
    corner_mag=7.5,
    moment_rate=None,
):
    if moment_rate is None:
        moment_rate = (
            fault['surface'].get_area()
            * fault['net_slip_rate']
            * SHEAR_MODULUS
            * 1e3
            * seismic_fraction
        )

    if mfd_type == 'TruncatedGRMFD':
        try:
            mfd = TruncatedGRMFD.from_moment(
                min_mag=min_mag,
                max_mag=max_mag,
                bin_width=bin_width,
                b_val=b_val,
                moment_rate=moment_rate,
            )
        except ValueError:
            mfd = TruncatedGRMFD.from_moment(
                min_mag=min_mag - bin_width,
                max_mag=max_mag,
                bin_width=bin_width,
                b_val=b_val,
                moment_rate=moment_rate,
            )
    elif mfd_type == 'TaperedGRMFD':
        if (max_mag - corner_mag) < 0.5:
            corner_mag = max_mag - 0.5
        if corner_mag < (min_mag + bin_width):
            corner_mag = min_mag + bin_width + 0.01
        if (max_mag - min_mag) < bin_width:
            min_mag = max_mag - bin_width

        mfd = TaperedGRMFD.from_moment(
            min_mag=min_mag,
            max_mag=max_mag,
            corner_mag=corner_mag,
            bin_width=bin_width,
            b_val=b_val,
            moment_rate=moment_rate,
        )

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
        raise NotImplementedError(
            "only truncated, tapered, and youngscoppersmith for now"
        )

    return mfd


def get_mag_counts(rups, key="M", cumulative=False):
    mag_counts = {}
    for rup in rups:
        if rup[key] in mag_counts:
            mag_counts[rup[key]] += 1
        else:
            mag_counts[rup[key]] = 1

    if cumulative is True:
        mag_counts = make_cumulative(mag_counts)

    return mag_counts


def get_mfd_occurrence_rates(mfd, mag_decimals=None, cumulative=False):
    if hasattr(mfd, "get_annual_occurrence_rates"):
        mfd_occ_rates = {r[0]: r[1] for r in mfd.get_annual_occurrence_rates()}
    elif isinstance(mfd, dict):
        mfd_occ_rates = {M: rate for M, rate in mfd.items()}
    else:
        raise ValueError("mfd must be a dictionary or an MFD object")

    if mag_decimals is not None:
        mfd_occ_rates = {
            round(m, mag_decimals): r for m, r in mfd_occ_rates.items()
        }

    if cumulative is True:
        mfd_occ_rates = make_cumulative(mfd_occ_rates)

    return mfd_occ_rates


def get_mfd_moment(mfd, mag_decimals=None):
    mfd_moment = sum(
        [
            mag_to_mo(k) * v
            for k, v in get_mfd_occurrence_rates(
                mfd, mag_decimals=mag_decimals
            ).items()
        ]
    )
    return mfd_moment


def get_mfd_uncertainties(mfd, unc_type='pctile'):
    rates = get_mfd_occurrence_rates(mfd)

    if unc_type == 'std':
        pass


def make_cumulative(dic):
    rev_keys = sorted(dic.keys(), reverse=True)
    new_dic = {}
    current = 0
    for k in rev_keys:
        current += dic[k]
        new_dic[k] = current

    new_dic = {k: new_dic[k] for k in dic.keys()}
    return new_dic


def set_single_fault_rupture_rates_by_mfd(
    ruptures,
    mfd,
    mag_decimals=1,
    scale_moment_rate=True,
    moment_rate=None,
    faults_or_subfaults='faults',
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

    if scale_moment_rate is True:
        if faults_or_subfaults == 'faults':
            fault_key = 'faults_orig'
        elif faults_or_subfaults == 'subfaults':
            fault_key = 'subfault_fracs'
        # check that this is the same as what is passed to the MFD!!

        if moment_rate is None:
            moment_rate = sum(
                [mag_to_mo(mag) * rate for mag, rate in mfd_rates.items()]
            )

        fault = None
        for rup in ruptures:
            if len(rup[fault_key]) == 1:
                fault = list(rup[fault_key].keys())[0]
                break
        if fault is None:
            raise ValueError("cannot determine fault")

        all_rup_moment = sum(
            [
                mag_to_mo(rup['M'])
                * rup_rates[rup['idx']]
                * rup[fault_key][fault]
                for rup in ruptures
            ]
        )

        rup_freq_adjust = moment_rate / all_rup_moment
        rup_rates *= rup_freq_adjust

    return rup_rates


def set_single_fault_rup_rates(
    fault_id,
    fault_network,
    rup_fault_lookup,
    mfd=None,
    b_val=1.0,
    corner_mag=7.5,
    seismic_fraction=1.0,
    rup_df='rupture_df',
    mfd_type='TruncatedGRMFD',
    scale_moment_rate=True,
    faults_or_subfaults='faults',
    moment_rate=None,
):
    if faults_or_subfaults == 'faults':
        fault = _get_fault_by_id(fault_id, fault_network['faults'])
    elif faults_or_subfaults == 'subfaults':
        fault = fault_network['subfault_df'].loc[fault_id]

    fault_rup_df = get_ruptures_on_fault(
        fault_id, fault_network[rup_df], rup_fault_lookup
    )
    rups = rup_df_to_rupture_dicts(
        fault_rup_df, mag_col='mag', displacement_col='displacement'
    )

    if moment_rate is None:
        moment_rate = (
            fault['surface'].get_area()
            * fault['net_slip_rate']
            * SHEAR_MODULUS
            * 1e3
            * seismic_fraction
        )

    if mfd is None:
        mfd = make_fault_mfd(
            fault,
            max_mag=fault_rup_df.mag.max(),
            min_mag=4.0,
            corner_mag=corner_mag,
            seismic_fraction=seismic_fraction,
            mfd_type=mfd_type,
            b_val=b_val,
            moment_rate=moment_rate,
        )
    rup_rates = set_single_fault_rupture_rates_by_mfd(
        rups,
        mfd,
        scale_moment_rate=scale_moment_rate,
        moment_rate=moment_rate,
        faults_or_subfaults=faults_or_subfaults,
    )

    return rup_rates


def _get_surface_moment_rate(
    surface_area: float,
    slip_rate: float,
    seismic_fraction: float = 1.0,
    shear_modulus: float = SHEAR_MODULUS,
) -> float:
    return surface_area * slip_rate * shear_modulus * 1e3 * seismic_fraction


def get_fault_moment_rate(
    fault, seismic_fraction=1.0, shear_modulus=SHEAR_MODULUS
):
    return _get_surface_moment_rate(
        fault['surface'].get_area(),
        fault['net_slip_rate'],
        seismic_fraction=seismic_fraction,
        shear_modulus=shear_modulus,
    )


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


def get_ruptures_on_fault_df(fault_id, rupture_df, key='faults'):
    """
    Gets all ruptures on a given fault or subfault, indicated by the fault_id.
    Pass `key='subfaults'` to get subfaults.
    """
    return rupture_df[rupture_df[key].apply(lambda x: fault_id in x)]


def get_ruptures_on_fault(fault_id, rupture_df, rup_fault_lookup):
    rups = rup_fault_lookup[fault_id]
    rup_df = rupture_df.loc[rups]
    return rup_df


def make_rup_fault_lookup(ruptures, key='subfaults'):
    """
    Makes a dictionary with fault_ids as keys and values the lists
    of the rupture idxs (not rupture count ids) of ruptures on that fault.

    Pass `key='subfaults'` to get subfaults (defauklty).
    """
    if isinstance(ruptures, pd.DataFrame):
        rup_fault_dict = ruptures[key].to_dict()
    else:
        # always 'faults'
        rup_fault_dict = {rup['idx']: rup['faults'] for rup in ruptures}

    fault_rup_dict = {}
    for rup, faults in rup_fault_dict.items():
        for fault in faults:
            if fault not in fault_rup_dict:
                fault_rup_dict[fault] = []
            fault_rup_dict[fault].append(rup)

    return fault_rup_dict


def get_fault_mfd_from_rup_rates(
    fault_idx, rup_df, rup_rates, rup_fault_lookup=None, fault_key='faults'
):
    """
    Calculates the MFD of a fault or subfault given all the ruptures that occur
    on it and their rupture rates from the inversion solution (or other
    solution).

    Parameters
    ----------
    fault_idx: str or int
        Index of fault or subfault
    rup_df: pd.DataFrame
        Dataframe of ruptures
    rup_rates: pd.Series
        Annual occurrence rates of ruptures with index shared with rup_df
    rup_fault_lookup: dict, optional
        Lookup table with keys of fault indices and values of lists of
        ruptures on each.
    fault_key: str
        `faults` or `subfaults` depending on interest

    Returns
    -------
    mfd_sort: dict
        Incremental MFD in dictionary form, with magnitudes as keys and
        rates as values
    """
    rup_df_use = rup_df.loc[rup_rates.index]
    if rup_fault_lookup is None:
        rup_fault_lookup = make_rup_fault_lookup(rup_df_use, key=fault_key)

    rups_on_fault = rup_fault_lookup[fault_idx]

    mfd = AccumDict()
    for rup_idx in rups_on_fault:
        rup = rup_df.loc[rup_idx]
        mfd += {rup['mag']: rup_rates[rup_idx]}

    mfd_sort = {k: mfd[k] for k in sorted(mfd.keys())}

    return mfd_sort


def get_rup_rates_from_fault_slip_rates(
    fault_network,
    b_val=1.0,
    corner_mag=7.5,
    mfd_type='TruncatedGRMFD',
    plot_fault_moment_rates=False,
    seismic_fraction=1.0,
    rupture_set_for_rates_from_slip_rates='all',
    faults_or_subfaults='subfaults',
    export_fault_mfds=False,
    exit_after_mfd_export=False,
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
    else:
        raise ValueError(
            "rupture_set_for_rates_from_slip_rates must be 'filtered' or 'all'"
            + f", not '{rupture_set_for_rates_from_slip_rates}'",
        )

    if faults_or_subfaults == 'faults':
        _key_ = 'faults'
        fault_iterator = {
            fault['fid']: fault for fault in fault_network['faults']
        }
    elif faults_or_subfaults == 'subfaults':
        _key_ = 'subfaults'
        fault_iterator = {
            sub_idx: fault
            for sub_idx, fault in fault_network['subfault_df'].iterrows()
        }
    else:
        raise ValueError(
            "faults_or_subfaults must be 'faults' or 'subfaults', not"
            + f"{faults_or_subfaults}"
        )

    rup_fault_lookup = make_rup_fault_lookup(fault_network[rup_df_key], _key_)

    logging.debug("getting moment rates")
    fault_moment_rates = {
        id: get_fault_moment_rate(
            fault,
            seismic_fraction=fault.get("seismic_fraction", seismic_fraction),
        )
        for id, fault in fault_iterator.items()
    }

    logging.debug("making mfds")
    fault_mfds = {
        id: make_fault_mfd(
            fault,
            max_mag=get_ruptures_on_fault(
                id, fault_network[rup_df_key], rup_fault_lookup
            ).mag.max(),
            mfd_type=mfd_type,
            b_val=b_val,
            corner_mag=corner_mag,
            seismic_fraction=fault.get("seismic_fraction", seismic_fraction),
            moment_rate=fault_moment_rates[id],
            **kwargs,
        )
        for id, fault in fault_iterator.items()
    }
    if export_fault_mfds:
        fault_network['fault_mfds'] = fault_mfds
        if exit_after_mfd_export:
            return

    logging.debug("setting single-fault rup rates")
    all_rup_rates = {
        id: set_single_fault_rup_rates(
            id,
            fault_network,
            rup_fault_lookup,
            mfd=fault_mfds[id],
            rup_df=rup_df_key,
            b_val=b_val,
            mfd_type=mfd_type,
            seismic_fraction=fault.get("seismic_fraction", seismic_fraction),
            faults_or_subfaults=faults_or_subfaults,
            **kwargs,
        )
        for id, fault in fault_iterator.items()
    }

    sf_inds = []
    if faults_or_subfaults == 'faults':
        sf_inds = fault_network['single_rup_df'].index
    elif faults_or_subfaults == 'subfaults':
        for ind, sfs in fault_network[rup_df_key].subfaults.items():
            if len(sfs) == 1:
                sf_inds.append(ind)

    logging.debug("doing final rup rates 1")
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

    logging.debug("doing final rup rates 2")
    mf_rup_rates = {}
    for rup, rates in mf_rates.items():
        if faults_or_subfaults == 'faults':
            faults, fault_fracs = fault_network[rup_df_key].loc[
                rup, ['faults', 'fault_frac_area']
            ]
        elif faults_or_subfaults == 'subfaults':
            faults, fault_fracs = fault_network[rup_df_key].loc[
                rup, ['subfaults', 'frac_area']
            ]
        fault_weights = fault_fracs
        if sum(fault_fracs) == 0.0:
            logging.warning(f"rupture {rup} has zero fault fraction")
            # need to handle this better probably
        fault_rates = [rates[flt] for flt in faults]
        weighted_mean_rate = weighted_mean(fault_rates, fault_weights)
        mf_rup_rates[rup] = weighted_mean_rate

    logging.debug("concatting rates")
    final_rup_rates = pd.concat(
        (pd.Series(final_rup_rates), pd.Series(mf_rup_rates))
    )

    # just to check moment rates for faults
    if plot_fault_moment_rates:
        import matplotlib.pyplot as plt

        rups = rup_df_to_rupture_dicts(fault_network[rup_df_key])
        fault_moment_rates_rup = {}
        if faults_or_subfaults == 'faults':
            frac_key = 'faults_orig'
        elif faults_or_subfaults == 'subfaults':
            frac_key = 'subfault_fracs'
        for rup in rups:
            for fault in rup[frac_key]:
                rup_moment_rate = (
                    rup[frac_key][fault]
                    * mag_to_mo(rup['M'])
                    * final_rup_rates[rup['idx']]
                )
                if fault not in fault_moment_rates_rup:
                    fault_moment_rates_rup[fault] = rup_moment_rate
                else:
                    fault_moment_rates_rup[fault] += rup_moment_rate

        plt.plot(
            [0, max(fault_moment_rates.values())],
            [0, max(fault_moment_rates.values())],
            '--',
            lw=0.25,
        )
        plt.plot(
            fault_moment_rates.values(),
            [
                fault_moment_rates_rup[fault]
                for fault in fault_moment_rates.keys()
            ],
            '.',
        )
        plt.xlabel("Fault moment rate from slip rates")
        plt.ylabel("Fault moment rate from ruptures")
        plt.title("Fault moment rates from slip rate and rupture rates")
        plt.show()

    return final_rup_rates


def get_earthquake_fault_distances(eqs, faults, dist: Optional[float] = None):
    eq_mesh = Mesh(eqs.longitude.values, eqs.latitude.values, eqs.depth.values)
    dist_df = np.zeros((len(eqs), len(faults)))

    for i, fault in enumerate(faults):
        dist_df[:, i] = fault['surface'].get_min_distance(eq_mesh)

    dist_df = pd.DataFrame(
        data=dist_df, columns=[f['fid'] for f in faults], index=eqs.index
    )
    dist_df_min_vals = dist_df.min(axis=1)

    eqs['fault_dist'] = dist_df_min_vals

    if dist is not None:
        eqs = eqs.loc[(eqs['fault_dist'] <= dist)]

    return eqs


def get_on_fault_likelihood(
    mag,
    distance,
    year,
    ref_mag=6.0,
    mag_decay_factor=1.5,
    ref_year=2024.0,
    time_decay_factor=0.02,
    base_distance_decay=0.05,
):

    time_diff = ref_year - year

    mag_diff = mag - ref_mag
    if np.isscalar(mag):
        if mag_diff < 0.0:
            mag_diff = 0.0
    else:
        mag_diff[mag_diff < 0.0] = 0.0

    decay_constant = base_distance_decay / (
        1 + time_decay_factor * time_diff + mag_decay_factor * mag_diff
    )
    on_fault_likelihood = np.exp(-decay_constant * distance)

    return on_fault_likelihood


def get_soln_slip_rates(soln, lhs, n_slip_rates, units="mm/yr"):
    if units == "mm/yr":
        coeff = 1e3
    elif units == "m/yr":
        coeff = 1.0

    pred_slip_rates = lhs.dot(soln)[:n_slip_rates] * coeff
    return pred_slip_rates


def point_to_triangle_distance(point, triangle_vertices):
    """
    Calculate the minimum distance between a point and a triangle in 3D space.

    Parameters:
    -----------
    point : numpy.ndarray
        3D coordinates of the point [x, y, z]
    triangle_vertices : numpy.ndarray
        3x3 array containing the coordinates of triangle vertices
        [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]

    Returns:
    --------
    float
        Minimum distance from point to triangle
    numpy.ndarray
        Closest point on the triangle
    """
    # Extract triangle vertices
    v1, v2, v3 = triangle_vertices

    # Calculate triangle normal
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)

    # Calculate point's projection onto triangle's plane
    v1_to_point = point - v1
    dist_to_plane = np.dot(v1_to_point, normal)
    projection = point - dist_to_plane * normal

    # Check if projection lies inside triangle using barycentric coordinates
    # Compute vectors for barycentric coordinate calculation
    v0 = v2 - v1
    v1_vec = v3 - v1
    v2_vec = projection - v1

    # Compute dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1_vec)
    d11 = np.dot(v1_vec, v1_vec)
    d20 = np.dot(v2_vec, v0)
    d21 = np.dot(v2_vec, v1_vec)

    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # If projection is inside triangle, return distance to plane
    if (u >= 0) and (v >= 0) and (w >= 0) and (abs(u + v + w - 1.0) < 1e-10):
        return abs(dist_to_plane), projection

    # If projection is outside triangle, find closest point on edges
    def point_to_line_segment(p, v1, v2):
        """Calculate minimum distance between point p and line segment v1-v2"""
        segment = v2 - v1
        length_sq = np.dot(segment, segment)
        if length_sq == 0:
            return np.linalg.norm(p - v1), v1

        t = max(0, min(1, np.dot(p - v1, segment) / length_sq))
        projection = v1 + t * segment
        return np.linalg.norm(p - projection), projection

    # Check each edge of the triangle
    d1, p1 = point_to_line_segment(point, v1, v2)
    d2, p2 = point_to_line_segment(point, v2, v3)
    d3, p3 = point_to_line_segment(point, v3, v1)

    # Return minimum distance and closest point
    min_dist = min(d1, d2, d3)
    if d1 == min_dist:
        return d1, p1
    elif d2 == min_dist:
        return d2, p2
    else:
        return d3, p3


def calculate_tri_mesh_distances(points, triangles, verbose=True):
    """
    Calculate minimum distances between multiple points and a triangular mesh.

    Parameters:
    -----------
    points : numpy.ndarray
        Nx3 array of point coordinates [[x1,y1,z1], [x2,y2,z2], ...]
    triangles : numpy.ndarray
        Mx3x3 array of triangle vertices
        [[[x11,y11,z11], [x12,y12,z12], [x13,y13,z13]], ...]

    Returns:
    --------
    numpy.ndarray
        Array of minimum distances for each point
    numpy.ndarray
        Array of indices of closest triangles for each point
    """
    n_points = len(points)
    n_triangles = len(triangles)

    distances = np.full(n_points, np.inf)
    closest_triangles = np.full(n_points, -1)

    n_digits = len(str(n_points))

    for i, point in enumerate(points):
        if verbose:
            print(
                "working on ",
                str(i).zfill(n_digits),
                f"/ {n_points}",
                end="\r",
            )
        min_dist = np.inf
        closest_triangle = -1

        for j, triangle in enumerate(triangles):
            dist, _ = point_to_triangle_distance(point, triangle)
            if dist < min_dist:
                min_dist = dist
                closest_triangle = j

        distances[i] = min_dist
        closest_triangles[i] = closest_triangle

    return distances, closest_triangles


def rescale_mfd(mfd, frac):
    return {
        mag: rate * frac for mag, rate in mfd.get_annual_occurrence_rates()
    }
