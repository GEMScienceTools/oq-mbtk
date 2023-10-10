import json

import toml
import numpy as np
import pandas as pd

from openquake.fnm.fault_system import get_rups_fsys
from openquake.fnm.importer import kite_surfaces_from_geojson

from .utils import get_rupture_displacement, weighted_mean


def build_subsec_fault_indices(fsys):
    subsec_start = 0
    inds = {}
    for i, (surf, subsecs) in enumerate(fsys):
        n_secs = len(subsecs[0])
        subsec_stop = subsec_start + n_secs - 1
        inds[i] = (subsec_start, subsec_stop)
        subsec_start = subsec_stop + 1

    assert inds[i][1] + 1 == sum([f[1].shape[1] for f in fsys])

    return inds


def get_fault(idx, subsec_inds):
    for fault, (start, stop) in subsec_inds.items():
        if start <= idx <= stop:
            return fault


def get_fault_property_for_subsec(subsec_idx, prop, faults, subsec_inds):
    fault_idx = get_fault(subsec_idx, subsec_inds)
    props = faults[fault_idx]["properties"]
    return props[prop]


def get_fault_property_for_rup(i, prop, single_sec_rups, faults, f_idx=6):
    fault_idx = single_sec_rups[i, f_idx]
    props = faults[fault_idx]["properties"]
    return props[prop]


def slip_vector_azimuth(strike, dip, rake):
    # Convert degrees to radians
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    rake_rad = np.radians(rake) * -1

    # Calculate the 3D Cartesian coordinates of the slip vector
    slip_x = -np.sin(rake_rad) * np.sin(strike_rad) - np.cos(rake_rad) * np.sin(
        dip_rad
    ) * np.cos(strike_rad)
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


def match_rup_with_subsecs(rup, subsecs, subsec_start_index=0):
    matching_subsecs = []
    rup_ul_row, rup_ul_col, rup_n_cols, rup_n_rows = rup[0:4]
    rup_lr_row, rup_lr_col = rup_ul_row + rup_n_rows, rup_ul_col + rup_n_cols

    for i, subsec in enumerate(subsecs):
        subsec_ul_row, subsec_ul_col, subsec_n_cols, subsec_n_rows = subsec
        subsec_lr_row, subsec_lr_col = (
            subsec_ul_row + subsec_n_rows,
            subsec_ul_col + subsec_n_cols,
        )

        # Check if subsec is contained within rup
        if (
            rup_ul_row <= subsec_ul_row < rup_lr_row
            and rup_ul_col <= subsec_ul_col < rup_lr_col
            and rup_ul_row < subsec_lr_row <= rup_lr_row
            and rup_ul_col < subsec_lr_col <= rup_lr_col
        ):
            matching_subsecs.append(i + subsec_start_index)

    return matching_subsecs


def match_rups_with_subsecs(single_sec_rups, fault_sys):
    subsec_start_indices = build_subsec_fault_indices(fault_sys)

    rup_matches = []
    for rup in single_sec_rups:
        f_idx = rup[6]
        fault, subsecs = fault_sys[f_idx]
        rup_match = match_rup_with_subsecs(
            rup, subsecs[0], subsec_start_index=subsec_start_indices[f_idx][0]
        )
        rup_matches.append(rup_match)
    return rup_matches


def get_subsections_for_all_rups(rupture_sections, rupture_subsections):
    all_rup_subsections = []
    for rup in rupture_sections:
        rup_subsections = []
        for sec in rup:
            rup_subsections.extend(rupture_subsections[sec])
        all_rup_subsections.append(rup_subsections)
    return all_rup_subsections


def build_subsection_df(
    fsys,
    faults,
    slip_rate_key="net_slip_rate",
    slip_rate_err_key="net_slip_rate_err",
    rake_key="rake",
    rake_err_key="rake_err",
):
    # strike and dip will be calculated from the surface of each subsection
    subsec_strikes = []
    subsec_dips = []
    for fault, subsecs in fsys:
        # fault_mesh = fault.mesh
        for subsec in subsecs[0]:
            # subsec_mesh = get_subsection(fault_mesh, subsec)
            # subsec_surface =
            strike = fault.get_strike()
            dip = fault.get_dip()

            subsec_strikes.append(strike)
            subsec_dips.append(dip)

    subsec_inds = build_subsec_fault_indices(fsys)

    n_subsecs = len(subsec_strikes)

    df = pd.DataFrame(index=np.arange(n_subsecs))
    df.index.rename("id", inplace=True)

    df["fault"] = [get_fault(i, subsec_inds) for i in range(n_subsecs)]

    df["slip_rate"] = [
        get_fault_property_for_subsec(i, slip_rate_key, faults, subsec_inds)
        for i in range(n_subsecs)
    ]

    if slip_rate_err_key is not None:
        df["slip_rate_err"] = [
            get_fault_property_for_subsec(
                i, slip_rate_err_key, faults, subsec_inds
            )
            for i in range(n_subsecs)
        ]

    df["rake"] = [
        get_fault_property_for_subsec(i, rake_key, faults, subsec_inds)
        for i in range(n_subsecs)
    ]

    if rake_err_key is not None:
        df["rake_err"] = [
            get_fault_property_for_subsec(i, rake_err_key, faults, subsec_inds)
            for i in range(n_subsecs)
        ]

    df["strike"] = subsec_strikes
    df["dip"] = subsec_dips

    df["slip_azimuth"] = [
        slip_vector_azimuth(*params)
        for params in zip(df.strike.values, df.dip.values, df.rake.values)
    ]

    return df


def build_rupture_dataframe(
    rup_sub_sections=None,
    magnitudes=None,
    areas=None,
    frac_areas=None,
    faults=None,
    single_sec_rups=None,
    connection_distances=None,
    connection_angles=None,
    subsection_df=None,
    fault_system=None,
    fault_idx=6,
):
    df = pd.DataFrame(
        {
            "single_ruptures": rup_sub_sections,
            "M": magnitudes,
            "area": areas,
            "frac_areas": frac_areas,
        }
    )

    # get the displacement
    df["D"] = get_rupture_displacement(df["M"], df["area"])

    rup_rakes = [
        [
            get_fault_property_for_rup(i, "rake", single_sec_rups, faults)
            for i in rss
        ]
        for rss in rup_sub_sections
    ]

    df["rake"] = [
        weighted_mean(rup_rakes[i], rup.frac_areas)
        for i, rup in enumerate(df.itertuples())
    ]

    if connection_angles is not None:
        df["connection_angles"] = connection_angles

    if connection_distances is not None:
        df["connection_distances"] = connection_distances

    df["faults"] = [
        [single_sec_rups[i, fault_idx] for i in rss] for rss in rup_sub_sections
    ]

    single_sec_subsections = match_rups_with_subsecs(
        single_sec_rups, fault_system
    )

    df["subsections"] = get_subsections_for_all_rups(
        df["single_ruptures"], single_sec_subsections
    )

    df["slip_azimuths"] = [
        [subsection_df.loc[ss, "slip_azimuth"] for ss in subsecs]
        for subsecs in df["subsections"].values
    ]

    return df


def build_info_from_faults(
    fault_geojson_file,
    settings=None,
    settings_file=None,
    slip_rate_key="net_slip_rate",
    slip_rate_err_key="net_slip_rate_err",
    rake_key="rake",
    rake_err_key="rake_err",
):
    if settings is None:
        if settings_file is None:
            raise ValueError(
                "Either settings or settings_file must be provided"
            )
        else:
            with open(settings_file) as f:
                settings = toml.load(f)

    with open(fault_geojson_file) as f:
        fgj = json.load(f)
        faults = fgj["features"]

    surfaces = kite_surfaces_from_geojson(fault_geojson_file)

    rup_fault_data = get_rups_fsys(surfaces, settings)

    subsection_df = build_subsection_df(
        rup_fault_data["fault_system"],
        faults,
        slip_rate_key,
        slip_rate_err_key,
        rake_key,
        rake_err_key,
    )

    rupture_df = build_rupture_dataframe(
        rup_sub_sections=rup_fault_data["ruptures_single_section_indexes"],
        magnitudes=rup_fault_data["magnitudes"],
        areas=rup_fault_data["areas"],
        frac_areas=rup_fault_data["rupture_fractional_area"],
        faults=faults,
        single_sec_rups=rup_fault_data["ruptures_single_section"],
        connection_angles=rup_fault_data["ruptures_connection_angles"],
        connection_distances=rup_fault_data["ruptures_connection_distances"],
        fault_system=rup_fault_data["fault_system"],
        subsection_df=subsection_df,
    )

    return rup_fault_data, subsection_df, rupture_df
