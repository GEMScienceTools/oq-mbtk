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

import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Polygon, MultiPolygon

from openquake.baselib.general import AccumDict
from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.mesh import RectangularMesh
from openquake.hazardlib.geo.surface import SimpleFaultSurface

from openquake.fnm.importer import (
    simple_fault_surface_from_feature,
)

from openquake.fnm.msr import area_to_mag

from openquake.fnm.inversion.utils import (
    get_rupture_displacement,
    SHEAR_MODULUS,
    slip_vector_azimuth,
)

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def simple_fault_from_feature(
    feature: dict,
    edge_sd: float = 5.0,
    lsd_default: float = 20.0,
    usd_default: float = 0.0,
) -> dict:
    """
    Builds a fault (a dictionary with the required parameters for the
    seismic source model, including a SimpeFaultSurface) from a GeoJSON
    feature.

    Parameters
    ----------
    feature : dict
        GeoJSON feature containing fault parameters. Required parameters
        are `fid` (the fault ID), `net_slip_rate`, `net_slip_rate_err`,
        and `rake`. The `geometry` key must contain a `LineString`
        representing the fault trace.
    edge_sd : float
        Edge sampling distance in km.
    lsd_default : float
        Lower seismogenic depth in km.
    usd_default : float
        Upper seismogenic depth in km.

    Returns
    -------
    fault : dict
        Dictionary containing fault parameters, formatted for further use
        in Fermi.
    """
    props_to_keep = [
        "fid",
        "net_slip_rate",
        "net_slip_rate_err",
        "rake",
    ]
    optional_props_to_keep = [
        "lsd",
        "rake_err",
        "usd",
        "seismic_fraction",
    ]

    fault = {prop: feature['properties'][prop] for prop in props_to_keep}
    for prop in optional_props_to_keep:
        if prop in feature['properties']:
            fault[prop] = feature['properties'][prop]
                    
    if fault['rake'] == -180.0:
        fault['rake'] = 180.0

    fault['surface'] = simple_fault_surface_from_feature(
        feature,
        edge_sd=edge_sd,
        lsd_default=fault.get("lsd", lsd_default),
        usd_default=fault.get("usd", usd_default),
    )

    fault['trace'] = feature['geometry']['coordinates']

    return fault


def get_trace_from_mesh(mesh):
    """
    Builds a fault trace from a mesh.

    Parameters
    ----------
    mesh : openquake.hazardlib.geo.mesh.Mesh
        Mesh to use for trace.

    Returns
    -------
    trace : openquake.hazardlib.geo.Line
        Fault trace.
    """
    trace = Line(
        [
            Point(lon, mesh.lats[0, i], mesh.depths[0, i])
            for i, lon in enumerate(mesh.lons[0])
        ]
    )

    return trace


def subdivide_simple_fault_surface(
    fault_surface: SimpleFaultSurface,
    subsection_size=[
        15.0,
        15.0,
    ],
    edge_sd=5.0,
    dip_sd=5.0,
    dip=None,
):
    """
    Divides a fault surface into subsections of equal size,
    as close to the specified parameters as possible.

    Parameters
    ----------
    fault_surface : SimpleFaultSurface
        Surface to divide into subsections.
    subsection_size : list of float or integers.
        Size of subsections in km. If negative, the number of subsections
        along strike (dip) is given by the absolute value of the number.
    edge_sd : float
        Edge (along-strike) sampling distance in km.
    dip_sd : float
        Down-dip sampling distance in km.
    dip : float
        Dip of the fault surface in degrees.

    Returns
    -------
    subsec_meshes : list of SimpleFaultSurface
        List of subsections.
    """

    fault_mesh = fault_surface.mesh
    fault_trace = get_trace_from_mesh(fault_mesh)

    # get basic geometric info
    if dip is None:
        dip = fault_surface.get_dip()
    strike = fault_trace[0].azimuth(fault_trace[-1])

    fault_length = fault_trace.get_length()
    fault_width = fault_surface.get_width()

    if subsection_size[1] > 0:
        subsec_width_init = subsection_size[1]
    elif subsection_size[1] < 0:
        assert (
            subsection_size[1] % 1 == 0.0
        ), "Negative down-dip number of sections must be integer"
        subsec_width_init = fault_width / abs(subsection_size[1])

    if subsection_size[0] > 0:
        subsec_length_init = subsection_size[0]
    elif subsection_size[0] < 0:
        subsec_length_init = fault_length / abs(subsection_size[0])

    # calculate number of segments and point spacing along strike
    num_segs_along_strike = max(
        int(round(fault_length / subsec_length_init)), 1
    )
    subsec_length = fault_length / num_segs_along_strike
    pt_spacing = subsec_length / round(subsec_length / edge_sd)
    n_pts_strike = fault_length / pt_spacing + 1
    assert (n_pts_strike % 1 <= 1e-5) or (n_pts_strike % 1 >= (1.0 - 1e-5)), (
        "Resampled trace not integer length: " + f"{n_pts_strike}"
    )
    n_pts_strike = max(int(round(n_pts_strike)), 2)

    # resample fault trace and calculate number of points along strike
    new_trace = fault_trace.resample_to_num_points(n_pts_strike)
    assert new_trace.coo.shape[0] == n_pts_strike, (
        "Resampled trace not correct length: "
        + f"{new_trace.coo.shape[0]} != {n_pts_strike}"
    )

    n_subsec_pts_strike = ((n_pts_strike - 1) / num_segs_along_strike) + 1

    assert n_subsec_pts_strike % 1 == 0.0, (
        "Resampled trace not dividing equally among subsegments: "
        + f"{n_subsec_pts_strike}"
    )

    n_subsec_pts_strike = int(n_subsec_pts_strike)

    # calculate number of segments and point spacing down dip
    num_segs_down_dip = max(int(round(fault_width / subsec_width_init)), 1)
    subsec_width = fault_width / num_segs_down_dip
    if round(subsec_width / dip_sd) > 0:
        dip_pt_spacing = subsec_width / round(subsec_width / dip_sd)
    else:  # when width is much larger than dip_sd
        dip_pt_spacing = subsec_width

    azimuth = (strike + 90) % 360

    mesh = []

    hor_dip_spacing = dip_pt_spacing * np.cos(np.radians(dip))
    vert_dip_spacing = dip_pt_spacing * np.sin(np.radians(dip))
    n_level_sets = max(int(round(fault_width / dip_pt_spacing)) + 1, 2)

    for i in range(n_level_sets):
        level_mesh = [
            p.point_at(hor_dip_spacing * i, vert_dip_spacing * i, azimuth)
            for p in new_trace
        ]
        mesh.append(level_mesh)

    surface_points = np.array(mesh).tolist()
    resampled_mesh = RectangularMesh.from_points_list(surface_points)

    n_pts_dip = resampled_mesh.lons.shape[0]
    n_subsec_pts_dip = ((n_pts_dip - 1) / num_segs_down_dip) + 1
    assert n_subsec_pts_dip % 1 == 0.0, (
        "Resampled mesh not dividing equally among subsegments down-dip: "
        + f"{n_pts_dip}, {num_segs_along_strike}"
    )
    n_subsec_pts_dip = int(n_subsec_pts_dip)

    subsec_meshes = subdivide_rupture_mesh(
        resampled_mesh.lons,
        resampled_mesh.lats,
        resampled_mesh.depths,
        num_segs_down_dip,
        num_segs_along_strike,
        n_subsec_pts_dip,
        n_subsec_pts_strike,
    )

    return subsec_meshes


def subdivide_rupture_mesh(
    lons: np.ndarray,
    lats: np.ndarray,
    depths: np.ndarray,
    num_segs_down_dip: int,
    num_segs_along_strike: int,
    n_subsec_pts_dip: int,
    n_subsec_pts_strike: int,
):
    """
    Breaks a mesh (represented by arrays of lons, lats, and depths) into
    subsections of equal size.

    Parameters
    ----------
    lons : np.ndarray
        Array of longitudes.
    lats : np.ndarray
        Array of latitudes.
    depths : np.ndarray
        Array of depths.
    num_segs_down_dip : int
        Number of subsections down dip.
    num_segs_along_strike : int
        Number of subsections along strike.
    n_subsec_pts_dip : int
        Number of points in each subsection down dip.
    n_subsec_pts_strike : int
        Number of points in each subsection along strike.

    Returns
    -------
    subsec_meshes : list of RectangularMesh
        List of subsection meshes.
    """
    assert (
        lons.shape == lats.shape == depths.shape
    ), "Lons, lats, and depths must have the same shape"

    assert (
        n_subsec_pts_dip == ((lons.shape[0] - 1) / num_segs_down_dip) + 1
    ), "Mesh does not divide equally among subsegments down-dip"

    assert (
        n_subsec_pts_strike
        == ((lons.shape[1] - 1) / num_segs_along_strike) + 1
    ), "Mesh does not divide equally among subsegments along-strike"

    subsec_meshes = []

    i_start = 0
    for i in range(num_segs_down_dip):
        j_start = 0
        i_end = i_start + n_subsec_pts_dip
        for j in range(num_segs_along_strike):
            j_end = j_start + n_subsec_pts_strike

            subsec_lons = lons[i_start:i_end, j_start:j_end]
            subsec_lats = lats[i_start:i_end, j_start:j_end]
            subsec_depths = depths[i_start:i_end, j_start:j_end]

            try:
                subsec_mesh = RectangularMesh(
                    subsec_lons, subsec_lats, subsec_depths
                )
                subsec_meshes.append({'row': i, 'col': j, 'mesh': subsec_mesh})
            except:
                print(i_start, i_end, j_start, j_end, i, j)

            j_start += n_subsec_pts_strike - 1
        i_start += n_subsec_pts_dip - 1

    return subsec_meshes


def get_subsections_from_fault(
    fault: dict,
    subsection_size=[
        15.0,
        15.0,
    ],
    edge_sd=2.0,
    dip_sd=2.0,
    surface=None,
    surface_type="simple_fault_surface",
) -> list[dict]:
    """
    Divides a fault (represented as a dictionary of parameters) and an
    OpenQuake SimpleFaultSurface into subsections of close to equal size.

    Parameters
    ----------
    fault : dict
        Dictionary containing fault parameters. Required parameters are `fid`
        (the fault ID), `net_slip_rate`, `net_slip_rate_err`, and `rake`.
    subsection_size : list of float or integers.
        Size of subsections in km. If negative, the number of subsections
        along strike (dip) is given by the absolute value of the number.
    edge_sd : float
        Edge (along-strike) sampling distance in km.
    dip_sd : float
        Down-dip sampling distance in km.
    surface : SimpleFaultSurface
        Surface to use for subsectioning.
    surface_type : str
        Type of surface to use for subsectioning. Currently,
        "simple_fault_surface" is the only option, though "kite_surface"
        will be supported in the future.

    Returns
    -------
    subsections : list of dicts
        List of dictionaries containing information about each subsection.
    """

    props_to_keep = [
        "fid",
        "net_slip_rate",
        "net_slip_rate_err",
        "rake",
    ]
    optional_props_to_keep = [
        "rake_err",
        "seismic_fraction",
    ]

    if np.isscalar(subsection_size):  # len(subsection_size) == 1:
        # or should we raise an error?
        subsection_size = [subsection_size, subsection_size]

    subsections = []
    subsec_meshes = subdivide_simple_fault_surface(
        surface,
        subsection_size=subsection_size,
        edge_sd=edge_sd,
        dip_sd=dip_sd,
    )

    for i, sub_mesh in enumerate(subsec_meshes):
        mesh = sub_mesh['mesh']
        subfault = {prop: fault[prop] for prop in props_to_keep}
        for prop in optional_props_to_keep:
            if prop in fault:
                subfault[prop] = fault[prop]

        subfault['fault_position'] = (sub_mesh['row'], sub_mesh['col'])
        subfault["trace"] = [
            [lon, mesh.lats[0, i], mesh.depths[0, i]]
            for i, lon in enumerate(mesh.lons[0])
        ]
        if surface_type == "simple_fault_surface":
            subfault["surface"] = SimpleFaultSurface(mesh)
        elif surface_type == "kite_surface":
            raise NotImplementedError("Kite surface not currently supported")

        subfault['length'] = Line(
            [Point(*p) for p in subfault['trace']]
        ).get_length()
        subfault['width'] = subfault['surface'].get_width()
        subfault["area"] = subfault["surface"].get_area()
        subfault["strike"] = subfault["surface"].get_strike()
        subfault["dip"] = subfault["surface"].get_dip()
        subfault["subsec_id"] = i

        subsections.append(subfault)

    return subsections


def make_subfault_df(all_subfaults):
    """
    Makes a Pandas DataFrame for each subfault (subsection) in
    the fault network.

    Parameters
    ----------
    all_subfaults : list of dicts
        List of dictionaries containing information about each subfault.
        See `get_subsections_from_fault` for more information on the format.

    Returns
    -------
    subfault_df : pd.DataFrame
        DataFrame containing information about each subfault.
    """
    subfault_df = pd.concat(pd.DataFrame(sf) for sf in all_subfaults)
    subfault_df = subfault_df.reset_index(drop=True)
    subfault_df.index.name = "subfault_id"
    subfault_df['slip_azimuth'] = [
        slip_vector_azimuth(*params)
        for params in zip(
            subfault_df.strike.values,
            subfault_df.dip.values,
            subfault_df.rake.values,
        )
    ]

    return subfault_df


def group_subfaults_by_fault(subfaults: list[dict]) -> dict:
    """
    Creates a dictionary with all of the subfaults from each fault as
    values, with the fault ID as the key.

    Parameters
    ----------
    subfaults : list of dicts

    Returns

    """
    subfault_dict = {
        fault_group[0]['fid']: fault_group for fault_group in subfaults
    }

    return subfault_dict


def angular_mean_degrees(angles) -> float:
    """
    Calculates the angular mean of a list/array of angles in degrees.

    Parameters
    ----------
    angles : list or array of floats
        Angles in degrees.

    Returns
    -------
    mean_angle : float
        Angular mean in degrees.
    """
    mean_angle = np.arctan2(
        np.mean(np.sin(np.radians(angles))),
        np.mean(np.cos(np.radians(angles))),
    )
    return np.degrees(mean_angle)


def weighted_angular_mean_degrees(angles, weights):
    """
    Calculates the weighted angular mean of a list/array of angles in degrees.

    Parameters
    ----------
    angles : list or array of floats
        Angles in degrees.
    weights : list or array of floats
        Weights for each angle.

    Returns
    -------
    mean_angle : float
        Weighted angular mean in degrees.
    """
    mean_angle = np.arctan2(
        np.sum(weights * np.sin(np.radians(angles))),
        np.sum(weights * np.cos(np.radians(angles))),
    )
    return np.degrees(mean_angle)


def make_rupture_df(
    single_fault_rup_df: pd.DataFrame,
    multi_fault_rups,
    subfault_df,
    area_mag_msr='Leonard2014_Interplate',
    mag_decimals=1,
) -> pd.DataFrame:
    """
    Makes a Pandas DataFrame, with a row for each rupture in the fault network.

    Parameters
    ----------
    single_fault_rup_df : pd.DataFrame
        DataFrame containing information about each single-fault rupture.
        See `get_single_fault_ruptures` for more information on the format.
    multi_fault_rups : list of lists
        List of lists of subfault indices for each multi-fault rupture.
    subfault_df : pd.DataFrame
        DataFrame containing information about each subfault.
        See `make_subfault_df` for more information on the format.
    area_mag_msr : str
        Area-to-magnitude scaling relationship to use. Must
        be in the `openquake.fnm.msr` library of scaling relationships.

    Returns
    -------
    rupture_df : pd.DataFrame
        DataFrame containing information about each rupture.
    """
    rups_involved = [[int(r)] for r in single_fault_rup_df.index.values]
    rupture_df = single_fault_rup_df[['subfaults']]

    rupture_df = pd.DataFrame(
        index=rupture_df.index,
        data={
            'subfaults': single_fault_rup_df.subfaults,
            'ruptures': rups_involved,
            'faults': [[fault] for fault in single_fault_rup_df.fault],
        },
    )

    srup_lookup = {i: row.subfaults for i, row in rupture_df.iterrows()}
    area_lookup = {i: row.area for i, row in subfault_df.iterrows()}
    rake_lookup = {i: row.rake for i, row in subfault_df.iterrows()}
    slip_az_lookup = {i: row.slip_azimuth for i, row in subfault_df.iterrows()}
    fault_lookup = {i: row.faults[0] for i, row in rupture_df.iterrows()}
    sub_fid_lookup = {i: row.fid for i, row in subfault_df.iterrows()}

    sf_rup_azimuths = {}
    for row in single_fault_rup_df.itertuples():
        row_slip_azimuths = [slip_az_lookup[sf] for sf in row.subfaults]
        sf_rup_azimuths[row.Index] = round(
            angular_mean_degrees(row_slip_azimuths), 1
        )

    mf_subs = []
    mf_faults_unique = []
    for mf in multi_fault_rups:
        subs = []
        faults = []
        for sf in mf:
            subs.extend(srup_lookup[sf])
            faults.append(fault_lookup[sf])

        mf_subs.append(subs)
        mf_faults_unique.append(faults)

    mf_df = pd.DataFrame(
        index=np.arange(len(mf_subs)) + len(rupture_df),
        data={
            'subfaults': mf_subs,
            'ruptures': multi_fault_rups,
            'faults': mf_faults_unique,
        },
    )

    rupture_df = pd.concat([rupture_df, mf_df], axis=0)

    frac_areas = []
    mean_rakes = []
    slip_azimuths = []
    all_areas = []
    mags = []
    fault_frac_areas = []

    for row in rupture_df.itertuples():
        areas = np.array([area_lookup[sf] for sf in row.subfaults])
        sum_area = areas.sum()
        area_fracs = areas / sum_area
        frac_areas.append(np.round(area_fracs, 4).tolist())

        rakes = np.array([rake_lookup[sf] for sf in row.subfaults])
        azimuths = [sf_rup_azimuths[sf] for sf in row.ruptures]
        slip_azimuths.append(azimuths)
        mean_rake = weighted_angular_mean_degrees(rakes, area_fracs)
        mean_rakes.append(mean_rake)

        mags.append(
            area_to_mag(areas.sum(), mstype=area_mag_msr, rake=mean_rake)
        )
        all_areas.append(sum_area)

        if len(row.faults) == 1:
            f_areas = [1.0]
        else:
            f_areas = []
            f_area_d = AccumDict()
            for sf in row.subfaults:
                f_area_d += {sub_fid_lookup[sf]: area_lookup[sf]}
            f_area_d /= sum(f_area_d.values())
            for fault in row.faults:
                f_areas.append(round(f_area_d[fault], 1))

        fault_frac_areas.append(f_areas)

    rupture_df['frac_area'] = frac_areas
    rupture_df['fault_frac_area'] = fault_frac_areas
    rupture_df['mean_rake'] = np.round(mean_rakes, 1)
    rupture_df['slip_azimuth'] = slip_azimuths
    rupture_df['mag'] = np.round(mags, mag_decimals)
    rupture_df['area'] = np.round(all_areas, 1)
    rupture_df['displacement'] = np.round(
        get_rupture_displacement(
            rupture_df['mag'], rupture_df['area'], shear_modulus=SHEAR_MODULUS
        ),
        3,
    )

    return rupture_df


def get_boundary_3d(smsh):
    """
    Builds a fault trace and a 3D boundary from a Surface mesh.

    Parameters
    ----------
    smsh : openquake.hazardlib.geo.mesh.Mesh
        Surface mesh.

    Returns
    -------
    trace : shapely.geometry.LineString
        Fault trace.
    boundary : shapely.geometry.Polygon
        3D boundary.
    """
    coo = []

    # Upper boundary + trace
    idx = np.where(np.isfinite(smsh.mesh.lons[0, :]))[0]
    tmp = [
        (
            smsh.mesh.lons[0, i],
            smsh.mesh.lats[0, i],
            smsh.mesh.depths[0, i] * -1,
        )
        for i in idx
    ]
    tmp = [c for c in tmp if c != (0.0, 0.0, -0.0)]
    trace = LineString(tmp)
    coo.extend(tmp)

    # Right boundary
    idx = np.where(np.isfinite(smsh.mesh.lons[:, -1]))[0]
    tmp = [
        (
            smsh.mesh.lons[i, -1],
            smsh.mesh.lats[i, -1],
            smsh.mesh.depths[i, -1] * -1,
        )
        for i in idx
    ]
    tmp = [c for c in tmp if c != (0.0, 0.0, -0.0)]
    coo.extend(tmp)

    # Lower boundary
    idx = np.where(np.isfinite(smsh.mesh.lons[-1, :]))[0]
    tmp = [
        (
            smsh.mesh.lons[-1, i],
            smsh.mesh.lats[-1, i],
            smsh.mesh.depths[-1, i] * -1,
        )
        for i in np.flip(idx)
    ]
    tmp = [c for c in tmp if c != (0.0, 0.0, -0.0)]
    coo.extend(tmp)

    # Left boundary
    idx = idx = np.where(np.isfinite(smsh.mesh.lons[:, 0]))[0]
    tmp = [
        (
            smsh.mesh.lons[i, 0],
            smsh.mesh.lats[i, 0],
            smsh.mesh.depths[i, 0] * -1,
        )
        for i in np.flip(idx)
    ]
    tmp = [c for c in tmp if c != (0.0, 0.0, -0.0)]
    coo.extend(tmp)

    return trace, Polygon(coo)


def make_subfault_gdf(subfault_df, keep_surface=False, keep_trace=False):
    polies = [
        get_boundary_3d(row.surface)[1] for row in subfault_df.itertuples()
    ]
    geometry = polies
    subfault_gdf = gpd.GeoDataFrame(subfault_df, geometry=geometry)

    subfault_gdf['fault_position'] = [
        str(row.fault_position) for row in subfault_gdf.itertuples()
    ]

    if not keep_surface:
        del subfault_gdf['surface']
    if not keep_trace:
        del subfault_gdf['trace']
    return subfault_gdf


def make_rupture_gdf(
    fault_network,
    rup_df_key='rupture_df',
    keep_sequences=False,
    same_size_arrays: bool = True,
) -> gpd.GeoDataFrame:
    """
    Makes a GeoDataFrame, with a row for each rupture in the fault network.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        DataFrame containing information about each rupture.
        See `make_rupture_df` for more information on the format.
    subfault_gdf : gpd.GeoDataFrame
        GeoDataFrame containing information about each subfault.
        See `make_subfault_gdf` for more information on the format.
    keep_sequences : bool
        Whether to keep the subfault sequences (i.e., the list or tuple of
        subfault indices that make up any given rupture) in the rupture_df.
        This defaults to False, as GeoPandas won't serialize these to GeoJSON.

    Returns
    -------
    rupture_gdf : gpd.GeoDataFrame
        GeoDataFrame containing information about each rupture.
    """
    single_rup_df = fault_network['single_rup_df']
    subfaults = fault_network['subfaults']
    rupture_df = fault_network[rup_df_key]
    sf_meshes = make_sf_rupture_meshes(
        single_rup_df['patches'],
        single_rup_df['fault'],
        subfaults,
        same_size_arrays=same_size_arrays,
    )
    # converting to surfaces because get_boundary_3d doesn't take meshes
    sf_surfs = [SimpleFaultSurface(sf_mesh) for sf_mesh in sf_meshes]

    rup_meshes = []
    for rup in rupture_df.itertuples():
        rup_polies = [
            get_boundary_3d(sf_surfs[sf_rup])[1] for sf_rup in rup.ruptures
        ]
        rup_meshes.append(MultiPolygon(rup_polies))

    rupture_gdf = gpd.GeoDataFrame(rupture_df, geometry=rup_meshes)
    if not keep_sequences:
        rupture_gdf['subfaults'] = [str(sf) for sf in rupture_gdf.subfaults]
        del rupture_gdf['frac_area']
        del rupture_gdf['fault_frac_area']

    return rupture_gdf


def merge_meshes_no_overlap(
    arrays, positions, same_size_arrays: bool = True
) -> np.ndarray:
    """
    Merges a list of arrays into a single array, with no overlap between
    the arrays.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of arrays to merge.
    positions : list of tuples
        List of tuples containing the position of each array in the final
        array. Each tuple should be in the format (row, column).

    Returns
    -------
    final_array : np.ndarray
        Merged array.
    """
    # Check that all arrays have the same shape
    # Optional, but should be used for simple faults
    # Kite faults can have different shapes
    if same_size_arrays:
        first_shape = arrays[0].shape
        for arr in arrays:
            assert (
                arr.shape == first_shape
            ), "All arrays must have the same shape"

    else:
        # check to see that all arrays share the same rows or same columns
        row_lengths = [arr.shape[0] for arr in arrays]
        col_lengths = [arr.shape[1] for arr in arrays]
        assert (
            len(set(row_lengths)) == 1 or len(set(col_lengths)) == 1
        ), "All arrays must have the same number of rows or columns"
        # `first_shape` is, in this case, the largest array
        first_shape = (max(row_lengths), max(col_lengths))

    # check that all positions are unique and accounted for
    all_rows = sorted(list(set(pos[0] for pos in positions)))
    all_cols = sorted(list(set(pos[1] for pos in positions)))
    for row in all_rows:
        for col in all_cols:
            assert (row, col) in positions, f"Missing position: {(row, col)}"
            assert (
                len([pos for pos in positions if pos == (row, col)]) == 1
            ), f"Duplicate position: {(row, col)}"

    # Adjust the positions so that the minimum starts at 0
    min_row = min(pos[0] for pos in positions)
    min_col = min(pos[1] for pos in positions)
    adjusted_positions = [(r - min_row, c - min_col) for r, c in positions]

    # Determine the size of the final array (assuming no overlaps)
    n_rows = len(all_rows) * first_shape[0]
    n_cols = len(all_cols) * first_shape[1]

    final_array = np.zeros((n_rows, n_cols))

    for arr, pos in zip(arrays, adjusted_positions):
        start_row = pos[0] * first_shape[0]
        end_row = start_row + arr.shape[0]
        start_col = pos[1] * first_shape[1]
        end_col = start_col + arr.shape[1]

        final_array[start_row:end_row, start_col:end_col] += arr

    return final_array


def make_mesh_from_subfaults(
    subfaults: list[dict], same_size_arrays: bool = True
) -> RectangularMesh:
    """
    Makes a RectangularMesh from a list of subfaults.

    Parameters
    ----------
    subfaults : list of dicts
        List of subfaults.

    Returns
    -------
    mesh : RectangularMesh
        Mesh composed of the meshes from all the subfaults.
    """
    if len(subfaults) == 1:
        return subfaults[0]['surface'].mesh

    big_lons = merge_meshes_no_overlap(
        [sf['surface'].mesh.lons for sf in subfaults],
        [sf['fault_position'] for sf in subfaults],
        same_size_arrays=same_size_arrays,
    )

    big_lats = merge_meshes_no_overlap(
        [sf['surface'].mesh.lats for sf in subfaults],
        [sf['fault_position'] for sf in subfaults],
        same_size_arrays=same_size_arrays,
    )
    big_depths = merge_meshes_no_overlap(
        [sf['surface'].mesh.depths for sf in subfaults],
        [sf['fault_position'] for sf in subfaults],
        same_size_arrays=same_size_arrays,
    )

    return RectangularMesh(big_lons, big_lats, big_depths)


def make_sf_rupture_mesh(
    rupture_indices, subfaults, same_size_arrays: bool = True
) -> RectangularMesh:
    """
    Makes a single-fault rupture mesh from a list of subfaults. This is
    a contiguous surface, unlike a multi-fault rupture surface.

    Parameters
    ----------
    rupture_indices : list of int
        List of subfault indices.
    subfaults : list of dicts
        List of subfaults.

    Returns
    -------
    mesh : RectangularMesh
        Mesh composed of the meshes from all the subfaults in the rupture.
    """
    subs = [subfaults[i] for i in rupture_indices]
    mesh = make_mesh_from_subfaults(subs, same_size_arrays=same_size_arrays)
    return mesh


def make_sf_rupture_meshes(
    all_rupture_indices, faults, all_subfaults, same_size_arrays: bool = True
) -> list[RectangularMesh]:
    """
    Makes a list of rupture meshes from a list of single-fault ruptures.

    Parameters
    ----------
    all_rupture_indices : list of lists
        List of lists of subfault indices for each single-fault rupture.
    faults : list of dicts
        List of faults.
    all_subfaults : list of dicts
        List of subfaults.

    Returns
    -------
    rup_meshes : list of RectangularMesh
        List of rupture meshes.
    """
    grouped_subfaults = group_subfaults_by_fault(all_subfaults)

    rup_meshes = []

    for i, rup_indices in enumerate(all_rupture_indices):
        try:
            subs_for_fault = grouped_subfaults[faults[i]]
            mesh = make_sf_rupture_mesh(
                rup_indices, subs_for_fault, same_size_arrays=same_size_arrays
            )
            rup_meshes.append(mesh)
        except IndexError as e:
            logging.error(f"Problems with rupture {i}: " + str(e))
        except AssertionError as e:
            logging.error(f"Problems with rupture {i}: " + str(e))

    return rup_meshes


def shapely_multipoly_to_geojson(multipoly, return_type='coords'):
    out_polies = [
        [[list(pt) for pt in poly.exterior.coords]] for poly in multipoly.geoms
    ]
    if return_type == 'coords':
        return out_polies
    elif return_type == "geometry":
        return {
            "type": "MultiPolygon",
            "coordinates": out_polies,
        }
    elif return_type == 'feature':
        return {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": out_polies,
            },
        }


def export_ruptures_new(
    fault_network, rup_df_key='rupture_df_keep', outfile=None
):
    # subfault_gdf = make_subfault_gdf(fault_network['subfault_df'])
    if rup_df_key != 'rupture_gdf':
        rupture_gdf = make_rupture_gdf(
            fault_network, rup_df_key=rup_df_key, keep_sequences=True
        )
    else:
        rupture_gdf = fault_network['rupture_gdf']

    outfile_type = outfile.split('.')[-1]

    if outfile_type in ['geojson', 'json', 'json_dict']:
        geoms = {
            i: shapely_multipoly_to_geojson(
                rup['geometry'], return_type='feature'
            )
            for i, rup in rupture_gdf.iterrows()
        }

        rup_json = fault_network[rup_df_key].to_dict(orient='index')
        features = []
        for i, rj in rup_json.items():
            f = geoms[i]
            f["properties"] = {k: v for k, v in rj.items() if k != 'geometry'}
            features.append(f)

        out_geojson = {"type": "FeatureCollection", "features": features}

        if outfile_type == 'json_dict':
            return out_geojson
        else:
            with open(outfile, 'w') as f:
                json.dump(out_geojson, f)
