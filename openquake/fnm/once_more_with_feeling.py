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
import time
import logging
from copy import deepcopy
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon

from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.mesh import RectangularMesh, Mesh
from openquake.hazardlib.geo.surface import SimpleFaultSurface

from openquake.fnm.importer import (
    simple_fault_surface_from_feature,
)

from openquake.fnm.msr import area_to_mag

from openquake.fnm.inversion.utils import (
    weighted_mean,
    get_rupture_displacement,
)


logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_simple_fault_from_feature(
    feature: dict,
    edge_sd: float = 2.0,
    lsd_default: float = 20.0,
    usd_default: float = 0.0,
) -> dict:
    props_to_keep = [
        "fid",
        "net_slip_rate",
        "net_slip_rate_err",
        "rake",
        "rake_err",
    ]
    fault = {prop: feature['properties'][prop] for prop in props_to_keep}

    fault['surface'] = simple_fault_surface_from_feature(
        feature,
        edge_sd=edge_sd,
        lsd_default=lsd_default,
        usd_default=usd_default,
    )

    fault['trace'] = feature['geometry']['coordinates']

    return fault


def subdivide_simple_fault_surface(
    fault_surface: SimpleFaultSurface,
    subsection_size=[
        15.0,
        15.0,
    ],
    edge_sd=2.0,
    dip_sd=2.0,
    dip=None,
):
    fault_mesh = fault_surface.mesh
    fault_trace = Line(
        [
            Point(lon, fault_mesh.lats[0, i], fault_mesh.depths[0, i])
            for i, lon in enumerate(fault_mesh.lons[0])
        ]
    )

    # get basic geometric info
    if dip is None:
        dip = fault_surface.get_dip()
    strike = fault_trace[0].azimuth(fault_trace[-1])

    fault_length = fault_trace.get_length()
    fault_width = fault_surface.get_width()

    if subsection_size[0] > 0:
        subsec_length_init = subsection_size[0]
    elif subsection_size[0] < 0:
        subsec_length_init = fault_length / abs(subsection_size[0])

    if subsection_size[1] > 0:
        subsec_width_init = subsection_size[1]
    elif subsection_size[1] < 0:
        subsec_width_init = fault_width / abs(subsection_size[1])

    # calculate number of segments and point spacing along strike
    num_segs_along_strike = max(
        int(round(fault_length / subsec_length_init)), 1
    )
    subsec_length = fault_length / num_segs_along_strike
    pt_spacing = subsec_length / round(subsec_length / edge_sd)
    n_pts_strike = (fault_length / pt_spacing) + 1
    assert (n_pts_strike % 1 <= 1e-5) or (n_pts_strike % 1 >= (1.0 - 1e-5)), (
        "Resampled trace not integer length: " + f"{n_pts_strike}"
    )
    n_pts_strike = int(round(n_pts_strike))

    # resample fault trace and calculate number of points along strike
    new_trace = fault_trace.resample_to_num_points(n_pts_strike)
    assert new_trace.coo.shape[0] == n_pts_strike, (
        "Resampled trace not correct length: "
        + f"{new_trace.coo.shape[0]} != {n_pts_strike}"
    )

    n_subsec_pts_strike = ((n_pts_strike - 1) / num_segs_along_strike) + 1

    if n_subsec_pts_strike != round(n_subsec_pts_strike):
        pass
        # from IPython.core.debugger import set_trace

        # set_trace()

    assert n_subsec_pts_strike % 1 == 0.0, (
        "Resampled trace not dividing equally among subsegments: "
        + f"{n_subsec_pts_strike}"
    )

    n_subsec_pts_strike = int(n_subsec_pts_strike)

    # calculate number of segments and point spacing down dip
    num_segs_down_dip = max(int(round(fault_width / subsec_width_init)), 1)
    subsec_width = fault_width / num_segs_down_dip
    dip_pt_spacing = subsec_width / round(subsec_width / dip_sd)

    azimuth = (strike + 90) % 360

    mesh = []

    hor_dip_spacing = dip_pt_spacing * np.cos(np.radians(dip))
    vert_dip_spacing = dip_pt_spacing * np.sin(np.radians(dip))
    n_level_sets = int(round(fault_width / dip_pt_spacing)) + 1

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
    assert (
        n_subsec_pts_dip % 1 == 0.0
    ), f"Resampled mesh not dividing equally among subsegments down-dip"
    n_subsec_pts_dip = int(n_subsec_pts_dip)

    subsec_meshes = subdivide_rupture_mesh(
        resampled_mesh.lons,
        resampled_mesh.lats,
        resampled_mesh.depths,
        num_segs_down_dip,
        num_segs_along_strike,
        n_subsec_pts_strike,
        n_subsec_pts_dip,
    )

    return subsec_meshes


def subdivide_rupture_mesh(
    lons,
    lats,
    depths,
    num_segs_down_dip,
    num_segs_along_strike,
    n_subsec_pts_strike,
    n_subsec_pts_dip,
):
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

            subsec_mesh = RectangularMesh(
                subsec_lons, subsec_lats, subsec_depths
            )
            subsec_meshes.append({'row': i, 'col': j, 'mesh': subsec_mesh})

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
    props_to_keep = [
        "fid",
        "net_slip_rate",
        "net_slip_rate_err",
        "rake",
        "rake_err",
    ]

    # With the current code organization, this causes a circular import
    # if surface is None:
    #    if surface_type == "simple_fault_surface":
    #        surface = simple_fault_surface_from_feature(
    #            fault,
    #            edge_sd=edge_sd,
    #            # dip_sd=dip_sd,
    #            lsd_default=lsd_default,
    #            usd_default=usd_default,
    #        )
    #    elif surface_type == "kite_surface":
    #        pass

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
        subfault['fault_position'] = (sub_mesh['row'], sub_mesh['col'])
        subfault["trace"] = [
            [lon, mesh.lats[0, i], mesh.depths[0, i]]
            for i, lon in enumerate(mesh.lons[0])
        ]
        if surface_type == "simple_fault_surface":
            subfault["surface"] = SimpleFaultSurface(mesh)
        elif surface_type == "kite_surface":
            pass

        subfault['length'] = Line(
            [Point(*p) for p in subfault['trace']]
        ).get_length()
        subfault['width'] = subfault['surface'].get_width()
        subfault["area"] = subfault["surface"].get_area()
        subfault["subsec_id"] = i

        subsections.append(subfault)

    return subsections


def make_subfault_df(all_subfaults):
    subfault_df = pd.concat(pd.DataFrame(sf) for sf in all_subfaults)
    subfault_df = subfault_df.reset_index(drop=True)
    subfault_df.index.name = "subfault_id"

    strikes = []
    dips = []

    for row in subfault_df.itertuples():
        strikes.append(row.surface.get_strike())
        dips.append(row.surface.get_dip())

    subfault_df['strike'] = strikes
    subfault_df['dip'] = dips

    return subfault_df


def group_subfaults_by_fault(subfaults: list[dict]) -> dict:
    subfault_dict = {}
    for fault_group in subfaults:
        if fault_group[0]['fid'] not in subfault_dict:
            subfault_dict[fault_group[0]['fid']] = []
        subfault_dict[fault_group[0]['fid']].append(fault_group)

    return subfault_dict


def make_rupture_df(
    single_fault_rup_df,
    multi_fault_rups,
    subfault_df,
    area_mag_msr='Leonard2014_Interplate',
):
    rups_involved = [[int(r)] for r in single_fault_rup_df.index.values]
    rupture_df = single_fault_rup_df[['subfaults']]

    rupture_df = pd.DataFrame(
        index=rupture_df.index,
        data={
            'subfaults': single_fault_rup_df.subfaults,
            'ruptures': rups_involved,
        },
    )

    srup_lookup = {i: row.subfaults for i, row in rupture_df.iterrows()}
    area_lookup = {i: row.area for i, row in subfault_df.iterrows()}
    rake_lookup = {i: row.rake for i, row in subfault_df.iterrows()}

    mf_subs = []
    for mf in multi_fault_rups:
        subs = []
        for sf in mf:
            subs.extend(srup_lookup[sf])

        mf_subs.append(subs)

    mf_df = pd.DataFrame(
        index=np.arange(len(mf_subs)) + len(rupture_df),
        data={'subfaults': mf_subs, 'ruptures': multi_fault_rups},
    )

    rupture_df = pd.concat([rupture_df, mf_df], axis=0)

    frac_areas = []
    mean_rakes = []
    all_areas = []
    mags = []

    # mean_slip_azimuths = []
    for i, row in rupture_df.iterrows():
        areas = np.array([area_lookup[sf] for sf in row.subfaults])
        sum_area = areas.sum()
        area_fracs = areas / sum_area
        frac_areas.append(np.round(area_fracs, 2))

        rakes = np.array([rake_lookup[sf] for sf in row.subfaults])
        mean_rake = weighted_mean(rakes, area_fracs)
        mean_rakes.append(mean_rake)

        mags.append(
            area_to_mag(areas.sum(), type=area_mag_msr, rake=mean_rake)
        )
        all_areas.append(sum_area)

    rupture_df['frac_area'] = frac_areas
    rupture_df['mean_rake'] = np.round(mean_rakes, 1)
    rupture_df['mag'] = np.round(mags, 2)
    rupture_df['area'] = np.round(all_areas, 1)

    return rupture_df


def get_boundary_3d(smsh):
    """Returns a polygon"""
    poly = []
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


def make_rupture_gdf(rupture_df, subfault_gdf, keep_sequences=False):
    geoms = []

    for row in rupture_df.itertuples():
        polys = [subfault_gdf.loc[sf, 'geometry'] for sf in row.subfaults]
        geoms.append(MultiPolygon(polys))

    rupture_gdf = gpd.GeoDataFrame(rupture_df, geometry=geoms)
    if not keep_sequences:
        rupture_gdf['subfaults'] = [str(sf) for sf in rupture_gdf.subfaults]
        del rupture_gdf['frac_area']

    return rupture_gdf
