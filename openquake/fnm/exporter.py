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
import os
from typing import Optional


import numpy as np
import pandas as pd
import geopandas as gpd

from pyproj import Transformer
from scipy.stats import poisson
from datetime import datetime, timezone
from shapely.geometry import MultiLineString
from geojson import Polygon, Feature, FeatureCollection, dump

from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.sourceconverter import SourceGroup
from openquake.hazardlib.nrml import SourceModel
from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.source import MultiFaultSource
from openquake.hazardlib.geo.surface import KiteSurface

from openquake.fnm.section import get_subsection


def _get_profiles(kite_surf):
    lons, lats, depths = kite_surf.mesh.array

    n_profiles = lons.shape[1]
    profiles = []

    profiles = [
        list(zip(lons[:, col], lats[:, col], depths[:, col]))
        for col in range(n_profiles)
    ]

    profiles = [Line([Point(*p) for p in profile]) for profile in profiles]

    return profiles


def make_multifault_source(
    fault_network,
    source_id: str = "test_source",
    name: str = "Test Source",
    tectonic_region_type: str = "Active Shallow Crust",
    investigation_time=1.0,
    infer_occur_rates: bool = False,
    surface_type="kite",
    ruptures_for_output='all',
    rupture_occurrence_rates=None,
):
    surfaces = []
    if surface_type == "kite":
        for sub_surface in fault_network['subfault_df']['surface']:
            if isinstance(sub_surface, KiteSurface):
                profiles = _get_profiles(sub_surface)
                sub_surface.profiles = profiles
                surfaces.append(sub_surface)
            else:
                sf_kite_surface = KiteSurface(sub_surface.mesh)
                profiles = _get_profiles(sf_kite_surface)
                sf_kite_surface.profiles = profiles
                surfaces.append(sf_kite_surface)

    elif surface_type == 'simple_fault':
        raise NotImplementedError(
            "Cannot use simple_fault surfaces with multifault sources"
        )

    if ruptures_for_output == 'all':
        rup_df = fault_network['rupture_df']
    elif ruptures_for_output == 'filtered':
        rup_df = fault_network['rupture_df_keep']
    else:
        raise ValueError(
            "`ruptures_for_output` must be `all` or `filtered`, not %s",
            ruptures_for_output,
        )

    rupture_idxs = rup_df['subfaults'].values.tolist()
    mags = rup_df['mag'].values
    rakes = rup_df['mean_rake'].values

    if rupture_occurrence_rates is None:
        occurrence_rates = rup_df['annual_occurrence_rate'].values

    pmfs = [
        poisson.pmf([0, 1, 2, 3, 4], r).tolist()
        for r in rupture_occurrence_rates
    ]

    mfs = MultiFaultSource(
        source_id=source_id,
        name=name,
        tectonic_region_type=tectonic_region_type,
        rupture_idxs=rupture_idxs,
        occurrence_probs=pmfs,
        magnitudes=mags,
        rakes=rakes,
        investigation_time=investigation_time,
        infer_occur_rates=infer_occur_rates,
    )

    mfs.sections = surfaces

    return mfs


def write_multifault_source(
    out_path,
    mf_source,
    source_name=None,
    investigation_time=1.0,
):

    if source_name is None:
        source_name = mf_source.source_id

    xml_outpath = os.path.join(out_path, f"{source_name}.xml")
    write_source_model(
        xml_outpath, [mf_source], investigation_time=investigation_time
    )


def make_multifault_source_old(
    fsys,
    ruptures: pd.DataFrame,
    source_id: str = "test_source",
    name: str = "Test Source",
    tectonic_region_type: str = "Active Shallow Crust",
    investigation_time=0.0,
    infer_occur_rates: bool = False,
    surface_type="kite",
):
    surfaces = []
    if surface_type == "kite":
        for fault, subsecs in fsys:
            fault_mesh = fault.mesh
            for subsec in subsecs[0]:
                subsec_mesh = get_subsection(fault_mesh, subsec)
                subsec_surface = KiteSurface(subsec_mesh)
                profiles = _get_profiles(subsec_surface)
                subsec_surface.profiles = profiles
                surfaces.append(subsec_surface)

    elif surface_type == 'simple_fault':
        pass

    rupture_idxs = ruptures.subsections.values.tolist()
    mags = ruptures.M.values
    rakes = ruptures.rake.values

    pmfs = [
        poisson.pmf([0, 1, 2, 3, 4], r).tolist()
        for r in ruptures.occurrence_rate.values
    ]

    mfs = MultiFaultSource(
        source_id=source_id,
        name=name,
        tectonic_region_type=tectonic_region_type,
        rupture_idxs=rupture_idxs,
        occurrence_probs=pmfs,
        magnitudes=mags,
        rakes=rakes,
        investigation_time=investigation_time,
        infer_occur_rates=infer_occur_rates,
    )

    mfs.sections = surfaces

    return mfs


def export_subsections(fsys, fname: str = None, format: str = "geojson"):
    """
    Exports subsections.

    :param mesh:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param fname:
        A string with name and path of the output file
    :param format:
        A string indicating the format.
    """
    if fname is None:
        fname = "subsections.geojson"

    if format == "geojson":
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        conv_fact = -1000

        sid = 0
        polygons = []
        for i_section, (surf, subss) in enumerate(fsys):
            mesh = surf.mesh.array

            for subs in subss[0]:
                lons = []
                lats = []
                deps = []

                # Get the submesh representing the surface of the
                # single-section rupture
                subs = subs.astype(int)
                submesh = mesh[
                    :, subs[0] : subs[0] + subs[2], subs[1] : subs[1] + subs[3]
                ]

                # Top
                tlo = submesh[0, 0, :]
                tla = submesh[1, 0, :]
                tde = submesh[2, 0, :]
                tidx = np.isfinite(tlo)
                tx, ty = transformer.transform(tla[tidx], tlo[tidx])
                lons.extend(list(tx))
                lats.extend(list(ty))
                deps.extend(list(tde * conv_fact))

                # Right
                tlo = submesh[0, 1:, -1]
                tla = submesh[1, 1:, -1]
                tde = submesh[2, 1:, -1]
                tidx = np.isfinite(tlo)
                tx, ty = transformer.transform(tla[tidx], tlo[tidx])
                lons.extend(list(tx))
                lats.extend(list(ty))
                deps.extend(list(tde[tidx] * conv_fact))

                # Bottom
                tlo = submesh[0, -1, :-2]
                tla = submesh[1, -1, :-2]
                tde = submesh[2, -1, :-2]
                tidx = np.isfinite(tlo)
                tx, ty = transformer.transform(tla[tidx], tlo[tidx])
                lons.extend(list(np.flipud(tx)))
                lats.extend(list(np.flipud(ty)))
                deps.extend(list(np.flipud(tde[tidx]) * conv_fact))

                # Left
                tlo = submesh[0, 1:-1, 0]
                tla = submesh[1, 1:-1, 0]
                tde = submesh[2, 1:-1, 0]
                tidx = np.isfinite(tlo)
                tx, ty = transformer.transform(tla[tidx], tlo[tidx])
                lons.extend(list(np.flipud(tx)))
                lats.extend(list(np.flipud(ty)))
                deps.extend(list(np.flipud(tde) * conv_fact))

                # Creating the feature
                tmp = [(lo, la, de) for lo, la, de in zip(lons, lats, deps)]
                feature = Feature(geometry=Polygon([tmp]))
                feature["id"] = sid
                feature["properties"] = {"section": i_section}
                polygons.append(feature)

                sid += 1

        # Create feature collection and set the CRS
        feature_collection = FeatureCollection(polygons)
        tmp = {"name": "urn:ogc:def:crs:EPSG::3857"}
        feature_collection["crs"] = {"type": "name", "properties": tmp}

        # Write the output file
        with open(fname, "w") as f:
            dump(feature_collection, f)


def export_ruptures(
    rups,
    single_rups,
    fsys,
    mags,
    rates: Optional[np.ndarray] = None,
    fname: str = None,
):
    """
    Exports subsections.

    :param mesh:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param single_rups:
        A
    :param fsys:
        A
    :param mags:
        A
    :param fname:
        A string with name and path of the output file
    """

    single_rups = single_rups.astype(int)

    if fname is None:
        fname = "ruptures.gpkg"

    if fname.split(".")[-1] != "gpkg":
        driver = "GPKG"
    elif fname.split(".")[-1] != "geojson":
        driver = "GeoJSON"

    # Geographic coordinates converter
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    # Loop through the ruptures
    data = {}
    data["geometry"] = []
    data["rupture_id"] = []
    data["datetime"] = []
    data["num_sections"] = []
    data["magnitude"] = []
    if rates is not None:
        data["occurrence_rate"] = []
    for i_rup, (rup, mag) in enumerate(zip(rups, mags)):
        # Loop through the single section ruptures composing the rupture
        lines = []
        for i_srup in rup:
            # Single section rupture
            ssrup = single_rups[i_srup]

            # Retrieve the mesh representing the section
            submesh = fsys[int(ssrup[6])][0].mesh.array

            # Get top of rupture and convert coordinates
            ilo = ssrup[1]
            iup = ssrup[1] + ssrup[2] + 1
            tlo, tla = transformer.transform(
                submesh[1, 0, ilo:iup], submesh[0, 0, ilo:iup]
            )

            lines.append([(lo, la) for lo, la in zip(tlo, tla)])

        # Create the multiline
        mline = MultiLineString(lines)

        # Update the geodataframe
        data["geometry"].append(mline)
        data["rupture_id"].append(i_rup)
        tmp_time = datetime.fromtimestamp(i_rup, timezone.utc)
        data["datetime"].append(tmp_time)
        data["num_sections"].append(len(rup))
        data["magnitude"].append(float(f"{mag:.2f}"))
        if rates is not None:
            data["occurrence_rate"].append(rates[i_rup])

    gdf = gpd.GeoDataFrame(data)
    gdf = gdf.set_crs(3857, allow_override=True)
    gdf.to_file(fname, layer="ruptures", driver=driver)
