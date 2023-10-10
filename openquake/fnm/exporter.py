# coding: utf-8

import numpy as np
import geopandas as gpd

from geojson import Polygon, Feature, FeatureCollection, dump
from pyproj import Transformer
from shapely.geometry import MultiLineString
from datetime import datetime, timezone


def export_subsections(fsys, fname: str = None, format: str = 'geojson'):
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
                    :, subs[0]: subs[0] + subs[2], subs[1]: subs[1] + subs[3]
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


def export_ruptures(rups, single_rups, fsys, mags, fname: str = None):
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

    # Geographic coordinates converter
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    # Loop through the ruptures
    data = {}
    data["geometry"] = []
    data["rupture_id"] = []
    data["datetime"] = []
    data["num_sections"] = []
    data["magnitude"] = []
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

    gdf = gpd.GeoDataFrame(data)
    gdf = gdf.set_crs(3857, allow_override=True)
    gdf.to_file(fname, layer="ruptures", driver="GPKG")
