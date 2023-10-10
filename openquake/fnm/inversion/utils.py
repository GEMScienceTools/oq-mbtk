import numpy as np
import pandas as pd
import pyproj as pj
import geopandas as gpd

from math import prod

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform

from openquake.hazardlib.mfd.tapered_gr_mfd import mag_to_mo

from openquake.fnm.constants import SHEAR_MODULUS


def geom_from_fault_trace(fault_trace):
    return LineString([Point(*c) for c in fault_trace])


def project_faults_and_polies(faults, polies: gpd.GeoDataFrame):
    lines = [geom_from_fault_trace(fault["trace"]) for fault in faults]

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
