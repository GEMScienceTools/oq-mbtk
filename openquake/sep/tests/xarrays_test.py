import os
import unittest

import numpy as np
import xarray as xr
from osgeo import gdal

from openquake.sep.utils import make_local_relief_raster
from openquake.sep.calculators import (
    calc_newmark_soil_slide_single_event,
    calc_newmark_soil_slide_event_set,
)


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
test_dem = os.path.join(BASE_DATA_PATH, "dem_small.tif")
test_relief = os.path.join(BASE_DATA_PATH, "relief_out.tif")
test_slope = os.path.join(BASE_DATA_PATH, "slope_small.tif")
test_saturation = os.path.join(BASE_DATA_PATH, "saturation.tif")
test_friction = os.path.join(BASE_DATA_PATH, "friction.tif")
test_cohesion = os.path.join(BASE_DATA_PATH, "cohesion.tif")
test_pga = os.path.join(BASE_DATA_PATH, "pga.nc")

relief_map = xr.open_rasterio(test_relief, parse_coordinates=True)[0]
slope_map = xr.open_rasterio(test_slope, parse_coordinates=True)[0]
pga = xr.open_dataset(test_pga)
saturation = xr.open_rasterio(test_saturation, parse_coordinates=True)[0]
friction = xr.open_rasterio(test_friction, parse_coordinates=True)[0]
cohesion = xr.open_rasterio(test_cohesion, parse_coordinates=True)[0]

slope_map.coords = relief_map.coords

slope_map + relief_map