import re
import pyproj
import shapely.ops as ops
from shapely.wkt import loads
from functools import partial
from openquake.man.mfd import get_rates_within_m_range
from openquake.hazardlib.source.area import AreaSource


def _get_area(geom):
    """
    Compute the area of a shapely polygon with lon, lat coordinates.
    See http://tinyurl.com/h35nde4

    :parameter geom:
        A shapely polygon
    :return:
        The area of the polygon in km2
    """
    geom_aea = ops.transform(partial(pyproj.transform,
                                     pyproj.Proj(init='EPSG:4326'),
                                     pyproj.Proj(proj='aea',
                                                 lat1=geom.bounds[1],
                                                 lat2=geom.bounds[3])),
                             geom)
    return geom_aea.area/1e6


def get_areas(model):
    """
    :parameter model:
        A list of openquake source instances
    :returns:
        A (key, value) dictionary, where key is the source ID and value
        corresponds to the area of the source.
    """
    areas = {}
    for src in model:
        if isinstance(src, AreaSource):
            areas[src.source_id] = _get_area(loads(src.polygon.wkt))
    return areas


def get_rates_density(model, mmint=0.0, mmaxt=11.0, trt='.*'):
    """
    :parameter model:
        A list of openquake source instances
    :returns:
        A (key, value) dictionary, where key is the source ID and value
        corresponds to the area of the source.
    """
    dens = {}
    for src in model:
        if (isinstance(src, AreaSource) and
                re.search(trt, src.tectonic_region_type)):
            trates = get_rates_within_m_range(src.mfd, mmint=mmint,
                                              mmaxt=mmaxt)
            mmin, mmax = src.mfd.get_min_max_mag()
            area = _get_area(loads(src.polygon.wkt))
            dens[src.source_id] = trates / area
    return dens
