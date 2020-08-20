import pyproj
import shapely.ops as ops
from functools import partial

SHEAR_MODULUS = 32e9  # Pascals


def get_area(geom):
    """
    Compute the area of a shapely polygon with lon, lat coordinates.
    See http://tinyurl.com/h35nde4

    :parameter geom:

    :return:
        The area of the polygon in km2
    """
    geom_aea = ops.transform(partial(pyproj.transform,
                                     pyproj.Proj('EPSG:4326'),
                                     pyproj.Proj(proj='aea',
                                                 lat1=geom.bounds[1],
                                                 lat2=geom.bounds[3])),
                             geom)
    return geom_aea.area/1e6


def slip_from_mo(mo, area):
    """
    :parameter mo:
        Scalar seismic moment [Nm]
    :parameter area:
        Area of the fault [km2]
    """
    return mo / (SHEAR_MODULUS * area*1e6)
