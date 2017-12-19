

from openquake.hazardlib.geo.point import Point


def merge_dictionaries(dic_a, dic_b):
    """
    :parameter dic_a:
    :parameter dic_b:
    """
    dall = dict(dic_a)
    dall.update(dic_b)
    return dall


def _get_point_list(lons, lats):
    """
    :returns:
        Returns a list of :class:` openquake.hazardlib.geo.point.Point`
        instances
    """
    points = []
    for i in range(0, len(lons)):
        points.append(Point(lons[i], lats[i]))
    return points

