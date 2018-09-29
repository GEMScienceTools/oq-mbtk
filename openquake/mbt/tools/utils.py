import re
import sys
import time

from openquake.hazardlib.geo.point import Point


def get_time(time_start, time_cell):
    time_end = time.time()
    delta_tot = (time_end-time_start) / 60.
    delta_cell = (time_end-time_cell) / 60.
    tstr = "Elapsed time: total %3.0f min - last cell %3.0f min"
    print(tstr % (delta_tot, delta_cell))
    return time_end


def find_oqmbtk_folder():
    for tstr in sys.path:
        if re.search('oq-mbtk', tstr):
            return tstr
    return None


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


class GetSourceIDs(object):

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.keys = set([key for key in self.model.sources])

    def keep_equal_to(self, param_name, values):
        """
        :parameter str param_name:
        :parameter list values:
        """
        assert type(values) is list
        tmp = []
        for key in self.keys:
            src = self.model.sources[key]
            param_value = getattr(src, param_name)
            for value in values:
                if param_value == value:
                    tmp.append(key)
                    continue
        self.keys = tmp

    def keep_gt_than(self, param_name, value):
        tmp = []
        for key in self.keys:
            src = self.model.sources[key]
            param_value = getattr(src, param_name)
            if param_value > value:
                tmp.append(key)
        self.keys = tmp

    def keep_lw_than(self, param_name, value):
        tmp = []
        for key in self.keys:
            src = self.model.sources[key]
            param_value = getattr(src, param_name)
            if param_value < value:
                tmp.append(key)
        self.keys = tmp
