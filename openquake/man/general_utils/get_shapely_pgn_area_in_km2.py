# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import pyproj
import shapely.ops as ops
from functools import partial


def get_area(geom):
    """
    Compute the area of a shapely polygon with lon, lat coordinates.
    See http://tinyurl.com/h35nde4

    :parameter geom:

    :return:
        The area of the polygon in km2
    """
    geom_aea = ops.transform(partial(pyproj.transform,
                                     pyproj.Proj(init='EPSG:4326'),
                                     pyproj.Proj(proj='aea',
                                                 lat_1=geom.bounds[1],
                                                 lat_2=geom.bounds[3])),
                             geom)
    return geom_aea.area/1e6
