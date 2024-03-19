# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2024 GEM Foundation
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

import numpy as np
import geopandas as gpd
from openquake.hazardlib.geo.geodetic import geodetic_distance
from openquake.hazardlib.geo.surface.base import _angle_difference


def check_duplicates(df, criteria):
    """
    The dataframe must contain a column named 'event_time' with dtype
    corresponding to a datetime
    """

    # Create a geopandas instance
    get_points = gpd.points_from_xy(df.ev_longitude, df.ev_latitude)
    gdf = gpd.GeoDataFrame(df, geometry=get_points, crs="EPSG:4326")

    # Loop over the events in the flatfile
    # ev_indexes = sorted(df.event_id.unique())
    events = df.groupby('event_id').first().reset_index()

    for iii, (i_row, row) in enumerate(events.iterrows()):

        dt_time = row.event_time - events.event_time
        dt_time = np.abs(dt_time.dt.total_seconds())

        idxs = dt_time < criteria['delta_epi_t']
        if np.sum(idxs) < 2:
            continue

        # TODO For now we assume that there can be only one duplicated event
        assert len(np.nonzero(idxs)[0]) == 2

        dupl_id = list(set(np.nonzero(idxs)[0]) - set([iii]))[0]

        # Check the distance between the epicenters
        dst = geodetic_distance(
            row.ev_longitude, row.ev_latitude,
            events.iloc[dupl_id].ev_longitude, events.iloc[dupl_id].ev_latitude)

        # Check if difference in distance [km] is lower than the threshold
        # defined in the configuration
        if dst > criteria['delta_epi']:
            continue

        print(f"{row.event_id:40s} {events.iloc[dupl_id].event_id:40s}")
