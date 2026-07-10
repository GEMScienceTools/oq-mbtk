# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation and Électricité de France
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
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
#
# This script is produced within the scope of Work Package 5, named Simulation 
# platform, under SIGMA3 project. For more detailed information about 
# the project, please visit to https://sigma-programs.com/.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
module :mod:`openquake.man.ses_utils.ses_source` provides a function to construct 
OQ Engine area source instance.
"""

import numpy as np

from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.source import AreaSource
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.scalerel import get_available_area_scalerel
from openquake.man.ses_utils.ses_geo import get_oq_polygons


def get_area_source(
            mfd,
            polygon_fname,
            hdd=None,
            grid_spacing=10.
        ):
    """
    Returns an OQ Engine area source instance

    :param mfd:
        An instance of OQ Engine magnitude-frequency distribution class
    :param polygon_fname:
        An instance of a OQ Engine polygon class. See :class:`openquake.hazardlib.geo.Polygon`
    :param hdd:
        An instance of a PMF class. See  :class:`openquake.hazardlib.pmf.PMF`
    :param grid_spacing:
        Grid spacing used for area source discretization.
    """

    # Magnitude scaling-relationship
    msrs = get_available_area_scalerel()
    msr = msrs['WC1994']()

    # Temporal occurrence model
    time_span = 1.0
    tom = PoissonTOM(time_span)

    # Nodal plane distribution
    npd = PMF([(1.0, NodalPlane(0.0, 90.0, 0.0))])

    # Upper and lower seismogenic depth
    usd = 0
    lsd = 15.0

    if hdd is None:
        hdd = PMF([(1.0, 7.5)])
    else:
        usd = np.min([v[1] for v in hdd.data])
        lsd = np.max([v[1] for v in hdd.data])

    # Get geometry
    polys = get_oq_polygons(polygon_fname)
    if len(polys) == 0:
        raise ValueError('The file does not contain a polygon')
    elif len(polys) > 1:
        raise ValueError(f'The file contains {len(polys)} polygons')

    # Area source
    src = AreaSource(
        source_id = 'test',
        name = 'test',
        tectonic_region_type = 'Undef',
        mfd = mfd,
        rupture_mesh_spacing = 5.0,
        magnitude_scaling_relationship = msr,
        rupture_aspect_ratio = 1.0,
        temporal_occurrence_model = tom,
        upper_seismogenic_depth = usd,
        lower_seismogenic_depth = lsd,
        nodal_plane_distribution = npd,
        hypocenter_distribution = hdd,
        polygon = polys[0],
        area_discretization = grid_spacing
    )

    return src
