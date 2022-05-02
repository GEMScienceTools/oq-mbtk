# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
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
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.mbt.tools.mfd import EEvenlyDiscretizedMFD
from openquake.hazardlib.sourcewriter import write_source_model


def get_bounding_box(src):
    """
    Get the bounding box of a simple fault source

    :param src:
        See :method:`get_data`
    """

    # This provides the convex hull of the surface projection
    coo = np.array(src.polygon.coords)
    return [min(coo[:, 0]), min(coo[:, 1]), max(coo[:, 0]), max(coo[:, 1])]


def get_data(src, coo_pnt_src, pnt_srcs, buffer=1.0):
    """
    Computes point sources within the bounding box and the corresponding
    rrup distances

    :param src:
        An instance of
    :param coo_pnt_src:

    :param pnt_srcs:
    """

    # Get the bounding box
    bbox = get_bounding_box(src)

    # Find the point sources within the extended buffer arround the fault
    # bounding box
    idxs = np.nonzero((coo_pnt_src[:, 0] > bbox[0]-buffer) &
                      (coo_pnt_src[:, 1] > bbox[1]-buffer) &
                      (coo_pnt_src[:, 0] < bbox[2]+buffer) &
                      (coo_pnt_src[:, 1] < bbox[3]+buffer))[0]
    sel_pnt_srcs = [pnt_srcs[i] for i in idxs]

    # No points selected
    if len(sel_pnt_srcs) < 1:
        return None, None, None, None

    # Coordinates of the selected points i.e. points within the bounding box
    # plus of the fault plus a buffers
    sel_pnt_coo = np.array([(p.location.longitude, p.location.latitude) for p
                            in sel_pnt_srcs])

    # Create the mesh
    mesh = Mesh(sel_pnt_coo[:, 0], sel_pnt_coo[:, 1])

    # Get the fault surface and compute rrup
    sfc = src.get_surface()
    rrup = sfc.get_min_distance(mesh)

    return idxs, sel_pnt_srcs, sel_pnt_coo, rrup


def get_stacked_mfd(srcs: list, within_idx: list, binw: float):
    """
    :param srcs:
    :param within_idx:
        Param
    :param binw:
    """
    for i, idx in enumerate(within_idx):
        if i == 0:
            tot_mfd = EEvenlyDiscretizedMFD.from_mfd(srcs[idx].mfd, binw)
        else:
            tot_mfd.stack(srcs[idx].mfd)
    return tot_mfd


def remove_buffer_around_faults(fname: str, path_point_sources: str,
                                 out_path: str, dst: float):
    """
    Remove the seismicity above a magnitude threshold for all the point
    sources within a buffer around faults.

    :param fname:
        The name of the file with the fault sources in .xml format
    :param path_point_sources:
        The pattern to select the .xml files of the point sources e.g.
        `./../m01_asc/oq/zones/src_*.xml`
    :param out_path:
        The path where to write the output .xml file
    :param dst:
        The distance in km of the buffer
    :returns:
        A .xml file with the ajusted point sources
    """

    # Load fault sources in the SAM models
    binw = 0.1
    sourceconv = SourceConverter(investigation_time=1.0,
                                 rupture_mesh_spacing=5.0,
                                 complex_fault_mesh_spacing=5.0,
                                 width_of_mfd_bin=binw)
    ssm_faults = to_python(fname, sourceconv)

    # Loading all the point sources in the NEW SAM model
    coo_pnt_src = []
    pnt_srcs = []
    for fname in glob.glob(path_point_sources):
        tssm = to_python(fname, sourceconv)
        tcoo = np.array([(p.location.longitude, p.location.latitude) for p in
                         tssm[0]])
        pnt_srcs.extend(tssm[0])
        coo_pnt_src.extend(tcoo)
    coo_pnt_src = np.array(coo_pnt_src)

    # Getting the list of faults
    faults = []
    for grp in ssm_faults:
        for s in grp:
            faults.append(s)

    # Processing faults
    for src in faults:

        # Getting the subset of point sources in the surrounding of the fault
        pnt_ii, sel_pnt_srcs, sel_pnt_coo, rrup = get_data(src, coo_pnt_src,
                                                           pnt_srcs)

        if pnt_ii is not None:
            within_idx = np.nonzero(rrup < dst)[0]
            for isrc in within_idx:
                pnt_srcs[pnt_ii[isrc]].mfd.max_mag = 6.5
                sel_pnt_srcs[isrc].mfd.max_mag = 6.5
        else:
            continue

        # Fault occurrences
        ocf = np.array(src.mfd.get_annual_occurrence_rates())

    fname_out = os.path.join(out_path, "points_ssm.xml")
    write_source_model(fname_out, pnt_srcs, 'Distributed seismicity')
    print('Created: {:s}'.format(fname_out))
