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
import shutil
import pathlib

import logging
import numpy as np
import matplotlib.pyplot as plt
from openquake.hazardlib.tom import PoissonTOM
from openquake.wkf.utils import _get_src_id, create_folder, get_list
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.mbt.tools.mfd import EEvenlyDiscretizedMFD
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.source import SimpleFaultSource, MultiPointSource
from openquake.hazardlib.geo.surface import SimpleFaultSurface
from openquake.hazardlib.mfd.multi_mfd import MultiMFD


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
    if isinstance(src, SimpleFaultSource):
        sfc = SimpleFaultSurface.from_fault_data(src.fault_trace,
                                                 src.upper_seismogenic_depth,
                                                 src.lower_seismogenic_depth,
                                                 src.dip, 1.0)
    else:
        raise ValueError('Not supported fault type')

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
                                out_path: str, dst: float,
                                threshold_mag: float = 6.5,
                                use: str='', remove=False):
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
    :param dst:
        The threshold distance used to separate seismicity on the fault and
        in the distributed seismicity sources
    :returns:
        A .xml file with the ajusted point sources
    """

    if len(use) > 0:
        use = get_list(use)

    # Load fault sources
    binw = 0.1
    sourceconv = SourceConverter(investigation_time=1.0,
                                 rupture_mesh_spacing=5.0,
                                 complex_fault_mesh_spacing=5.0,
                                 width_of_mfd_bin=binw)
    ssm_faults = to_python(fname, sourceconv)

    # Loading all the point sources in the distributed seismicity model
    coo_pnt_src = []
    pnt_srcs = []
    for fname in glob.glob(path_point_sources):

        # Info
        logging.info(f'Processing: {fname}')

        # Check if the source must be processed
        src_id = _get_src_id(fname)
        if len(use) > 0 and src_id not in use:
            logging.info(f'Skipping {fname}')
            continue

        # Reading file content
        tssm = to_python(fname, sourceconv)

        # Removing this file
        tmp_fle = pathlib.Path(fname)
        if remove:
            tmp_fle.unlink()

        # Processing
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

    fig, axs = plt.subplots(1, 1)
    plt.plot(coo_pnt_src[:, 0], coo_pnt_src[:, 1], '.')

    # Processing faults
    buffer = []
    bco = []
    for src in faults:

        # Getting the subset of point sources in the surrounding of the fault
        # `src`. `coo_pnt_src` is a numpy.array with two columns (i.e. lon and
        # lat). `pnt_srcs` is a list containing the point sources that
        # collectively describe the distributed seismicity souces provided as
        # input
        pnt_ii, sel_pnt_srcs, sel_pnt_coo, rrup = get_data(src, coo_pnt_src,
                                                           pnt_srcs)


        if pnt_ii is not None:

            # Find the index of points within the buffer zone
            within_idx = np.nonzero(rrup < dst)[0]
            idxs = sorted([pnt_ii[i] for i in within_idx], reverse=True)

            plt.plot(coo_pnt_src[idxs, 0], coo_pnt_src[idxs, 1], 'or', mfc='none')

            for isrc in idxs:

                # Updating mmax for the point source
                # sel_pnt_srcs[isrc].mfd.max_mag = threshold_mag
                pnt_srcs[isrc].mfd.max_mag = threshold_mag

                # Adding point source to the buffer
                buffer.append(pnt_srcs[isrc])
                bco.append([coo_pnt_src[isrc, 0], coo_pnt_src[isrc, 1]])

                # Removing the point source from the list of sources outside
                # of buffers
                pnt_srcs.remove(pnt_srcs[isrc])

            mask = np.ones(len(coo_pnt_src), dtype=bool)
            mask[pnt_ii[within_idx]] = False
            coo_pnt_src = coo_pnt_src[mask, :]

        else:
            continue

        # Fault occurrences
        ocf = np.array(src.mfd.get_annual_occurrence_rates())

    bco = np.array(bco)
    plt.plot(bco[:,0], bco[:,1], 'x')

    tmpsrc = from_list_ps_to_multipoint(pnt_srcs, 'pnts')
    fname_out = os.path.join(out_path, "src_points.xml")
    write_source_model(fname_out, [tmpsrc], 'Distributed seismicity')
    print('Created: {:s}'.format(fname_out))

    tmpsrc = from_list_ps_to_multipoint(buffer, 'buf')
    fname_out = os.path.join(out_path, "src_buffers.xml")
    write_source_model(fname_out, [tmpsrc], 'Distributed seismicity')
    print('Created: {:s}'.format(fname_out))



def from_list_ps_to_multipoint(srcs, src_id):

    # Looping over the points
    lons = []
    lats = []
    avals = []
    settings = False

    for src in srcs:

        minmaxmag = src.get_min_max_mag()
        mmx = minmaxmag[1]
        mmin = minmaxmag[0]

        avals.append(src.mfd.a_val)

        lons.append(src.location.longitude)
        lats.append(src.location.latitude)

        if not settings:

            trt = src.tectonic_region_type
            msr = src.magnitude_scaling_relationship
            rar = src.rupture_aspect_ratio
            usd = src.upper_seismogenic_depth
            lsd = src.lower_seismogenic_depth
            npd = src.nodal_plane_distribution
            hyd = src.hypocenter_distribution

    name = src_id
    mmfd = MultiMFD('truncGutenbergRichterMFD',
                    size=len(avals),
                    min_mag=[mmin],
                    max_mag=[mmx],
                    bin_width=[src.mfd.bin_width],
                    b_val=[src.mfd.b_val],
                    a_val=avals)

    tom = PoissonTOM(1)

    mesh = Mesh(np.array(lons), np.array(lats))
    srcmp = MultiPointSource(src_id, name, trt, mmfd, msr, rar, usd, lsd,
                             npd, hyd, mesh, tom)

    return srcmp
