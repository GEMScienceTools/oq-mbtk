# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2024 GEM Foundation
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
import copy
import pathlib

import logging
import numpy as np
import matplotlib.pyplot as plt
from openquake.hazardlib.tom import PoissonTOM
from openquake.wkf.utils import _get_src_id, get_list
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.sourceconverter import SourceGroup
from openquake.hazardlib.nrml import SourceModel
from openquake.mbt.tools.mfd import EEvenlyDiscretizedMFD
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.source import (
        SimpleFaultSource, MultiPointSource, AreaSource, PointSource,
        BaseSeismicSource, MultiFaultSource)
from openquake.hazardlib.geo.surface import SimpleFaultSurface
from openquake.hazardlib.mfd.multi_mfd import MultiMFD
from openquake.hazardlib.pmf import PMF

PLOTTING = False


def get_bounding_box(sfc):
    """
    Get the bounding box of a simple fault source

    :param src:
        See :method:`get_data`
    :returns:
        A list with four floats i.e. the coordinates of the lower left and
        upper right corners of the bounding box.
    """

    breakpoint()
    # This provides the convex hull of the surface projection
    coo = np.array(src.polygon.coords)
    return [min(coo[:, 0]), min(coo[:, 1]), max(coo[:, 0]), max(coo[:, 1])]


def get_data(sfc, coo_pnt_src, pnt_srcs, dist_type='rjb', buffer=1.0):
    """
    Computes point sources within the bounding box and the corresponding
    rjb distances.

    :param sfc:
        An instance of :class:`openquake.hazardlib.geo.surface.BaseSurface`
    :param coo_pnt_src:
        An array with the coordinates of the point sources
    :param pnt_srcs:
        A list of :class:`openquake.hazardlib.source.PointSource`
    :param dist_type:
        A string specifying the metric used to measure the distance between the
        fault plane and the point sources
    :param buffer:
        A float [km] indicating the threshold distance within which point
        sources are considered within the buffer surrounding the fault.
    """

    # Get the bounding box
    if dist_type == 'rjb':

        # Bounding box
        bbox = sfc.get_bounding_box()

        # Find the point sources within the extended buffer arround the fault
        # bounding box
        idxs = np.nonzero((coo_pnt_src[:, 0] > bbox[0] - buffer) &
                          (coo_pnt_src[:, 1] > bbox[1] - buffer) &
                          (coo_pnt_src[:, 0] < bbox[2] + buffer) &
                          (coo_pnt_src[:, 1] < bbox[3] + buffer))[0]
        sel_pnt_srcs = [pnt_srcs[i] for i in idxs]

        # No points selected
        if len(sel_pnt_srcs) < 1:
            return None, None, None, None

        # Coordinates of the selected points i.e. points within the bounding
        # box plus of the fault plus a buffers
        sel_pnt_coo = np.array([(p.location.longitude, p.location.latitude)
                                for p in sel_pnt_srcs])

        # Create the mesh
        mesh = Mesh(sel_pnt_coo[:, 0], sel_pnt_coo[:, 1])

        # Compute rjb
        dist = sfc.get_joyner_boore_distance(mesh)

    elif dist_type == 'rrup':

        # Create the mesh
        lld = pnt_srcs.location
        lld.depth = pnt_srcs.hypocenter_distribution.data[0][1]
        mesh = Mesh.from_points_list([lld])
        idxs, sel_pnt_srcs, sel_pnt_coo = [], [], []
        dist = sfc.get_min_distance(mesh)

    return idxs, sel_pnt_srcs, sel_pnt_coo, dist


def get_stacked_mfd(srcs: list, within_idx: list, binw: float):
    """
    This returns a stacked MFD for the sources in the `srcs` provided as
    input.

    :param srcs:
        A list of sources
    :param within_idx:
        A list with the indexes of the sources in `srcs` whose mfd must be used
        in the stacking
    :param binw:
        A float indicating the bin width of the
    """
    for i, idx in enumerate(within_idx):
        if i == 0:
            tot_mfd = EEvenlyDiscretizedMFD.from_mfd(srcs[idx].mfd, binw)
        else:
            tot_mfd.stack(srcs[idx].mfd)
    return tot_mfd


def explode(srcs):
    """
    Takes sources with hypocentral depth distribution and divides them into
    one source for each depth
    """
    exploded_srcs = []
    for src in srcs:
        hpd = src.hypocenter_distribution.data
        for h in hpd:
            nsrc = copy.deepcopy(src)
            dep = h[1]
            wei = h[0]
            if 'TruncatedGRMFD' in str(type(src.mfd)):
                nsrc.mfd.a_val = wei * src.mfd.a_val
                nsrc.hypocenter_distribution = PMF([(1.0, dep)])
            else:
                msg = 'Not implementd for MFD of type {}'.format(src.mfd)
                raise ValueError(msg)
            exploded_srcs.append(nsrc)

    return exploded_srcs


def remove_buffer_around_faults(fname: str, path_point_sources: str,
                                out_path: str, dst: float,
                                threshold_mag: float = 6.5, use: str = ''):
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
    out_path = pathlib.Path(out_path)

    if len(use) > 0:
        use = get_list(use)

    # Create a source converter
    binw = 0.1
    sourceconv = SourceConverter(investigation_time=1.0,
                                 rupture_mesh_spacing=5.0,
                                 complex_fault_mesh_spacing=5.0,
                                 width_of_mfd_bin=binw,
                                 area_source_discretization=5.0,
                                 )

    # Get the surfaces representing the faults
    faults = _get_fault_surfaces(fname, sourceconv)

    # Process the point sources in the distributed seismicity model
    for point_fname in glob.glob(path_point_sources):

        coo_pnt_src = []
        pnt_srcs = []

        # Info
        logging.info(f'Processing: {point_fname}')

        # Check if the source must be processed
        src_id = _get_src_id(point_fname)
        if len(use) > 0 and src_id not in use:
            logging.info(f'Skipping {point_fname}')
            continue

        # Read the file content
        tssm = to_python(point_fname, sourceconv)

        # Get the point sources used to model distributed seismicity
        wsrc = _get_point_sources(tssm)

        # Create an array with the coordinates of the point sources
        tcoo = np.array([(p.location.longitude, p.location.latitude) for p in
                         wsrc])
        pnt_srcs.extend(wsrc)
        coo_pnt_src.extend(tcoo)
        coo_pnt_src = np.array(coo_pnt_src)

        if PLOTTING:
            fig, axs = plt.subplots(1, 1)
            plt.plot(coo_pnt_src[:, 0], coo_pnt_src[:, 1], '.')

        # Processing faults
        buffer_pts = []
        bco = []
        for src in faults:

            # Getting the subset of point sources in the surrounding of the
            # fault `src`. `coo_pnt_src` is a numpy.array with two columns
            # (i.e. lon and lat). `pnt_srcs` is a list containing the point
            # sources that collectively describe the distributed seismicity
            # sources provided as input.
            pnt_ii, sel_pnt_srcs, sel_pnt_coo, rjb = get_data(
                    src, coo_pnt_src, pnt_srcs)

            # If we find some point sources around the fault
            if pnt_ii is not None:

                # Find the index of points within the buffer zone
                within_idx = np.nonzero(rjb < dst)[0]
                idxs = sorted([pnt_ii[i] for i in within_idx], reverse=True)

                if PLOTTING:
                    plt.plot(coo_pnt_src[idxs, 0], coo_pnt_src[idxs, 1], 'or',
                             mfc='none')

                # Loop over the indexes of the point sources within the buffer
                for isrc in idxs:

                    # Explode sources. i.e. create individual point sources at
                    # each individual hypocentral depth
                    pnt_srcs_exp = explode(pnt_srcs[isrc])

                    # Check which of the individual point sources are within
                    # the buffer
                    cnt = 0
                    for pnt_src_exp in pnt_srcs_exp:
                        _, _, _, rrup = get_data(src, [], pnt_src_exp,
                                                 dist_type='rrup')

                        # Updating mmax for the point source
                        if rrup < dst:
                            pnt_src_exp.mfd.max_mag = threshold_mag

                    # Adding point sources to the buffer
                    buffer_pts.extend(pnt_srcs_exp)
                    bco.append([coo_pnt_src[isrc, 0], coo_pnt_src[isrc, 1]])

                    # Removing the point source from the list of sources
                    # outside of buffers
                    pnt_srcs.remove(pnt_srcs[isrc])

                # Update the array containing the coordinates of the point 
                # sources 
                mask = np.ones(len(coo_pnt_src), dtype=bool)
                mask[pnt_ii[within_idx]] = False
                coo_pnt_src = coo_pnt_src[mask, :]

            else:
                continue

        # Plot option used for debugging
        if PLOTTING:
            bco = np.array(bco)
            plt.plot(bco[:, 0], bco[:, 1], 'x')
            plt.show()

        # Create the multi-point source
        tmpsrc = from_list_ps_to_multipoint(pnt_srcs, 'pnts')

        # Save the multipoint source to a nrml file
        tmp = pathlib.Path(fname)
        tmp_name = f"src_points_{tmp.stem.split('_')[-1]}.xml"
        fname_out = out_path / tmp_name
        write_source_model(fname_out, [tmpsrc], 'Distributed seismicity')
        logging.info(f'Created: {fname_out}')

        # Currently must print the buffer points as single point sources, then
        # upgrade nrml because the function below won't handle correctly the
        # hypocentral distribution
        tmp_name = f"src_buffers_{tmp.stem.split('_')[-1]}.xml"
        fname_out =  out_path / tmp_name

        # Save the point sources within the buffers to a nrml file
        if buffer_pts:
            write_source_model(fname_out, buffer_pts, 'Distributed seismicity')
            logging.info(f'Created: {fname_out}')


def _get_fault_surfaces(fname: str, sourceconv: SourceConverter) -> list:
    """
    :param fname:
    :param sourceconv:
        An instance of the class
        :class:`openquake.hazardlib.sourceconverter.SourceConverter`
    """

    fname = pathlib.Path(fname)

    # Read file the fault sources
    ssm_faults = to_python(fname, sourceconv)

    # Check content of the seismic source model. We want only one group.
    msg = 'The seismic source model for fault contains more than one group'
    assert len(ssm_faults) == 1

    # Read sections in case of a multi fault source.
    fname = pathlib.Path(str(fname.parent / fname.stem) + '_sections.xml')
    if fname.exists():
        geom = to_python(fname, sourceconv)
        ssm_faults[0][0].sections = geom

    # Create surfaces
    surfaces = []
    for src in ssm_faults[0]:
        if isinstance(src, SimpleFaultSource):
            sfc = SimpleFaultSurface.from_fault_data(
                    src.fault_trace, src.upper_seismogenic_depth,
                    src.lower_seismogenic_depth, src.dip, 1.0)
            surfaces.append(sfc)
        elif isinstance(src, MultiFaultSource):
            for key in src.sections.sections:
                surfaces.append(src.sections.sections[key])
        else:
            raise ValueError('Not supported fault type')
    return surfaces


def _get_point_sources(tssm):

    # Create a list of groups 
    grps = []
    if isinstance(tssm, SourceModel):
        grps = [grp for grp in tssm]
    elif isinstance(tssm, SourceGroup):
        grps = [tssm]
    elif isinstance(tssm, BaseSeismicSource):
        grps = [[tssm]]

    wsrc = []
    for grp in grps:
        for src in grp:
            # Convert the multi-point source into a list of point sources
            if isinstance(src, (MultiPointSource, AreaSource)):
                tmp = [s for s in src]
                tmpmx = np.max([s.mfd.get_min_max_mag()[1] +
                                s.mfd.bin_width/2 for s in tmp])
                msg = f'Reading source {src.source_id}: {len(tmp)} points'
                msg += f' max mag {tmpmx}'
                logging.info(msg)
                wsrc.extend(tmp)
            elif isinstance(src, PointSource):
                wsrc.append(src)
            else:
                msg = f'{type(src)} not supported'
                raise ValueError(msg)
    return wsrc


def from_list_ps_to_multipoint(srcs: list, src_id: str):
    """
    Converts a list of point sources into a multi-point source

    :param srcs:
        A list of point sources
    :param src_id:
        The ID of the multipoint source created
    """

    # Looping over the points
    lons = []
    lats = []
    avals = []
    mmaxs = []
    settings = False

    for src in srcs:

        minmaxmag = src.get_min_max_mag()
        mmx = minmaxmag[1] + src.mfd.bin_width/2
        mmin = minmaxmag[0] - src.mfd.bin_width/2

        # Update list
        avals.append(src.mfd.a_val)
        mmaxs.append(mmx)

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

    # Set maximum magnitude
    mmaxs = [mmaxs[0]] if np.all(np.abs(np.diff(mmaxs)) < 0.01) else mmaxs

    # Instantiate the multi MFD
    name = src_id
    mmfd = MultiMFD('truncGutenbergRichterMFD',
                    size=len(avals),
                    min_mag=[mmin],
                    max_mag=mmaxs,
                    bin_width=[src.mfd.bin_width],
                    b_val=[src.mfd.b_val],
                    a_val=avals)

    # Set a temporal occurrence model
    tom = PoissonTOM(1)

    # Instantiate the multi-point source
    mesh = Mesh(np.array(lons), np.array(lats))
    srcmp = MultiPointSource(src_id, name, trt, mmfd, msr, rar, usd, lsd,
                             npd, hyd, mesh, tom)

    return srcmp
