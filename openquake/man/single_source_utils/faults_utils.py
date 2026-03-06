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

import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.source import CharacteristicFaultSource
from openquake.hazardlib.geo.surface import SimpleFaultSurface
from openquake.hazardlib.geo.geodetic import min_geodetic_distance


def _get_mesh_ch_from_char_fs(fault, mesh_spacing):
    """
    :parameter fault:
        An instance of the
        :class:`openquake.hazardlib.source.CharacteristicFaultSource`
    :parameter float mesh_spacing:
        The spacing [km] of the grid used to represent the fault surface
    """
    srfa = fault.surface
    mesh = srfa.get_mesh()
    chul = mesh.get_convex_hull()
    return mesh, chul


def _get_mesh_ch_from_simple_fs(fault, mesh_spacing):
    """
    :parameter fault:
        An instance of the
        :class:`openquake.hazardlib.source.CharacteristicFaultSource`
    :parameter float mesh_spacing:
        The spacing [km] of the grid used to represent the fault surface
    """
    srfa = SimpleFaultSurface.from_fault_data(fault.fault_trace,
                                              fault.upper_seismogenic_depth,
                                              fault.lower_seismogenic_depth,
                                              fault.dip,
                                              mesh_spacing)
    mesh = srfa.mesh
    chul = mesh.get_convex_hull()
    return mesh, chul


def fault_surface_distance(srcs, mesh_spacing, types=None, dmax=40):
    """
    :parameter srcs:
        A list of openquake sources
    :parameter float mesh_spacing:
        The spacing [km] of the grid used to represent the fault surface
    :parameter types:
        The type of sources to be analysed
    :parameter dmax:
    """
    # Fault distance array
    lnkg = np.ones((len(srcs), len(srcs)))*dmax
    
    # Check input information
    if types is None:
        types = (SimpleFaultSource, CharacteristicFaultSource)
    
    # Process srcs
    for idxa in range(0, len(srcs)):
        srca = srcs[idxa]
        if isinstance(srca, types):
            
            # Set the mesh for the first fault surface
            if isinstance(srca, SimpleFaultSource):
                meshA, chA = _get_mesh_ch_from_simple_fs(srca, mesh_spacing)
            elif isinstance(srca, CharacteristicFaultSource):
                meshA, chA = _get_mesh_ch_from_char_fs(srca, mesh_spacing)
            else:
                raise ValueError('Unsupported fault type')
            
            # Second loop
            for idxB in range(idxa, len(srcs)):
                srcb = srcs[idxB]
                if isinstance(srcb, types):
                    
                    # Set the mesh for the second fault surface
                    if isinstance(srcb, SimpleFaultSource):
                        meshB, chB = _get_mesh_ch_from_simple_fs(srcb,
                                                                 mesh_spacing)
                    elif isinstance(srcb, CharacteristicFaultSource):
                        meshB, chB = _get_mesh_ch_from_char_fs(srcb,
                                                               mesh_spacing)
                    else:
                        raise ValueError('Unsupported fault type')
                    
                    # Calculate the distance between the two bounding boxes
                    la = [(a, b) for a, b in zip(chA.lons, chA.lats)]
                    lb = [(a, b) for a, b in zip(chB.lons, chB.lats)]
                    tmpd = min_geodetic_distance(la, lb)
                    
                    # Calculate the distance between the two fault surfaces if required
                    if (np.amin(tmpd) > dmax):
                        mindst = np.amin(tmpd)
                    else:
                        mindst = np.amin(meshA.get_min_distance(meshB))
                    
                    # Update the array
                    lnkg[idxa, idxB] = mindst
    
    # Check that the size of the linkage matrix corresponds
    # to the size of the initial list of fault sources
    assert len(lnkg) == len(srcs)
    return lnkg


def plt_mmax_area(model):
    """
    :parameter model:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    mmaxs = get_mmax(model)
    areas = get_areas(model)
    plt.plot(mmaxs, areas, 'o')
    plt.xlabel(r'Maximum magnitude')
    plt.ylabel(r'Fault surface area [km2]')
    plt.yscale('log')
    plt.grid(which='both', linestyle='--')
    text = plt.text(0.025, 0.95,
                    'maximum magnitude {0:.2f}'.format(max(mmaxs)),
                    transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                  foreground="white")])
    return fig


def get_areas(model):
    """
    Get maximum magnitudes from MFDs

    :parameter model:
        A list of hazardlib source instances
    :returns:
        A list with the areas of each fault
    """
    msp = 2
    areas = []
    for src in model:
        if isinstance(src, SimpleFaultSource):
            fault_surface = SimpleFaultSurface.from_fault_data(
                                                src.fault_trace,
                                                src.upper_seismogenic_depth,
                                                src.lower_seismogenic_depth,
                                                src.dip,
                                                msp)
            area = fault_surface.get_area()
            areas.append(area)
            
    return areas


def plt_mmax_length(model):
    """
    :parameter model:
        A list of :class:`openquake.hazardlib.source` instances
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    mmaxs = get_mmax(model)
    lngths = get_lengths(model)
    plt.plot(mmaxs, lngths, 'o')
    plt.xlabel(r'Maximum magnitude')
    plt.ylabel(r'Fault length [km]')
    plt.yscale('log')
    plt.grid(which='both', linestyle='--')
    text = plt.text(0.025, 0.95,
                    'maximum magnitude {0:.2f}'.format(max(mmaxs)),
                    transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                  foreground="yellow")])
    return fig


def get_mmax(model):
    """
    Get maximum magnitudes from MFDs

    :parameter model:
        A list of hazardlib source instances
    :returns:
        A list with the maximum magnitude values
    """
    mmaxs = []
    for src in model:
        if isinstance(src, SimpleFaultSource):
            min_mag, max_mag = src.mfd.get_min_max_mag()
            mmaxs.append(max_mag)
    return mmaxs


def _add_mmax_histogram(mmaxs):
    binw = 0.5
    mmin = np.floor(min(mmaxs)/binw)*binw
    mmax = np.ceil(max(mmaxs)/binw)*binw
    edges = np.arange(mmin, mmax+0.01*binw, step=binw)
    hist, bin_edges = np.histogram(mmaxs, bins=edges)
    plt.hist(mmaxs, bins=bin_edges, log=True, edgecolor='white')


def plt_mmax_histogram(mmaxs):
    """
    :parameter mmaxs:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    _add_mmax_histogram(mmaxs)
    plt.xlabel(r'Maximum magnitude')
    plt.grid(which='both', linestyle='--')
    text = plt.text(0.7, 0.95, '{0:d} sources'.format(len(mmaxs)),
                    transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3,
                                                  foreground="y")])
    return fig


def get_lengths(model, trt='.*'):
    """
    Compute the length of all the simple fault sources

    :parameter model:
        A list of hazardlib source instances
    :returns:
        A dictionary with the length [km] for all the simple fault sources
        included in the model.
    """
    lngths = {}
    for src in model:
        if (isinstance(src, SimpleFaultSource) and
                re.search(trt, src.tectonic_region_type)):
            lngths[src.source_id] = src.fault_trace.get_length()
        else:
            logging.warning('unsupported fault type %s', type(src).__name__)
    return lngths


def _add_length_histogram(flens):
    """
    This adds an histogram to

    :parameter flens:
        A list of values representing the lengths of fault traces
    """
    hist, bin_edges = np.histogram(flens)
    plt.hist(flens, bins=bin_edges, log=True, edgecolor='white')


def plt_length_histogram(flens):
    """
    :parameter flens:
        A list of values representing the lengths of fault traces
    :returns:
        A matplotlib figure
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    _add_length_histogram(flens)
    plt.xlabel(r'Fault length [km]')
    plt.grid(which='both', linestyle='--')
    plt.text(0.7, 0.95, '# of sources: {0:d}'.format(len(flens)),
             transform=ax.transAxes)
    return fig
