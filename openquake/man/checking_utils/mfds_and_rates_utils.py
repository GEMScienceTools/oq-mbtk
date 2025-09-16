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

import numpy as np
import copy
import matplotlib.pyplot as plt

from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD
from openquake.hazardlib.source.non_parametric import NonParametricSeismicSource

from openquake.man.model import read
from openquake.mbt.oqt_project import OQtProject
from openquake.mbt.tools.mfd import get_evenlyDiscretizedMFD_from_truncatedGRMFD
import openquake.mbt.tools.mfd as mfdt


# MFDs
def plot_mfd_cumulative(mfd, fig=None, label='', color=None, linewidth=1, title=''):
    aa = np.array(mfd.get_annual_occurrence_rates())
    cml = np.cumsum(aa[::-1, 1])
    if color is None:
        color = np.random.rand(3)
    plt.plot(aa[:, 0], cml[::-1], label=label, lw=linewidth, color=color)
    plt.title(title)


def plot_mfd(mfd, fig=None, label='', color=None, linewidth=1):
    bw = 0.1
    aa = np.array(mfd.get_annual_occurrence_rates())
    occs = []
    if color is None:
        color = np.random.rand(3)
    for mag, occ in mfd.get_annual_occurrence_rates():
        plt.plot([mag-bw/2, mag+bw/2], [occ, occ], lw=2, color='grey')
        occs.append(occ)
    plt.plot(aa[:, 0], aa[:, 1], label=label, lw=linewidth)


def get_total_mfd(sources, trt=None):
    """
    :param list sources:
        A list of :class:`openquake.hazardlib.source.Source` instances
    :returns:
        A :class:`openquake.man.checking_utils.mfds_and_rates_utils.EvenlyDiscretizedMFD` instance
    """
    cnt = 0
    for src in sources:
        if ((trt is not None and trt == src.tectonic_region_type) or
                (trt is None)):
            mfd = src.mfd
            if isinstance(src.mfd, TruncatedGRMFD):
                mfd = mfdt.get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd)
            if cnt == 0:
                mfdall = copy.copy(mfd)
            else:
                mfdall.stack(mfd)
            cnt += 1
    return mfdall


# Rates
def get_mags_rates(source_model_fname: str, time_span: float):
    """
    This computes the total rate for a non-parameteric source modelling the
    occurrence of a single magnitude value.

    :param str source_model_fname:
        The name of the xml shapefile
    :param float time_span:
        The time in years to which the probability of occurrence refers to
    :returns:
        A tuple with two floats. The magnitude modelled and the corresponding
        total annual rate of occurrence.
    """
    # Read the source_model
    src_model, _ = read(source_model_fname, False)

    # Process sources
    rate = 0.
    mag = None
    for src in src_model:
        if isinstance(src, NonParametricSeismicSource):
            for dat in src.data:
                rupture = dat[0]
                pmf = dat[1].data
                rate += pmf[1][0]
                if mag is None:
                    mag = rupture.mag
                else:
                    assert abs(mag-rupture.mag) < 1e-2
    return mag, rate/time_span


def mfd_from_xml(source_model_fname):
    """
    :param str source_model_fname:
        The name of the xml
    """
    # Read the source_model
    src_model, info = read(source_model_fname)
    
    return get_total_mfd(src_model) # Total mfd sources


def xml_vs_mfd(source_id, source_model_fname, model_id,
               oqmbt_project_fname):
    """
    :param str source_id:
        The ID of the source to be analysed
    :param str source_model_fname:
        The name of the xml shapefile
    :param str model_id:
        The model ID
    """    
    # Read the source_model
    src_model, info = read(source_model_fname)
    
    # Compute total mfd sources
    tmfd = get_total_mfd(src_model)
    
    # Read project
    oqtkp = OQtProject.load_from_file(oqmbt_project_fname)
    model_id = oqtkp.active_model_id
    model = oqtkp.models[model_id]
    
    # Get source mfd
    src = model.sources[source_id]
    mfd = src.mfd
    if isinstance(src.mfd, TruncatedGRMFD):
        mfd = get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd)
    
    # Compute total mfd sources
    plt.figure(figsize=(10, 8))
    plot_mfd_cumulative(tmfd)
    plot_mfd_cumulative(mfd, title=source_model_fname)
