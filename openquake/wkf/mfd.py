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

import toml
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from openquake.wkf.utils import _get_src_id
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.mbt.tools.mfd import (
    EEvenlyDiscretizedMFD,
    get_evenlyDiscretizedMFD_from_truncatedGRMFD)


def check_mfds(fname_input_pattern: str, fname_config: str = None,  *,
               src_id: str = None):
    """
    Given a set of .xml files and a configuration file with GR params, this
    code compares the total MFD of the sources against the original one in the
    configuration file. The ID of the source if not provided is taken from the
    name of the files (i.e., last label preceded by `_`)
    
    :param fname_input_pattern:
    	pattern to match for xml files
    :param fname_config:
    	configuration file with GR params to compare
    """
    for fname in sorted(glob(fname_input_pattern)):

        srcs = _get_src_id(fname)
        if src_id is not None and srcs not in src_id:
        	continue
        

        binw = 0.1
        sourceconv = SourceConverter(investigation_time=1.0,
                                     rupture_mesh_spacing=5.0,
                                     complex_fault_mesh_spacing=5.0,
                                     width_of_mfd_bin=binw)
        ssm = to_python(fname, sourceconv)

        for grp in ssm:
            for i, src in enumerate(grp):
                if i == 0:
                    nmfd = EEvenlyDiscretizedMFD.from_mfd(src.mfd, binw)
                else:
                    ged = get_evenlyDiscretizedMFD_from_truncatedGRMFD
                    tmfd = ged(src.mfd, nmfd.bin_width)
                    nmfd.stack(tmfd)
            
            occ = np.array(nmfd.get_annual_occurrence_rates())

            if fname_config:
                model = toml.load(fname_config)
                bgr = model["sources"][srcs]["bgr"]
                agr = model["sources"][srcs]["agr"]
                tmp = occ[:, 0] - binw
                mfd = 10.0**(agr-bgr*tmp[:-1])-10.0**(agr-bgr*(tmp[:-1]+binw))

            _ = plt.figure(figsize=(8, 6))
            plt.plot(occ[:, 0], occ[:, 1], 'o', 
                label = 'model occurrence rate for source')
            if fname_config:
                plt.plot(tmp[:-1]+binw/2, mfd, 'x', 
                    label = ('config: a = ', agr, ", b = ", bgr))
            plt.title(fname)
            plt.xlabel('Magnitude')
            plt.ylabel('Annual occurrence rate')
            plt.yscale('log')
            plt.legend()
            plt.show()
