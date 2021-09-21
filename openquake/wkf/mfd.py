#!/usr/bin/env python
# coding: utf-8

import toml
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from openquake.wkf.utils import _get_src_id
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.mbt.tools.mfd import (EEvenlyDiscretizedMFD,
        get_evenlyDiscretizedMFD_from_truncatedGRMFD)


def check_mfds(fname_input_pattern: str, fname_config: str, *,
               src_id: str = None):
    """
    Given a set of .xml files and a configuration file with GR params, this
    code compares the total MFD of the sources against the original one in the
    configuration file. The ID of the source if not provided is taken from the
    name of the files (i.e., last label preceded by `_`)
    """

    for fname in sorted(glob(fname_input_pattern)):

        if src_id is None:
            src_id = _get_src_id(fname)
        model = toml.load(fname_config)

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

            bgr = model["sources"][src_id]["bgr_weichert"]
            agr = model["sources"][src_id]["agr_weichert"]

            tmp = occ[:, 0] - binw
            mfd = 10.0**(agr-bgr*tmp[:-1])-10.0**(agr-bgr*(tmp[:-1]+binw))

            _ = plt.figure(figsize=(8, 6))
            plt.plot(occ[:, 0], occ[:, 1], 'o')
            plt.plot(tmp[:-1]+binw/2, mfd, 'x')
            plt.title(fname)
            plt.xlabel('Magnitude')
            plt.ylabel('Annual occurrence rate')
            plt.yscale('log')
            plt.show()
