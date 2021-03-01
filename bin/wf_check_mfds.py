#!/usr/bin/env python
# coding: utf-8

import toml
import numpy as np
import matplotlib.pyplot as plt
from openquake.baselib import sap
from glob import glob
from utils import _get_src_id
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.mbt.tools.mfd import (EEvenlyDiscretizedMFD, 
        get_evenlyDiscretizedMFD_from_truncatedGRMFD)


def check_mfds(fname_input_pattern: str, fname_config: str):

    for fname in sorted(glob(fname_input_pattern)):

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
                    tmfd = get_evenlyDiscretizedMFD_from_truncatedGRMFD(src.mfd, nmfd.bin_width)
                    nmfd.stack(tmfd)

            occ = np.array(nmfd.get_annual_occurrence_rates())

            bgr = model["sources"][src_id]["bgr_weichert"]
            agr = model["sources"][src_id]["agr_weichert"]

            tmp = occ[:,0] - binw
            mfd = 10.0**(agr-bgr*tmp[:-1])-10.0**(agr-bgr*(tmp[:-1]+binw))
            
            fig = plt.figure(figsize=(8, 6))
            plt.plot(occ[:, 0], occ[:, 1], 'o')
            plt.plot(tmp[:-1]+binw/2, mfd, 'x')
            print(mfd)
            plt.title(fname) 
            plt.xlabel('Magnitude')
            plt.ylabel('Annual occurrence rate')
            plt.yscale('log')

            plt.show()

check_mfds.fname_input_pattern = "Pattern for input .csv files"
check_mfds.fname_config = "Name of the configuration file"

if __name__ == '__main__':
    sap.run(check_mfds)
