#!/usr/bin/env python

import os
import toml

from openquake.baselib import sap
from openquake.sub.edges_set import EdgesSet

from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.wkf.utils import create_folder


def main(fname_config, label, edges_folder, out_file, *, resampling=None):
    """
    Creates the .xml input for the interface sources
    """

    create_folder(os.path.dirname(out_file))

    # check edges folder
    assert os.path.exists(edges_folder)

    # Read the config file
    config = toml.load(fname_config)

    # Create .xml
    es = EdgesSet.from_files(edges_folder)
    src = es.get_complex_fault(section_length=float(resampling))

    binw = config['bin_width']
    agr = config['sources'][label]['agr']
    bgr = config['sources'][label]['bgr']
    mmin = config['mmin']
    mmax = config['sources'][label]['mmax']
    mfd = TruncatedGRMFD(min_mag=mmin, max_mag=mmax, bin_width=binw,
                         a_val=agr, b_val=bgr)
    src.mfd = mfd
    src.rupture_mesh_spacing = 10.0

    write_source_model(out_file, [src], 'Name')


descr = 'The path to the configuration file'
main.fname_config = descr
descr = 'The label identifying the source'
main.label = descr
descr = 'The path to the folder with the edges'
main.edges_folder = descr
descr = 'The path to the output file'
main.out_file = descr
descr = 'Edge resampling distance [km]'
main.resampling = descr


if __name__ == '__main__':
    sap.run(main)
