#!/usr/bin/env python3

import os
from openquake.baselib import sap
from openquake.cat.hmg.hmg import process_dfs


def main(settings, odf_fname, mdf_fname, outfolder='./h5/'):
    """
    The hom function reads the information in the settings file and creates
    a homogenised catalogue following the rules there defined.
    """

    # Homogenise
    save, work = process_dfs(odf_fname, mdf_fname, settings)

    # Outname
    fname = os.path.basename(odf_fname).split('_')[0]

    # Saving results
    fmt = '{:s}_catalogue_homogenised.h5'
    tmp = os.path.join(outfolder, fmt.format(fname))
    save.to_hdf(tmp, '/events', append=False)

    fmt = '{:s}_leftout.h5'
    tmp = os.path.join(outfolder, fmt.format(fname))
    work.to_hdf(tmp, '/events', append=False)


main.settings = '.toml file with the settings'
main.odf_fname = '.h5 file with origins'
main.mdf_fname = '.h5 file with magnitudes'
main.outfolder = 'output folder'

if __name__ == "__main__":
    sap.run(main)
