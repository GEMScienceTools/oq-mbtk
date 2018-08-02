#!/usr/bin/env python3.5

import os
import re
import sys
import h5py
import logging
import numpy as np
from openquake.baselib import sap
from openquake.hazardlib.const import TRT
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.surface.gridded import GriddedSurface
from openquake.hazardlib.source import NonParametricSeismicSource
from openquake.hazardlib.source import BaseRupture
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.sourcewriter import write_source_model

from openquake.hazardlib.nrml import SourceModel
from openquake.hazardlib.sourceconverter import SourceGroup

from decimal import Decimal, getcontext
getcontext().prec = 10


def create_nrml_source(rup, mag, sid, name, tectonic_region_type):
    """
    :param rup:
    :param mag:
    :param sid:
    :param name:
    :param tectonic_region_type:
    """
    data = []
    for key in rup.keys():
        d = rup[key][:]
        #
        # creating the surface
        llo = np.squeeze(d['lons'])
        lla = np.squeeze(d['lats'])
        lde = np.squeeze(d['deps'])
        #
        # find a node in the middle of the rupture
        if len(llo.shape):
            ihyp = (int(np.round(llo.shape[0]/2)))
            if len(llo.shape) > 1:
                ihyp = (ihyp, int(np.round(llo.shape[1]/2)))
            hlo = llo[ihyp]
            hla = lla[ihyp]
            hde = lde[ihyp]
            #
            #
            ppp = np.squeeze(d['prbs'])
            i = np.isfinite(llo)
            points = [Point(x, y, z) for x, y, z in
                      zip(llo[i], lla[i], lde[i])]
            srf = GriddedSurface.from_points_list(points)
            """
            br = BaseRupture(mag=mag,
                             rake=-90.,
                             tectonic_region_type=tectonic_region_type,
                             hypocenter=Point(hlo, hla, hde),
                             surface=srf,
                             source_typology=NonParametricSeismicSource)
            """
            br = BaseRupture(mag=mag,
                             rake=-90.,
                             tectonic_region_type=tectonic_region_type,
                             hypocenter=Point(hlo, hla, hde),
                             surface=srf)
            xxx = Decimal('{:.8f}'.format(ppp[1]))
            pmf = PMF(data=[((Decimal('1')-xxx), 0), (xxx, 1)])
            data.append((br, pmf))
    src = NonParametricSeismicSource(sid, name, tectonic_region_type, data)
    return src


def create(label, rupture_hdf5_fname, output_folder, investigation_t):
    """
    :param label:
    :param rupture_hdf5_fname:
    :param output_folder:
    """
    #
    #
    f = h5py.File(rupture_hdf5_fname, 'r')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    #
    #
    trt = TRT.SUBDUCTION_INTRASLAB
    for mag in f['ruptures'].keys():
        #
        # check the number of ruptures defined for the current magnitude value
        grp = f['ruptures'][mag]
        if len(grp) < 1:
            tmps = 'Skipping ruptures for magnitude {:.2f}'.format(float(mag))
            print(tmps)
            logging.warning(tmps)
            continue
        #
        # set the name of the output nrml file
        fnrml = os.path.join(output_folder, '{:s}.nrml'.format(mag))
        #
        # source ID
        mags = re.sub('\.', 'pt', mag)
        sid = 'src_{:s}_{:s}'.format(label, mags)
        name = 'Ruptures for mag bin {:s}'.format(mags)
        #
        # creates a non-parametric seismic source
        src = create_nrml_source(grp, float(mag), sid, name, trt)
        #
        # create source group
        sgrp = SourceGroup(trt, [src])
        #
        # create source model
        name = 'Source model for {:s} magnitude {:s}'.format(label, mags)
        mdl = SourceModel([sgrp], name, investigation_t)
        #
        # write source model
        write_source_model(fnrml, mdl, mag)
    f.close()
    print('Done')


def main(argv):
    p = sap.Script(create)
    p.arg(name='label', help='TR label')
    p.arg(name='rupture_hdf5_fname', help='hdf5 file with the ruptures')
    p.arg(name='output_folder', help='Name of the output folder')
    p.arg(name='investigation_t', help='Investigation time')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == '__main__':
    main(sys.argv[1:])
