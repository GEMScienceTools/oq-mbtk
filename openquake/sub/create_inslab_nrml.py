#!/usr/bin/env python

"""
:module:`openquake.sub.create_inslab_nrml` creates a set of .xml input files
for the OpenQuake Engine.
"""

import os
import re
from decimal import Decimal, getcontext
import logging
import h5py
import numpy as np

from openquake.baselib import sap
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.const import TRT
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.nrml import SourceModel
from openquake.hazardlib.source import BaseRupture
from openquake.hazardlib.sourceconverter import SourceGroup
from openquake.hazardlib.source import NonParametricSeismicSource
from openquake.hazardlib.geo.surface.gridded import GriddedSurface
from openquake.hazardlib.sourcewriter import write_source_model

getcontext().prec = 10


def create_source(rup, mag, sid, name, tectonic_region_type):
    """
    :param rup:
        A h5py dataset with the rupture information
    :param mag:
        The magnitude of the ruptures
    :param sid:
        Source ID
    :param name:
        Name of the source
    :param tectonic_region_type:
        Tectonic region type
    """
    data = []
    for key in rup.keys():
        d = rup[key][:]

        # Creating the surface
        llo = np.squeeze(d['lons'])
        lla = np.squeeze(d['lats'])
        lde = np.squeeze(d['deps'])

        # Create the gridded surface
        if len(llo.shape) > 0:

            # Hypocenter computed in the 'rupture.py' module
            hypo = np.squeeze(d['hypo'][:])
            hlo = hypo[0]
            hla = hypo[1]
            hde = hypo[2]

            # Probabilities of occurrence
            ppp = np.squeeze(d['prbs'])
            i = np.isfinite(llo)
            points = [Point(x, y, z) for x, y, z in
                      zip(llo[i], lla[i], lde[i])]

            # Create the surface and the rupture
            srf = GriddedSurface.from_points_list(points)
            brup = BaseRupture(mag=mag, rake=-90.,
                               tectonic_region_type=tectonic_region_type,
                               hypocenter=Point(hlo, hla, hde),
                               surface=srf)
            xxx = Decimal('{:.8f}'.format(ppp[1]))
            pmf = PMF(data=[((Decimal('1')-xxx), 0), (xxx, 1)])
            data.append((brup, pmf))
    src = NonParametricSeismicSource(sid, name, tectonic_region_type, data)
    return src


def create(label, rupture_hdf5_fname, output_folder, investigation_t,
           trt=TRT.SUBDUCTION_INTRASLAB):
    """
    :param label:
        A string identifying the source
    :param rupture_hdf5_fname:
        Name of the .hdf5 file containing the ruptures
    :param output_folder:
        Folder where to write the .xl files
    :param investigation_t:
        Investigation time in years
    :param trt:
        Tectonic region type label
    """

    # Open the input .hdf5 file with the ruptures
    f = h5py.File(rupture_hdf5_fname, 'r')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Create xml
    for mag in f['ruptures'].keys():

        # Check the number of ruptures defined for the current magnitude value
        grp = f['ruptures'][mag]
        if len(grp) < 1:
            tmps = 'Skipping ruptures for magnitude {:.2f}'.format(float(mag))
            logging.warning(tmps)
            continue

        # Set the name of the output nrml file
        fxml = os.path.join(output_folder, '{:s}.xml'.format(mag))

        # Set the source ID
        mags = re.sub('\\.', 'pt', mag)
        sid = 'src_{:s}_{:s}'.format(label, mags)
        name = 'Ruptures for mag bin {:s}'.format(mags)

        # Creates a non-parametric seismic source
        src = create_source(grp, float(mag), sid, name, trt)

        # Create source group
        sgrp = SourceGroup(trt, [src])

        # Create source model
        name = 'Source model for {:s} magnitude {:s}'.format(label, mags)
        mdl = SourceModel([sgrp], name, investigation_t)

        # Write source model
        write_source_model(fxml, mdl, mag)

    f.close()


create.label = 'TR label'
create.rupture_hdf5_fname = 'hdf5 file with the ruptures'
create.output_folder = 'Name of the output folder'
create.investigation_t = 'Investigation time'

if __name__ == '__main__':
    sap.run(create)
