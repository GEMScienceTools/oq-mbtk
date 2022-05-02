#!/usr/bin/env python
# coding: utf-8
# ------------------- The Model Building Toolkit ------------------------------
#
# Copyright (C) 2022 GEM Foundation
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

import h5py
import numpy as np
from openquake.wkf.utils import create_folder
from openquake.baselib import sap
from openquake.mbt.tools.model_building.dclustering import decluster
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue


def catalogue_declustering(fname: str, output_folder: str,
                           subcatalogues: bool = False):
    """
    Declusters a catalogue using a standard set of declustering algorithms.
    """

    create_folder(output_folder)
    create_folder('./tmp')

    # Create a fake file with the classification.
    tr_fname = './tmp/fake.hdf5'
    cat = _load_catalogue(fname)
    label = np.ones_like(np.array(cat['magnitude']))
    f = h5py.File(tr_fname, 'w')
    _ = f.create_dataset("undef", data=label)
    f.close()

    labels = ['undef']

    # Declustering with the classical GK algorithm
    declustering_meth = 'GardnerKnopoffType1'
    declustering_params = {'time_distance_window': 'GardnerKnopoffWindow',
                           'fs_time_prop': 0.9}
    out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=subcatalogues,
                    olab='gk',
                    save_af=True,
                    fix_defaults=True)

    declustering_meth = 'GardnerKnopoffType1'
    declustering_params = {'time_distance_window': 'UhrhammerWindow',
                           'fs_time_prop': 0.9}
    out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=subcatalogues,
                    olab='uh',
                    save_af=True,
                    fix_defaults=True)


    declustering_meth = 'GardnerKnopoffType1'
    declustering_params = {'time_distance_window': 'GruenthalWindow',
                           'fs_time_prop': 0.9}
    _ = decluster(fname,
                  declustering_meth,
                  declustering_params,
                  output_folder,
                  labels=labels,
                  tr_fname=tr_fname,
                  subcatalogues=subcatalogues,
                  olab='gr',
                  save_af=True,
                  fix_defaults=True)


def main(fname: str, output_folder: str, *, subcatalogues: bool = False):
    """
    Creates catalogues with mainshocks and after/foreshocks using the GK
    algorithm and three declustering windows:
        - Gardner and Knopoff  : gk
        - Urhammer             : uh
        - Gruenthal            : gr
    """
    catalogue_declustering(fname, output_folder, subcatalogues)


main.fname = 'Name of the .csv formatted catalogue'
main.output_folder = 'Path to the output folder'
msg = 'Boolean, when true it creates subcatalogues'
main.subcatalogues = msg

if __name__ == '__main__':
    sap.run(main)
