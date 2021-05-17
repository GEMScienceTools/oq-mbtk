#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
from openquake.wkf.utils import create_folder
from openquake.baselib import sap
from openquake.mbt.tools.model_building.dclustering import decluster
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue


def catalogue_declustering(fname: str, output_folder: str):
    """
    Decluster a catalogue
    """

    create_folder(output_folder)
    create_folder('./tmp')

    # Create a fake file with the classification. We use a fake classification
    # since earthquakes in this analysis are just from stable crust
    tr_fname = './tmp/fake.hdf5'
    cat = _load_catalogue(fname)
    label = np.ones_like(np.array(cat['magnitude']))
    f = h5py.File(tr_fname, 'w')
    _ = f.create_dataset("stable", data=label)
    f.close()

    # Declustering with the classical GK algorithm
    declustering_meth = 'GardnerKnopoffType1'
    declustering_params = {'time_distance_window': 'GardnerKnopoffWindow',
                           'fs_time_prop': 0.9}
    labels = ['stable']
    out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=True,
                    olab='_gk_whole',
                    save_af=True,
                    fix_defaults=True)

    declustering_meth = 'GardnerKnopoffType1'
    declustering_params = {'time_distance_window': 'UhrhammerWindow',
                           'fs_time_prop': 0.9}
    labels = ['stable']
    out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=True,
                    olab='_uh_whole',
                    save_af=True,
                    fix_defaults=True)


    declustering_meth = 'GardnerKnopoffType1'
    declustering_params = {'time_distance_window': 'GruenthalWindow',
                           'fs_time_prop': 0.9}
    labels = ['stable']
    out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=True,
                    olab='_gr_whole',
                    save_af=True,
                    fix_defaults=True)


catalogue_declustering.calc_ref = 'first calculation'
catalogue_declustering.calc = 'second calculation'
catalogue_declustering.nsigma = 'tolerance as number of sigma'

if __name__ == '__main__':
    sap.run(catalogue_declustering)
