#!/usr/bin/env python

import os
import h5py
import numpy
import pickle
import importlib

from pathlib import Path
from oqmbt.tools.model_building.plt_tools import _load_catalogue
from openquake.hmtk.seismicity.selector import CatalogueSelector


def decluster(catalogue_hmtk_fname, declustering_meth, declustering_params,
              output_path, label, tr_fname):
    """
    :param catalogue_hmtk_fname:
    :param declustering_meth:
    :param declustering_params:
    :param output_path:
    """
    #
    # check if the initial catalogue file exists
    assert os.path.exists(catalogue_hmtk_fname)
    #
    # Create output filename
    lbl = ''
    if label is not None:
        lbl = label
        assert tr_fname is not None
    sfx = Path(os.path.basename(catalogue_hmtk_fname)).suffix
    ext = '_dec_{:s}{:s}'.format(lbl, sfx)
    #
    # Output filename
    out_fname = Path(os.path.basename(catalogue_hmtk_fname)).stem+ext
    if output_path is not None:
        assert os.path.exists(output_path)
    else:
        output_path = os.path.dirname(catalogue_hmtk_fname)
    tmps = os.path.join(output_path, out_fname)
    out_fname = os.path.abspath(tmps)
    print(out_fname)
    #
    # Read the catalogue
    cat = _load_catalogue(catalogue_hmtk_fname)
    #
    # Select earthquakes belonging to a given TR. if combining multiple TRs, 
    # use label <TR_1>AND<TR_2>AND...
    idx = numpy.full(cat.data['magnitude'].shape, True, dtype=bool)
    if label is not None and tr_fname is not None:
        f = h5py.File(tr_fname, 'r')
        labs = label.split('AND')
        idx = numpy.array([False for i in range(len(f[labs[0]]))])
        for lab in labs:
            idx_tmp = f[lab][:]
            idx[numpy.where(idx_tmp)] = True
        f.close()
    #
    # Filter catalogue
    if label is not None:
        sel = CatalogueSelector(cat, create_copy=False)
        sel.select_catalogue(idx)
    #
    # Create declusterer
    module = importlib.import_module('openquake.hmtk.seismicity.declusterer.dec_gardner_knopoff')
    my_class = getattr(module, declustering_meth)
    declusterer = my_class()
    #
    # Declustering parameters
    # config = {'time_distance_window': distance_time_wind, 'fs_time_prop': .9}
    config = {}
    if declustering_params is not None:
        config = eval(declustering_params)
    #
    # Declustering
    from openquake.hmtk.seismicity.declusterer.distance_time_windows import GardnerKnopoffWindow
    distance_time_wind = GardnerKnopoffWindow()
    config = {'time_distance_window': distance_time_wind, 'fs_time_prop': .9}
    vcl, flag = declusterer.decluster(cat, config)
    #
    # select mainshocks
    cat.select_catalogue_events(numpy.where(flag == 0)[0])
    #
    # Create pickle
    fou = open(out_fname, 'wb')
    pickle.dump(cat, fou)
    fou.close()
