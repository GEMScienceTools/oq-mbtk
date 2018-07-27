#!/usr/bin/env python

import os
import h5py
import numpy
import copy
import pickle
import importlib
import logging

from pathlib import Path
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.hmtk.seismicity.selector import CatalogueSelector


def _add_defaults(cat):
    """
    Adds default values for month, day, hour, minute and second

    :param cat:
        An instance of :class:`openquake.hmtk.seismicity.catalogue.Catalogue`
    :returns:
        An instance of :class:`openquake.hmtk.seismicity.catalogue.Catalogue`
    """

    for lab in ['month', 'day', 'hour', 'minute', 'second']:
        idx = numpy.isnan(cat.data[lab])
        if lab in ['day', 'month']:
            cat.data[lab][idx] = 1
        elif lab == 'second':
            cat.data[lab][idx] = 0.0
        else:
            cat.data[lab][idx] = 0
    return cat


def decluster(catalogue_hmtk_fname, declustering_meth, declustering_params,
              output_path, labels=None, tr_fname=None, subcatalogues=False,
              format='csv', olab='', save_af=False, out_fname_ext=None,
              fix_defaults=False):
    """
    :param str catalogue_hmtk_fname:
        Full path to the file containing the initial catalogue
    :param str declustering_meth:
        A string indicating the type of declustering
    :param dict declustering_params:
        Parameters required by the declustering algorithm
    :param str output_path:
        Folder where the output catalogue/s will be created
    :param list labels:
        It can be a string or a list of strings
    :param str tr_fname:
        An .hdf5 file containing the TR classification of the catalogue
    :param str format:
        Can be either 'csv' or 'pkl'
    :param str olab:
        Optional label for output catalogues
    :param boolean save_af:
        Save aftershocks and foreshocks
    :param str out_fname_ext:
        String to be added to the putput filename
    :param str fix_defaults:
        Fix defaults values when missing
    """
    #
    # check if the initial catalogue file exists
    assert os.path.exists(catalogue_hmtk_fname)
    #
    # Create output filename
    lbl = 'all'
    if labels is not None and out_fname_ext is None:
        labels = [labels] if isinstance(labels, str) else labels
        if len(labels) < 2:
            lbl = labels[0]
        else:
            lbl = '-'.join([l for l in labels])
        assert tr_fname is not None
        assert os.path.exists(tr_fname)
        ext = '_dec_{:s}{:s}.{:s}'.format(lbl, olab, format)
    else:
        ext = '_dec_{:s}{:s}.{:s}'.format(out_fname_ext, olab, format)
    #
    # Output filename
    out_fname = Path(os.path.basename(catalogue_hmtk_fname)).stem+ext
    if output_path is not None:
        assert os.path.exists(output_path)
    else:
        output_path = os.path.dirname(catalogue_hmtk_fname)
    out_fname = os.path.abspath(os.path.join(output_path, out_fname))
    #
    # Read the catalogue and adding default values
    cat = _load_catalogue(catalogue_hmtk_fname)
    if fix_defaults:
        cat = _add_defaults(cat)
    cato = copy.deepcopy(cat)
    #
    # Select earthquakes belonging to a given TR. When necessary combining
    # multiple TRs, use label <TR_1>,<TR_2>AND...
    idx = numpy.full(cat.data['magnitude'].shape, True, dtype=bool)
    sumchk = 0
    if labels is not None and tr_fname is not None:
        f = h5py.File(tr_fname, 'r')
        idx = numpy.array([False for i in range(len(f[labels[0]]))])
        for lab in labels:
            idx_tmp = f[lab][:]
            idx[numpy.where(idx_tmp.flatten())] = True
            print(lab, sum(idx_tmp.flatten()))
            sumchk += sum(idx_tmp.flatten())
        f.close()
    idx = idx.flatten()
    #
    # Filter catalogue
    num_eqks_sub = len(cat.data['magnitude'])
    if labels is not None:
        sel = CatalogueSelector(cat, create_copy=False)
        sel.select_catalogue(idx)
        num_eqks_sub = len(cat.data['magnitude'])
        assert sumchk == num_eqks_sub
    #
    # Declustering parameters
    config = declustering_params
    #
    # Create declusterer
    modstr = 'openquake.hmtk.seismicity'
    module = importlib.import_module(modstr)
    my_class = getattr(module, declustering_meth)
    declusterer = my_class()
    #
    # Create distance-time window
    if 'time_distance_window' in config:
        my_class = getattr(module, config['time_distance_window'])
        config['time_distance_window'] = my_class()
    #
    # Declustering
    vcl, flag = declusterer.decluster(cat, config)
    #
    # Save foreshocks and aftershocks
    catt = copy.deepcopy(cat)
    catt.select_catalogue_events(numpy.where(flag != 0)[0])
    if save_af:
        ext = '_dec_af_{:s}{:s}.{:s}'.format(lbl, olab, format)
        outfa_fname = Path(os.path.basename(catalogue_hmtk_fname)).stem+ext
        outfa_fname = os.path.abspath(os.path.join(output_path, outfa_fname))
    #
    # Select mainshocks
    cat.select_catalogue_events(numpy.where(flag == 0)[0])
    tmps = 'Number of earthquakes in the original subcatalogue: {:d}'
    print('Total eqks       : {:d}'.format(num_eqks_sub))
    num_main = len(cat.data['magnitude'])
    num_foaf = len(catt.data['magnitude'])
    print('Mainshocks       : {:d}'.format(num_main))
    print('Fore/Aftershocks : {:d}'.format(num_foaf))
    assert num_main + num_foaf == num_eqks_sub
    #
    # Save output
    if format == 'csv':
        cat.write_catalogue(out_fname)
        if save_af:
            catt.write_catalogue(outfa_fname)
    elif format == 'pkl':
        fou = open(out_fname, 'wb')
        pickle.dump(cat, fou)
        fou.close()
        if save_af:
            fou = open(outfa_fname, 'wb')
            pickle.dump(catt, fou)
            fou.close()
    #
    # Create subcatalogues
    icat = numpy.nonzero(idx)[0]
    if subcatalogues:
        f = h5py.File(tr_fname, 'r')
        for lab in labels:
            #
            # Select mainshocks in a given tectonic region
            jjj = numpy.where(flag == 0)[0]
            tmpi = numpy.full((len(idx)), False, dtype=bool)
            tmpi[icat[jjj.astype(int)]] = True
            idx_tmp = f[lab][:].flatten()
            kkk = numpy.logical_and(tmpi, idx_tmp)
            if save_af:
                jjj = numpy.where(flag != 0)[0]
                tmpi = numpy.full((len(idx)), False, dtype=bool)
                tmpi[icat[jjj.astype(int)]] = True
                idx_tmp = f[lab][:].flatten()
                jjj = numpy.logical_and(tmpi, idx_tmp)
            #
            # Create output catalogue
            tsel = CatalogueSelector(cato, create_copy=True)
            ooo = tsel.select_catalogue(kkk)
            if save_af:
                aaa = tsel.select_catalogue(jjj)
            #
            # Info
            tmps = 'Cat: {:s}\n'
            tmps += '    Earthquakes: {:5d} Mainshocks {:5d} {:4.1f}%'
            pct = sum(kkk)/sum(idx_tmp)*100.
            tmpr = '    mmin: {:5.2f} mmax {:5.2f}'
            logging.info(tmps.format(lab, sum(idx_tmp), sum(kkk), pct))
            print(tmps.format(lab, sum(idx_tmp), sum(kkk), pct))
            print(tmpr.format(min(ooo.data['magnitude']),
                              max(ooo.data['magnitude'])))
            #
            # Output filename
            ext = '_dec_{:s}{:s}.{:s}'.format(lab, olab, format)
            tcat_fname = Path(os.path.basename(catalogue_hmtk_fname)).stem+ext
            tmps = os.path.join(output_path, tcat_fname)
            tcat_fname = os.path.abspath(tmps)
            if save_af:
                ext = '_dec_af_{:s}{:s}.{:s}'.format(lab, olab, format)
                tcataf_fname = Path(
                    os.path.basename(catalogue_hmtk_fname)).stem + ext
                tmps = os.path.join(output_path, tcataf_fname)
                tcataf_fname = os.path.abspath(tmps)
            #
            # Dumping data into the pickle file
            if ooo is not None:
                if format == 'csv':
                    ooo.write_catalogue(tcat_fname)
                    if save_af:
                        aaa.write_catalogue(tcataf_fname)
                elif format == 'pkl':
                    fou = open(tcat_fname, 'wb')
                    pickle.dump(ooo, fou)
                    fou.close()
                    if save_af:
                        fou = open(tcataf_fname, 'wb')
                        pickle.dump(aaa, fou)
                        fou.close()
            else:
                tstr = 'Catalogue for region {:s} is empty'.format(lab)
                logging.warning(tstr)
        f.close()

    outl = [out_fname]
    if save_af:
        outl.append(outfa_fname)

    return [out_fname]
