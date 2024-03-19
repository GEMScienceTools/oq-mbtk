#!/usr/bin/env python

import os
import re
import sys
import numpy as np
import pandas as pd
import pickle
import glob
import time
import toml

from openquake.baselib import sap
from scipy.stats import truncnorm
from copy import deepcopy

from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
from openquake.mbi.ccl.decluster_multiple_TR import main as decluster_multiple_TR
from openquake.cat.completeness.generate import get_completenesses
from openquake.cat.completeness.analysis import (_completeness_analysis, 
                                                 read_compl_params,
                                                 read_compl_data)
from openquake.cat.completeness.mfd_eval_plots import make_all_plots


def _get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def _write_cat_instance(catalogue, format, fileout, cnum):
    if format == 'hmtk':
        catalogue.write_catalogue(fileout.format(cnum))
    elif format == 'pickle':
        with open(fileout.format(cnum), 'wb') as f:
            pickle.dump(catalogue, f)

def _create_catalogue_versions(catfi, outdir, numcats=None, stype='random'):
    """
    catfi: catalogue filename (pkl or hmtk/csv)
    stype: 
        random - samples from truncnorm
        even - same sample for every magnitude; -1 : 1 by 0.2
    """
    # check if output folder is empty

    if os.path.exists(outdir):
        tmps = f'\nError: {outdir} already exists! '
        tmps += '\n Choose an empty directory.'
        print(tmps)
        sys.exit(1)
    else:
        os.makedirs(outdir)
        
    csvout = os.path.join(outdir, 'v{}_'+catfi.split('/')[-1])
    fileout = os.path.join(outdir, 'v_mags.csv')
    factors = np.arange(-1,1,0.1)
    
    # read catalogue
    if 'pkl' in catfi:
        format = 'pickle'
        catalogue = pd.read_pickle(catfi) 

    elif ('csv' in catfi) or ('hmtk' in catfi):
        parser = CsvCatalogueParser(catfi) # From .csv to hmtk
        catalogue = parser.read_file()
        format = 'hmtk'

    else:
        print(sys.stderr, "Use a supported catalogue format.")
        sys.exit(1)

    data = catalogue.data

    if numcats and stype == 'even': 
        print(sys.stderr, \
              "Cannot specify number of catalogues for even sampling.")
        sys.exit(1)

    if stype == 'random': 

        mags = []
        
        time_strt = time.time()
        for ii in np.arange(0,len(data['magnitude'])): 
            m=data['magnitude'][ii]
            sd=data['sigmaMagnitude'][ii]
            allm = _get_truncated_normal(mean=m, sd=sd, low=m-sd, upp=m+sd)
            mags_perm = allm.rvs(size=numcats)
            mags.append(mags_perm)

        time_end = time.time()
        print('Run time for generating magnitudes: ')
        print(time_end - time_strt)

        marr = np.array(mags)
        np.savetxt(fileout, marr, delimiter=',', fmt='%.2f')

        for ii, ms in enumerate(marr.T): 
            time_strt = time.time()
            catalogue = deepcopy(catalogue)
            catalogue.data['magnitude'] = np.array(ms)
            _write_cat_instance(catalogue, format, csvout, ii)
            time_end = time.time()
            print('Run time for writing catalogue: ')
            print(time_end - time_strt)


    elif stype == 'even':
        full_mags = {}
        for jj, f in enumerate(factors):
            mags = [f * ms + m for m, ms in zip(data['magnitude'], data['sigmaMagnitude'])]
            catalogue = deepcopy(catalogue)
            catalogue.data['magnitude'] = np.array(mags)
            _write_cat_instance(catalogue, format, fileout, jj)
        full_mags[jj] = mags
        pd.DataFrame(full_mags).to_csv(fileout, index=None)

    else:
        print(sys.stderr, "Use a supported sampling type.")
        sys.exit(1)



def _decl_all_cats(outdir, cat, dcl_toml_tmp, decdir):

    """
    """
    catvs = glob.glob(os.path.join(outdir, '*pkl'))

    config = toml.load(dcl_toml_tmp)
    config['main']['save_aftershocks'] = False
    for cat in catvs: 
        config['main']['catalogue'] = cat
        config['main']['output'] = decdir
        tmpfi = 'tmp-config-dcl.toml'
        with open(tmpfi, 'w') as f:
            f.write(toml.dumps(config))

        decluster_multiple_TR(tmpfi)

    labels = []
    for key in config:
        if re.search('^case', key):
            labels.extend(config[key]['regions'])



def _gen_comple(compl_toml, dec_outdir, compdir, tmpfi):
    """
    """
    config = toml.load(compl_toml)

    cref = config['completeness'].get('completeness_ref', None)
    
    mmin_compl = config['completeness']['min_mag_compl']

    if cref == None:
        mags = np.arange(4, 8.0, 0.1)
        years = np.arange(1900, 2020, 5.0)
    else:
        mrange = config['completeness'].get('deviation', 1.0)
        mags_cref = [c[1] for c in cref]
        mags_min = min(mags_cref) - mrange
        mags_max = max(mags_cref) + mrange
        mags_all = np.arange(mags_min, mags_max, 0.2)
        #mags_lo = np.arange(mags_min, 5.8, 0.1)
        #mags_hi = np.arange(5.8, mags_max, 0.2)
        #mags_all = mags_lo.tolist() + mags_hi.tolist()
        mags = list(set([round(m,2) for m in mags_all if m >= mmin_compl ]))
        mags.sort()
        print(mags)

        years = [c[0] for c in cref]
        years.sort()

    config['completeness']['mags'] = mags
    config['completeness']['years'] = years

    with open(tmpfi, 'w') as f:
        f.write(toml.dumps(config))

    get_completenesses(tmpfi, compdir)


def _compl_analysis(decdir, compdir, compl_toml, labels, fout, fout_figs):
    """
    """
    ms, yrs, bw, r_m, r_up_m, bmin, bmax, crit = read_compl_params(compl_toml)
    compl_tables, mags_chk, years_chk = read_compl_data(compdir)

    # Fixing sorting of years
    if np.all(np.diff(yrs)) >= 0:
        yrs = np.flipud(yrs)

    for lab in labels:
        dec_catvs = glob.glob(os.path.join(decdir, f'*{lab}*'))
        fout_lab = os.path.join(fout, lab)
        fout_figs_lab = os.path.join(fout_figs, lab, 'mfds')
        for ii, cat in enumerate(dec_catvs):
            try:
                res = _completeness_analysis(cat, yrs, ms, bw, r_m,
                                      r_up_m, [bmin, bmax], crit,
                                      compl_tables, f'{ii}',
                                      folder_out_figs=fout_figs_lab,
                                      folder_out=fout_lab,
                                      rewrite=False)
            except:
                print(f'Impossible for catalogue {ii}')


def main(configfile):
    """
    """

    config = toml.load(configfile)

    # read basic inputs
    catfi = config['main']['catalogue_filename']
    outdir = config['main']['output_directory']
    labs = config['mfds'].get('labels', None)


    # make subdirs based on outdir name
    catdir = os.path.join(outdir, 'catalogues')
    decdir = os.path.join(outdir, 'declustered')
    compdir = os.path.join(outdir, 'completeness')
    resdir = os.path.join(outdir, 'results')
    figdir = os.path.join(outdir, 'figures')
    tmpconf = os.path.join(outdir, 'tmp-config-compl.toml')


    # if create_cats, make the sampled catalogues
    create_cats = config['catalogues'].get('create_catalogues', True)
    if create_cats:
        ncats = int(config['catalogues'].get('number', 1000))
        stype = config['catalogues'].get('sampling_type', 'random')
        _create_catalogue_versions(catfi, catdir, numcats=ncats, stype=stype)

    # perform all the declustering possibilities - links TRTs following configs
    decluster = config['decluster'].get('decluster_catalogues', True)
    if decluster:
        dcl_toml_tmpl = config['decluster']['decluster_settings']
        _decl_all_cats(catdir, catfi, dcl_toml_tmpl, decdir)


    # generate the completeness tables 
    generate = config['completeness'].get('generate_completeness', True)
    if generate:
        compl_toml = config['completeness']['completeness_settings']
        _gen_comple(compl_toml, decdir, compdir, tmpconf)

    # create the mfds for the given TRT labels 
    mfds = config['mfds'].get('create_mfds', True)
    if mfds:
        if not labs:
            print('Must specify the TRTs')
            sys.exit()

        _compl_analysis(decdir, compdir, tmpconf, labs, resdir, figdir)

    plots = config['plot'].get('make_plots', True)
    if plots:
        if not labs:
            print('Must specify the TRTs')
            sys.exit()
        make_all_plots(resdir, compdir, figdir, labs)


main.configfile = 'path to configuration file'

if __name__ == '__main__':
    freeze_support()
    sap.run(main)
