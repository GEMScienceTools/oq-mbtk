#!/usr/bin/env python
# coding: utf-8

import os
import re
import toml
import copy
from openquake.baselib import sap
from openquake.mbt.tools.model_building.dclustering import decluster


def main(config_fname, *, root=None):
    """
    Decluster a catalogue and create subcatalogues
    """

    if root is None:
        root = os.getcwd()

    print('\nReading:', config_fname)
    config = toml.load(config_fname)

    fname_cat = os.path.join(root, config['main']['catalogue'])
    fname_reg = os.path.join(root, config['main']['tr_file'])
    fname_out = os.path.join(root, config['main']['output'])
    create_sc = config['main']['create_subcatalogues']
    save_afrs = config['main']['save_aftershocks']
    add_deflt = config['main']['catalogue_add_defaults']

    assert os.path.exists(fname_cat)
    assert os.path.exists(fname_reg)
    assert os.path.exists(fname_out)

    methods = []
    for key in config:
        if re.search('^method', key):
            method = config[key]['name']
            params = config[key]['params']
            label = config[key]['label']
            methods.append([method, params, label])

    for key in config:
        if re.search('^case', key):
            print('\n Case {:s}'.format(key))
            regions = config[key]['regions']
            cat_lab = config[key]['label']

            for meth in methods:
                print('')
                params = copy.deepcopy(meth[1])
                _ = decluster(fname_cat, meth[0], params, fname_out, regions,
                              fname_reg, create_sc, 'csv', meth[2], save_afrs,
                              cat_lab, add_deflt)


MSG = 'Path to the configuration fname - typically a .toml file'
main.config_fname = MSG
MSG = 'Root folder used as a reference for paths [pwd]'
main.root = MSG

if __name__ == "__main__":
    sap.run(main)
