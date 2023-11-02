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
    Decluster a catalogue for one or more tectonic regions and 
    create the declustered subcatalogues

    :param config_fname:
        name of the .toml file that includes the declustering 
        settings

        Example of .toml file
        ```
        [main]

        catalogue = './input/cam.pkl'
        tr_file = './cam_ccl/classified.hdf5'
        output = './cam_ccl/dec_subcats/'
        
        create_subcatalogues = 'true'
        save_aftershocks = 'true'
        catalogue_add_defaults = 'true'

        [method1]
        name = 'GardnerKnopoffType1'
        params = {'time_distance_window' = 'UhrhammerWindow', 'fs_time_prop' = 0.1}
        label = 'UH'

        [case1]
        regions = ['crustal', 'int_A']
        label = 'int_cru'

        [case2]
        regions = ['slab_A']
        label = 'slab'
        ```

        Configuration keys that include the string 'method' are used to 
        define the declustering parameters that should be used for all cases. 
        Keys that include 'case' are a list of one or more tectonic region
        types (as defined when running classify.py) should be declustered 
        together. 

    :param root:
        root folder to which all other paths are relative. 
        default is the current working directory


    
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
    if not os.path.exists(fname_out):
        os.makedirs(fname_out)

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
