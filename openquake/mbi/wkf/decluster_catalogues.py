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
import toml
import numpy as np
from openquake.wkf.utils import create_folder
from openquake.baselib import sap
from openquake.mbt.tools.model_building.dclustering import decluster
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue


def decluster_catalogue(fname: str, config: str,  output_folder: str, 
                           subcatalogues: bool = False, fmat: str = 'csv'):
    """
    Declusters a catalogue using parameters specified in a toml file 

    :param fname: 
        filename specifying location of catalogue to be declustered
    :param config:
        Location of .toml file specifying necessary parameters for clustering method of choice. See individual clustering functions for further details.
    :param output_folder:
        Location in which to save output - a declustered earthquake catalogue that keeps only identified mainshocks
    :param bool subcatalogues:
        When true creates subcatalogues per tectonic region
    :param str fmat:
        Format of output. Can be either 'csv' or 'pkl'        

    """
    # Load the config folder
    config = toml.load(config)
    print(config)
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
    
    declustering_meth = config['declustering']['method']
    print(declustering_meth)
    # Check correct parameters are included for given method
    
    ## Windowing
    if declustering_meth == "windowing":
        declustering_meth = 'GardnerKnopoffType1'
        declustering_params = {'time_distance_window': config['time_distance_window'], 'fs_time_prop': config['fs_time_prop']}
        
     
        out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=subcatalogues,
                    olab= config['time_distance_window'],
                    save_af=True,
                    fix_defaults=True)
    
    ## Reasenberg
    elif declustering_meth == "Reasenberg":
        declustering_params = {'taumin' : config['taumin'],  # look ahead time for not clustered events, days
            'taumax' : config['declustering']['taumax'],  # maximum look ahead time for clustered events, days
            'P' : config['declustering']['P'],  # confidence level that this is next event in sequence
            'xk' : config['declustering']['xk'],  # factor used with xmeff to define magnitude cutoff
            'xmeff' : config['declustering']['xmeff'],  # magnitude effective, used with xk to define magnitude cutoff
            'rfact' : config['declustering']['rfact'],  # factor for interaction radius for dependent events
            'horiz_error' : config['declustering']['horiz_error'],  # epicenter error, km.  if unspecified or None, it is pulled from the catalogue
            'depth_error' : config['declustering']['depth_error'],  # depth error, km.  if unspecified or None, it is pulled from the catalogue
            'interaction_formula' : config['declustering']['interaction_formula'],  # either `Reasenberg1985` or `WellsCoppersmith1994`
            'max_interaction_dist' : config['declustering']['max_interaction_dist'] }

        out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=subcatalogues,
                    olab= 'Reasenberg',
                    save_af=True,
                    fix_defaults=True)


    
    ## Zaliapin and Ben-Zion
    elif declustering_meth == "Zaliapin":
        declustering_params = {'fractal_dim': config['declustering']['fractal_dim'], 
            'b_value': config['declustering']['b_value'], 
            'depth': config['declustering']['depth']}
        
        out = decluster(fname,
                    declustering_meth,
                    declustering_params,
                    output_folder,
                    labels=labels,
                    tr_fname=tr_fname,
                    subcatalogues=subcatalogues,
                    olab= 'Zaliapin',
                    save_af=True,
                    fix_defaults=True)

    
    else:
        print("Unrecognised declustering algorithm. Please choose from `windowing`, `Reasenberg`, `Zaliapin`and supply necessary parameters in config")
    


def main(fname: str,  config: str, output_folder: str, *, subcatalogues: bool = False, fmat: str = 'csv'):
    """
    Create declustered catalogues using the algorithm and parameters supplied in the config file.

    
    """
    decluster_catalogue(fname, config, output_folder, subcatalogues, fmat)


main.fname = 'Name of the .csv formatted catalogue'
main.config = 'toml file specifiying declustering parameters'
main.output_folder = 'Path to the output folder'
msg = 'Boolean, when true it creates subcatalogues'
main.subcatalogues = msg

if __name__ == '__main__':
    sap.run(main)
