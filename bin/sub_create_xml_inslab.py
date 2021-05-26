#!/usr/bin/env python
# coding: utf-8

import os
import toml
import configparser
from openquake.baselib import sap
from openquake.wkf.utils import create_folder
from openquake.sub.create_inslab_nrml import create


def main(config_fname: str, output_folder: str):

    investigation_t = 1.0

    # Parsing config
    model = toml.load(config_fname)
    path = os.path.dirname(config_fname)

    # Creating xml
    for key in model['sources']:
        ini_fname = os.path.join(path, model['sources'][key]['ini_fname'])
        config = configparser.ConfigParser()
        config.read_file(open(ini_fname))
        tmp = config.get('main', 'out_hdf5_fname')
        rupture_hdf5_fname = os.path.join(path, tmp)
        outf = os.path.join(output_folder, key)
        create_folder(outf)
        create(key, rupture_hdf5_fname, outf, investigation_t)


descr = 'A string with a comma separated list of source ids'
main.labels = descr
descr = 'Name of the folder where to store the profiles'
main.output_folder = descr

if __name__ == '__main__':
    sap.run(main)
