#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.compute_mmax_from_catalogues import compute_mmax


def main(fname_input_pattern: str, fname_config: str, label: str):
    compute_mmax(fname_input_pattern, fname_config, label)


descr = 'Pattern to select input files or list of files'
main.fname_input_pattern = descr
descr = 'Name of the .toml file with configuration parameters'
main.fname_config = descr
descr = 'Label identifying the catalogue'
main.label = descr

if __name__ == '__main__':
    sap.run(main)
