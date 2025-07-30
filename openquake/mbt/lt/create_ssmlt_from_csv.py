#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from ltreader import SmmLt
from openquake.baselib import sap

"""
csv file example: 

sources,set,unc_type,weight
src_1.xml src_2.xml,base,sourceModel,1.000
src_0.xml,sz0,extendModel,0.417
src_0.xml,sz0,extendModel,0.400
src_0.xml,sz0,extendModel,0.183


Other uncertainty types than sourceModel and extendModel are not yet implemented
"""

def main(in_fname: str, out_fname: str, *, extended: bool = False):
    """
    Create a file gmm logic tree file .xml from a csv formatted file
    """
    path = os.path.dirname(out_fname)
    Path(path).mkdir(parents=True, exist_ok=True)
    lt = SmmLt.from_csv(in_fname, lttype='ssmlt')
    if extended in ['true', 'True']:
        extended = True
    lt.write(out_fname, extended=extended)


main.in_path = "Path to the input .csv file with the GMM model"
main.out_folder = "Path to the output .xml file"
main.extended_format = "When True uses the extended format [False]"

if __name__ == "__main__":
    sap.run(main)
