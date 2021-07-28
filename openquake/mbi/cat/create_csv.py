#!/usr/bin/env python3

import os
import shutil
import pandas as pd
from pathlib import Path
from openquake.baselib import sap


def create_folder(folder: str, clean: bool = False):
    """
    Create a folder. If the folder exists, it's possible to
    clean it.

    :param folder:
        The name of the folder tp be created
    :param clean:
        When true the function removes the content of the folder
    """
    if os.path.exists(folder):
        if clean:
            shutil.rmtree(folder)
    else:
        Path(folder).mkdir(parents=True, exist_ok=True)


def main(cat_fname, fname_out):

    # Read catalogue
    df = pd.read_hdf(cat_fname)

    # Create folder
    create_folder(os.path.dirname(fname_out))

    # Save file
    df.to_csv(fname_out)


main.cat_fname = 'Name of the .hdf5 file containing the homogenized catalogue'
main.cat_fname = 'Name of output .csv that will be created'

if __name__ == "__main__":
    """
    The function creates the .csv file with the events in the homogenised
    catalog
    """
    sap.run(main)
