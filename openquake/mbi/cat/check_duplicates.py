#!/usr/bin/env python3

from openquake.baselib import sap
from openquake.cat.hmg.check import check_catalogue


def main(settings: str, cat_fname: str):
    """
    This script searches for duplicates in the homogenised catalogue. It
    generates a file called `check.geojson`
    """
    nchecks = check_catalogue(cat_fname, settings)
    print(nchecks)


main.settings = '.toml file with the settings'
main.cat_fname = '.h5 file with origins'

if __name__ == "__main__":
    sap.run(main)
