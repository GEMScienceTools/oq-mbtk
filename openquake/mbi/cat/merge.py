#!/usr/bin/env python3

from openquake.baselib import sap
from openquake.cat.hmg.merge import process_catalogues


def main(settings: str):
    """
    Reads the information in the settings file and creates two .h5 files
    containing the origins and magnitudes included in the catalogues
    specified in the settings
    """
    process_catalogues(settings)


main.settings = '.toml file with the settings'

if __name__ == "__main__":
    sap.run(main)
