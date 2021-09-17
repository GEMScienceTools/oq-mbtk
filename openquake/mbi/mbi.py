

import toml
from subprocess import call
from openquake.baselib import sap


def main(fname_config: str):

    # Read the settings
    model = toml.load(fname_config)

    # Processing
    for key in model['unc']:
        print('>> key', key)


main.in_path = 'Path to the input folder'

if __name__ == "__main__":
    sap.run(main)
