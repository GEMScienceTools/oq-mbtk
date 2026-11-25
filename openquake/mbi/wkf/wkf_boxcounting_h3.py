#!/usr/bin/env python
# coding: utf-8

import datetime
from openquake.baselib import sap
from openquake.wkf.utils import create_folder
from openquake.wkf.seismicity.box_counting import boxcounting


def main(cat: str, mapping: str, config: str, h3_level: int, folder_out: str,
         *, year_end: int = None, weighting: str = 'one'):
    """
    Counts the earthquakes in each H3 cell at the resolution `h3_level`
    """
    if year_end is None:
        year_end = datetime.date.today().year
    create_folder(folder_out)
    boxcounting(cat, int(h3_level), mapping, config, folder_out,
                int(year_end), weighting)


main.cat = 'Name of the .csv file with the catalogue'
main.mapping = ('Name of the .csv file with the mapping between h3 cells '
                'and zones')
main.config = 'Name of the .toml configuration file'
main.h3_level = 'The h3 level i.e. the grid resolution'
main.folder_out = 'Folder where to save the output files'
main.year_end = 'Latest year in the catalogue'
main.weighting = 'Weighting option for events [one, mfd, completeness]'

if __name__ == '__main__':
    sap.run(main)
