#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.sub.plotting.plot_multiple_cross_sections_map import plot


def main(config_fname, cs_file=None):
    """
    Creates file with parameters needed to plot cross sections.
    Output file is a list for each cross section with the format:

        lon lat length depth azimuth id <config>.ini
    """

    plot(config_fname, cs_file)


main.config_fname = 'config file to datasets'
main.cs_file = 'existing cross sections details'

if __name__ == "__main__":
    sap.run(main)
