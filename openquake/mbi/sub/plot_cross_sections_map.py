#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.sub.plotting.plot_multiple_cross_sections_map import plot


def main(config_fname, cs_file=None):
    """
    Plots map of cross sections with earthquake data

    Example:
        oqm sub plot_cross_sections_map config.ini cs_profiles.cs

    Note: paths in config.ini are relative to cwd
    """

    plot(config_fname, cs_file)


main.config_fname = 'config file to datasets'
main.cs_file = 'existing cross sections details'

if __name__ == "__main__":
    sap.run(main)
