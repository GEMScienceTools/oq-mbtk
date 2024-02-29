#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.sub.plotting.plot_multiple_cross_sections import pcs


def main(cs_file, output_folder=None):
    """
    Plots cross section of data for each cross section in
    cs_file including the data in the ini file referenced
    in the cs_file lines. Saves pdfs to output_folder

    Example:

        oqm sub plot_multiple_cross_sections cs_profiles.cs pdf 
    """

    pcs(cs_file, output_folder)


main.cs_file = 'file with cross sections details'
main.output_folder = 'place to store the pdfs'

if __name__ == "__main__":
    sap.run(main)
