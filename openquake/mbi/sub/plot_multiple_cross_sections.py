#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.sub.plotting.plot_multiple_cross_sections import pcs


def main(cs_file, output_folder=None):
    """
    Creates file with parameters needed to plot cross sections.
    Output file is a list for each cross section with the format:

        lon lat length depth azimuth id <config>.ini
    """

    pcs(cs_file, output_folder)


main.cs_file = 'file with cross sections details'
main.output_folder = 'place to store the pdfs'

if __name__ == "__main__":
    sap.run(main)
