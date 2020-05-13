#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) 2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import re
import toml
import numpy as np
import pandas as pd


def apply_mag_conversion_rule(low_mags, conv_eqs, rows, save, work):
    """
    This function applies sequentially a set of rules to the information in
    the `save` :class:`pandas.DataFrame` instance.

    :param low_mags:
        A list
    :param conv_eqs:
        A list
    :param rows:
    :param save:
    :param work:
        One :class:`pandas.DataFrame` instance
    :return:
        Two :class:`pandas.DataFrame` instances. The first one with the
        homogenised catalogue, the second one with the information not
        processed (if any).
    """

    # Formatting
    fmt2 = "m[m >= {:.2f}]"

    # Temporary assigning magnitude
    m = rows['value'].values
    tmp = np.zeros_like(m)

    for mlow, conversion in zip(low_mags, conv_eqs):
        if conversion == 'm':
            tmp = m
        else:
            tmpstr = re.sub('m', fmt2.format(mlow), conversion)
            cmd = "tmp[m >= {:.2f}] = {:s}".format(mlow, tmpstr)

            try:
                exec(cmd)
            except ValueError:
                fmt = 'Cannot execute the following conversion rule:\n{:s}'
                print(fmt.format(conversion))

    rows = rows.copy()
    rows.loc[:, 'magMw'] = tmp
    save = save.copy()
    save = pd.concat([save, rows], ignore_index=True, sort=False)

    # Cleaning
    eids = rows['eventID'].values
    cond = work['eventID'].isin(eids)
    work.drop(work.loc[cond, :].index, inplace=True)

    return save, work


def get_mag_selection_condition(agency, mag_type, df_name="work"):
    """
    Given an agency code and a magnitude type this function creates a
    condition that can be used to filter a :class:`pandas.DataFrame`
    instance.

    :param agency:
        A string with the name of the agency that originally defined
        magnitude values
    :param mag_type:
        A string defining the typology of magnitude to be selected
    :param df_name:
        A string with the name of the dataframe to which apply the query
    :return:
        A string. When evaluated, it creates a selection condition for the
        magnitude dataframe
    """

    # Create the initial selection condition using the agency name
    if re.search("^\\*", agency):
        cond = "({:s}['magType'] == '{:s}')".format(df_name, mag_type)
    else:
        cond = "{:s}['magAgency'] == '{:s}'".format(df_name, agency)
        # Adding magnitude type selection condition
        fmt1 = "({:s}) & ({:s}['magType'] == '{:s}')"
        cond = fmt1.format(cond, df_name, mag_type)
    return cond


def get_ori_selection_condition(agency):
    """
    Given an agency code this function creates a
    condition that can be used to filter a :class:`pandas.DataFrame`
    instance.
    """
    return "odf['Agency'] == '{:s}'".format(agency)


def process_origin(odf, ori_rules):
    """
    :param odf:
        A :class:`pandas.DataFrame` instance containing origin data
    :param mag_rules:
        A dictionary with the rules to be used for processing the origins
    :return:
        An updated version of the origin dataframe.
    """
    # This is a new dataframe used to store the processed origins
    save = pd.DataFrame(columns=odf.columns)

    for agency in ori_rules["ranking"]:
        print('   Agency: ', agency)

        # Create the first selection condition and select rows
        if agency in ["PRIME", "prime"]:
            rows = odf[odf["prime"] == 1]
        else:
            cond = get_ori_selection_condition(agency)

            try:
                rows = odf.loc[eval(cond), :]
            except ValueError:
                fmt = 'Cannot execute the following selection rule:\n{:s}'
                print(fmt.format(cond))

        # Saving results
        save = pd.concat([save, rows], ignore_index=True, sort=False)

        # Cleaning
        eids = rows['eventID'].values
        cond = odf['eventID'].isin(eids)
        odf.drop(odf.loc[cond, :].index, inplace=True)

    return save


def process_magnitude(work, mag_rules):
    """
    :param work:
        A :class:`pandas.DataFrame` instance obtained by joining the origin
        and magnitude dataframes
    :param mag_rules:
        A dictionary with the rules to be used for processing the catalogue
    :return:
        Two :class:`pandas.DataFrame` instances. The first one with the
        homogenised catalogue, the second one with the information not
        processed (if any).
    """

    # Add a column for destination
    if "magMw" not in list(work.columns):
        work["magMw"] = np.nan

    # This is a new dataframe used to store the processed events
    save = pd.DataFrame(columns=work.columns)

    # Looping over agencies
    for agency in mag_rules.keys():
        print('   Agency: {:s} ('.format(agency), end='')
        #
        # Looping over magnitude-types
        for mag_type in mag_rules[agency].keys():
            print('{:s} '.format(mag_type), end='')

            # Create the first selection condition and select rows
            cond = get_mag_selection_condition(agency, mag_type)
            try:
                rows = work.loc[eval(cond), :]
            except ValueError:
                fmt = 'Cannot evaluate the following condition:\n {:s}'
                print(fmt.format(cond))

            # Magnitude conversion
            if len(rows) > 0:
                low_mags = mag_rules[agency][mag_type]['low_mags']
                conv_eqs = mag_rules[agency][mag_type]['conv_eqs']
                save, work = apply_mag_conversion_rule(low_mags, conv_eqs,
                                                       rows, save, work)
        print(")")

    return save, work


def process_dfs(odf_fname, mdf_fname, settings_fname=None):
    """
    :param odf_fname:
        Name of the .h5 file containing the origin dataframe
    :param mdf_fname:
        Name of the .h5 file containing the magnitudes dataframe
    :param settings_fname:
        Name of the file with the settings for selection and homogenisation
    """

    # Initialising output
    save = None
    work = None

    # Reading settings
    rules = toml.load(settings_fname)

    # Checking input
    if not('origin' in rules.keys() or 'magnitude' in rules.keys()):
        raise ValueError('At least one set of settings must be defined')

    # These are the tables with origins and magnitudes
    odf = pd.read_hdf(odf_fname)
    mdf = pd.read_hdf(mdf_fname)

    print("Number of EventIDs {:d}\n".format(len(odf["eventID"].unique())))

    # Processing origins
    if 'origin' in rules.keys():
        print('Selecting origins')
        odf = process_origin(odf, rules['origin'])

    print("Number of origins selected {:d}\n".format(len(odf)))

    # Processing magnitudes
    if 'magnitude' in rules.keys():
        print('Homogenising magnitudes')
        # Creating a single dataframe by joining
        work = pd.merge(odf, mdf, on=["eventID"])
        save, work = process_magnitude(work, rules['magnitude'])

    print("Number of origins with final mag type {:d}\n".format(len(save)))

    return save, work
