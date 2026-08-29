#!/usr/bin/env python
# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8


import re
import toml
import warnings
import numpy as np
import pandas as pd


def apply_mag_conversion_rule_keep_all(low_mags, conv_eqs, conv_sigs, rows, save):
    """
    This function applies sequentially a set of rules to the information in
    the `save` :class:`pandas.DataFrame` instance.

    :param low_mags:
        A list
    :param conv_eqs:
        A list
    :param rows:
        The :class:`pandas.DataFrame` instance containing the information
        still to be processed
    :param save:
        One :class:`pandas.DataFrame` instance
        :return:
        One :class:`pandas.DataFrame` instances  with the
        homogenised magnitudes.
    """

    # Formatting
    fmt2 = "m[m >= {:.2f}]"

    # Temporary assigning magnitude
    m = np.round(rows['value'].values, 3)
    tmp = np.zeros_like(m)

    for mlow, conversion in zip(low_mags, conv_eqs):
        m_inds = m >= mlow
        if conversion == 'm':
            tmp[m_inds] = m[m_inds]
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

    return save


def process_magnitude_keep_all(work, mag_rules, msig=0.2):
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
        print('   Agency: {:s} ('.format(agency), end="")

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

            # TODO
            # This is an initial solution that is not ideal since it does
            # not take the best information available.
            # Remove duplicates. This can happen when we process a magnitude
            # type without specifying the agency

            flag = rows["eventID"].duplicated(keep='first')

            if any(flag):
                # this line is so the larger M is taken - expiremental based
                # on MEX issue
                rows = rows.sort_values(
                    "value", ascending=False).drop_duplicates('eventID').sort_index()

            # Magnitude conversion
            if len(rows) > 0:
                low_mags = mag_rules[agency][mag_type]['low_mags']
                conv_eqs = mag_rules[agency][mag_type]['conv_eqs']
                conv_sigma = mag_rules[agency][mag_type]['sigma']
                save = apply_mag_conversion_rule_keep_all(
                    low_mags, conv_eqs, conv_sigma, rows, save)
        print(")")

    return save


def apply_mag_conversion_rule(low_mags, conv_eqs, conv_sigs, rows, save, work, m_sigma):
    """
    This function applies sequentially a set of rules to the information in
    the `save` :class:`pandas.DataFrame` instance.

    :param low_mags:
        A list
    :param conv_eqs:
        A list
    :param rows:
        The :class:`pandas.DataFrame` instance containing the information
        still to be processed
    :param save:
        One :class:`pandas.DataFrame` instance
    :param work:
        One :class:`pandas.DataFrame` instance
    :return:
        Two :class:`pandas.DataFrame` instances. The first one with the
        homogenised catalogue, the second one with the information not yet
        processed (if any).
    """

    # Formatting
    fmt2 = "m[m >= {:.2f}]"

    # Temporary assigning magnitude
    m = np.round(rows['value'].values, 3)
    sig = rows['sigma'].values
    sig[sig==0.0] = m_sigma
    sig[np.isnan(sig)] = m_sigma
    tmp = np.zeros_like(m)
    tmpsig = np.zeros_like(m)
    tmpsiga = np.zeros_like(m)
    tmpsigb = np.zeros_like(m)

    try:
        assert len(low_mags) == len(conv_eqs) == len(conv_sigs)
    except ValueError:
        fmt = 'Must include a low mangitude and sigma for each'
        fmt += ' conversion equation.'
        print(fmt)

    for mlow, conversion, sigma in zip(low_mags, conv_eqs, conv_sigs):
        m_inds = m >= mlow
        if conversion == 'm':
            tmp[m_inds] = m[m_inds]
            tmpsig[m_inds] = sig[m_inds]
        else:
            tmpstr = re.sub('m', fmt2.format(mlow), conversion)
            tmpstrP = re.sub('m', '(' + fmt2.format(mlow)+'+ 0.001)', conversion)
            tmpstrM = re.sub('m', '(' + fmt2.format(mlow)+ '- 0.001)', conversion)
            cmd = "tmp[m >= {:.2f}] = {:s}".format(mlow, tmpstr)
            cmdsp = "tmpsiga[m >= {:.2f}] = {:s}".format(mlow, tmpstrP)
            cmdsm = "tmpsigb[m >= {:.2f}] = {:s}".format(mlow, tmpstrM)

            try:
                exec(cmd)
                exec(cmdsp)
                exec(cmdsm)
                deriv = [(ta-tb)/0.002 for ta, tb in zip(tmpsiga, tmpsigb)]
                sig_new = np.array([np.sqrt(s**2 + d**2 * sigma**2) for s, d in zip(sig, deriv)])
                tmpsig[m_inds] = sig_new[m_inds]

            except ValueError:
                fmt = 'Cannot execute the following conversion rule:\n{:s}'
                print(fmt.format(conversion))

    rows = rows.copy()
    rows.loc[:, 'magMw'] = tmp
    rows.loc[:, 'sig_tot'] = tmpsig
    rows = rows.drop(rows[rows['magMw']==0.0].index)
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


def process_magnitude(work, mag_rules, msig=0.2):
    """
    This function applies the magnitude conversion rules

    :param work:
        A :class:`pandas.DataFrame` instance obtained by joining the origin
        and magnitude dataframes
    :param mag_rules:
        A dictionary with the rules to be used for processing the catalogue
    :param msig:
        Standard deviation of magnitude.
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
        print('   Agency: {:s} ('.format(agency), end="")

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

            # TODO
            # This is an initial solution that is not ideal since it does
            # not take the best information available.
            # Remove duplicates. This can happen when we process a magnitude
            # type without specifying the agency

            flag = rows["eventID"].duplicated(keep='first')

            if any(flag):
                # this line is so the larger M is taken - expiremental based on MEX issue
                rows = rows.sort_values("value", ascending=False).drop_duplicates('eventID').sort_index()
                #tmp = sorted_rows[~flag].copy()
                #rows = tmp

            # Magnitude conversion
            if len(rows) > 0:

                low_mags = mag_rules[agency][mag_type]['low_mags']
                conv_eqs = mag_rules[agency][mag_type]['conv_eqs']
                conv_sigma = mag_rules[agency][mag_type]['sigma']

                if 'mag_sigma' in mag_rules[agency][mag_type]:
                    m_sigma = mag_rules[agency][mag_type]['mag_sigma']
                else:
                    m_sigma = msig

                save, work = apply_mag_conversion_rule(
                    low_mags, conv_eqs, conv_sigma, rows, save, work, m_sigma)
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

    # Reading settings. These include the priority agencies for the origin and
    # the rules for magnitude conversion.
    rules = toml.load(settings_fname)

    mag_n_sigma = 0.0
    if 'default' in rules.keys():
        mag_n_sigma = rules['default'].get('mag_sigma', mag_n_sigma)
    else:
        rules['default'] = {'mag_sigma': mag_n_sigma}

    for agency in rules['magnitude'].keys():
        for mag_type in rules['magnitude'][agency].keys():
            if 'sigma' not in rules['magnitude'][agency][mag_type].keys():
                n_mags = len(rules['magnitude'][agency][mag_type]['low_mags'])
                tmp = [mag_n_sigma for i in range(n_mags)]
                rules['magnitude'][agency][mag_type]['sigma'] = tmp

    # Checking input
    if not ('origin' in rules.keys() or 'magnitude' in rules.keys()):
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
        save, work = process_magnitude(
            work, rules['magnitude'], msig=mag_n_sigma)

        work_all_m = pd.merge(odf, mdf, on=["eventID"])
        save_all_m = process_magnitude_keep_all(
            work_all_m, rules['magnitude'], msig=mag_n_sigma)

    print("Number of origins with final mag type {:d}\n".format(len(save)))

    computed = len(save)
    expected = len(save['eventID'].unique())
    if computed - expected > 0:
        fmt = "The catalogue contains {:d} duplicated eventIDs"
        msg = fmt.format(computed - expected)
        warnings.warn(msg)

    return save, work, save_all_m
