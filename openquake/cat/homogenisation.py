#!/usr/bin/env python3
# coding: utf-8

import re
import toml
import numpy as np
import pandas as pd


def apply_mag_conversion_rule(low_mags, conv_eqs, rows, save, work):
    """
    :param low_mags:
        A list
    :param conv_eqs:
        A list
    :param rows:
    :param save:
    :param work:
    :return:
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
            exec(cmd)
    rows = rows.copy()
    rows.loc[:, 'magMw'] = tmp
    save = save.copy()
    save = pd.concat([save, rows], ignore_index=True, sort=False)
    print(len(save), len(work))

    # Cleaning
    eids = rows['eventID'].values
    cond = work['eventID'].isin(eids)
    work.drop(work.loc[cond, :].index, inplace=True)

    return save, work


def get_mag_selection_condition(agency, mag_type, default=["Mw", "MW"]):
    """
    :param agency:
        A string with the name of the agency that originally defined
        magnitude values
    :param mag_type:

    :return:
        A string. When evaluated, it creates a selection condition for the
        magnitude dataframe
    """

    # Create the initial selection condition using the agency name
    fmt1 = "({:s}) & (work['magType'] == '{:s}')"
    cond = "work['magAgency'] == '{:s}'".format(agency)

    # If the original magnitude type is not the default one adjust the
    # magnitude selection condition
    if mag_type not in default:
        cond = fmt1.format(cond, mag_type)

    return cond


def process_origin(odf, ori_rules):
    """
    :param work:
        A :class:`pandas.DataFrame` instance containing origin data
    :param mag_rules:
        A dictionary with the rules to be used for processing the origins
    :return:
        An updated version of the origin dataframe.
    """
    # TODO
    return odf


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
    # This is a new dataframe used to store the processed events
    save = pd.DataFrame(columns=work.columns)

    # Looping over agencies
    for agency in mag_rules.keys():
        print('\nAgency: ', agency)
        #
        # Looping over magnitude-types
        for mag_type in mag_rules[agency].keys():

            # Create the first selection condition and select rows
            cond = get_mag_selection_condition(agency, mag_type)
            rows = work.loc[eval(cond), :]

            # Magnitude conversion
            if len(rows):
                low_mags = mag_rules[agency][mag_type]['low_mags']
                conv_eqs = mag_rules[agency][mag_type]['conv_eqs']
                save, work = apply_mag_conversion_rule(low_mags, conv_eqs,
                                                       rows, save, work)

    return save, work


def process_dfs(odf_fname, mdf_fname, o_settings_fname=None,
                m_settings_fname=None):
    """
    :param odf_fname:
        Name of the .h5 file containing the origin dataframe
    :param mdf_fname:
        Name of the .h5 file containing the magnitudes dataframe
    :param o_settings_fname:
        Name of the file with the origin settings for homogenisation
    :param m_settings_fname:
        Name of the file with the magnitude settings for homogenisation
    """

    # Checking input
    if not(o_settings_fname or m_settings_fname):
        raise ValueError('At least one set of settings must be defined')

    # These are the tables with origins and magnitudes
    odf = pd.read_hdf(odf_fname)
    mdf = pd.read_hdf(mdf_fname)

    # Loading settings from .toml file and processing origins
    if o_settings_fname:
        ori_rules = toml.load(o_settings_fname)
        nodf = process_origin(odf, ori_rules)
    else:
        nodf = odf

    # Loading settings from .toml file and processing magnitudes
    if m_settings_fname:
        mag_rules = toml.load(m_settings_fname)

        # Creating a single dataframe by joining
        work = pd.merge(nodf, mdf, on=["eventID"])

        # Processing magnitudes
        save, work = process_magnitude(work, mag_rules)

    # FIXME as it stands save and work are not defined if m_settings_fname is
    # none

    return save, work
