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

SKIPLIST = ['aus', 'kor', 'pac', 'phl', 'sam', 'ssa', 'usa',
            'ucerf']

# limits are defined as: lomin, lomax, lamin, lamax
SUBSETS = {
           'als': {'USA': ['-180 50 -127 50 -127 73 -180 73',
                           '170 47 180 47 180 57 170 57']},
           'cca': {'COL': ['-83 11 -80 11 -80 14 -83 14']},
           'haw': {'USA': ['-162 18 -153 18 -153 24 -162 24']},
           'idn': {'MYS': ['108 0 120 0 120 9 108 9']},
           'nea': {'RUS': ['75 0 180 0 180 88 75 88',
                            '-180 0 -160 0 -160 88 -180 88']},
           'nwa': {'RUS': ['75 0 25 0 25 10 25 20 25 40 25 50 25 88 75 88']},
           'pac': {'NZL': ['-180 -20 -160 -20 -160 -10 -180 -10'],
                   'USA': ['-180 -20 -160 -20 -160 -10 -180 -10']},
           'naf': {'DZA': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39'],
                   'TCD': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39'],
                   'LBY': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39'],
                   'MLI': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39'],
                   'MRT': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39'],
                   'NER': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39'],
                   'SDN': ['-21 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20 25 20 30 20 39 20 39 39 -21 39']},
           'sea': {'MYS': ['98 0 106 0 106 8 98 8']},
           'ssa': {'SDN': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 60 -28.5 60 20 40 20 35 20 30 20'],
                   'CAF': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 60 -28.5 60 20 40 20 35 20 30 20'],
                   'AGO': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 60 -28.5 60 20 40 20 35 20 30 20'],
                   'COG': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 60 -28.5 60 20 40 20 35 20 30 20'],
                   'COD': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 60 -28.5 60 20 40 20 35 20 30 20'],
                   'NAM': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 60 -28.5 60 20 40 20 35 20 30 20']},
           'usa': {'USA': ['-128 22 -90 22 -75 22 -60 22 -60 53 -128 53']},
           'waf': {'DZA': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'AGO': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'BWA': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'TCD': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'COD': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'CAF': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'LBY': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'MLI': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'MRT': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'NER': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'ZAF': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'SDN': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20'],
                   'NAM': ['25 20 25 5 18 -5.5 18.50 -9.80 18.95 -14.17 20.6 -30.25 9 -31 -29 20 -20 20 -15 20 -10 20 -5 20 0 20 5 20 10 20 15 20 20 20']}
          }

# XX corresponds to the portion of Russia covered by the European model
# XW corresponds to the portion of Malaysia covered by the Indonesian model
# XY corresponds to the portion of Colombia covered by the CAC model

DATA = {
       'als': ['USA'],
       'arb': ['ARE', 'BHR', 'OMN', 'QAT', 'SAU',
               'YEM'],
       'aus': ['AUS'],
       'can': ['CAN'],
       'cca': ['ABW', 'AIA', 'ATG', 'BHS', 'BLZ',
               'BRB', 'CCY', 'CRI', 'CUB', 'CYM',
               'DMA', 'DOM', 'GLP', 'GRD', 'GTM',
               'HND', 'HTI', 'JAM', 'KNA', 'LCA',
               'MSR', 'MTQ', 'NIC', 'PAN', 'PRI',
               'SLV', 'TCA', 'TTO', 'VCT', 'VGB',
               'VIR'],
       'cea': ['KAZ', 'KGZ', 'TJK', 'TKM', 'UZB'],
       'chn': ['CHN', 'Z02', 'Z03', 'Z08'],
       'gld': ['GRL'],
       'mie': ['AFG', 'ARM', 'AZE', 'CYP', 'GEO',
               'IRN', 'IRQ', 'ISR', 'JOR', 'KWT',
               'LBN', 'PAK', 'Z06', 'PSA', 'PSE',
               'SYR', 'TUR'],
       'eur': ['ALB', 'AND', 'AUT', 'BEL', 'BGR',
               'BIH', 'BLR', 'CHE', 'CZE', 'DEU',
               'DNK', 'ESP', 'EST', 'FIN', 'FRA',
               'FRO', 'GBR', 'GRC', 'HRV', 'HUN',
               'IMN', 'IRL', 'ISL', 'ITA', 'JEY',
               'LIE', 'LTU', 'LUX', 'LVA', 'MCO',
               'MDA', 'MKD', 'MLT', 'MNE', 'NLD',
               'NOR', 'POL', 'PRT', 'ROU', 'RUX',
               'SRB', 'SVK', 'SVN', 'SWE', 'UKR'],
       'haw': ['USA'],
       'ind': ['BGD', 'BTN', 'IND', 'Z01', 'Z04',
               'Z05', 'Z07', 'Z09', 'LKA', 'NPL'],
       'idn': ['BRN', 'IDN', 'MXW', 'TLS'],
       'jpn': ['JPN'],
       'kor': ['KOR', 'PRK'],
       'mex': ['MEX'],
       'nea': ['MNG', 'RUS'],
       'nwa': ['RUS'],
       'naf': ['DZA', 'EGY', 'ESH', 'GIB', 'LBY',
               'MAR', 'MLI', 'MRT', 'NER', 'PSA',
               'SDN', 'TCD', 'TUN'],
       'nzl': ['NZL'],
       'pac': ['FJI', 'KIR', 'NCL', 'NFK', 'NIU',
               'NRU', 'NZL', 'SLB', 'TON', 'TUV',
               'USA', 'VUT', 'WLF', 'WSM'],
       'phl': ['PHL'],
       'png': ['PNG'],
       'sam': ['ABW', 'ARG', 'BOL', 'BRA', 'CHL',
               'COL', 'CUW', 'ECU', 'FLK', 'GUF',
               'GUY', 'PER', 'PRY', 'SUR', 'URY',
               'VEN'],
       'sea': ['KHM', 'LAO', 'MMR', 'MYS', 'SGP',
               'THA', 'VNM'],
       'ssa': ['AGO', 'BDI', 'BWA', 'CAF', 'COD',
               'COG', 'COM', 'DJI', 'ERI', 'ETH',
               'KEN', 'MDG', 'MOZ', 'MWI', 'NAM',
               'RWA', 'SDN', 'SOM', 'TZA', 'UGA',
               'ZMB', 'ZWE'],
       'twn': ['TWN'],
       'usa': ['USA'],
       'ucf': ['CAL'],
       'waf': ['AGO', 'BEN', 'BFA', 'BWA', 'CAF',
               'CIV', 'CMR', 'COD', 'COG', 'DZA',
               'GAB', 'GHA', 'GIN', 'GMB', 'GNB',
               'GNQ', 'LBR', 'LBY', 'MLI', 'MRT',
               'NAM', 'NER', 'NGA', 'SDN', 'SEN',
               'SHN', 'SLE', 'STP', 'TCD', 'TGO'],
       'zaf': ['LSO', 'SWZ', 'ZAF'],
}
