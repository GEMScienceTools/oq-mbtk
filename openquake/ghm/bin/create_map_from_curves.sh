#!/usr/bin/env bash
#
# Shell script which creates a hazard maps from a csv file containing hazard
# curves.
#
# folder with .json files
I_JSON='/tmp/ghm/'
#
# Output folder
PREFIX='map'
#
# Output file and folder
O_NAME='map.csv'
O_PATH='/tmp/ghm/global'
#
# String with the intensity measure type
IMTSTR='PGA'
#
# Probability of exceedance
PEX='-p 0.002105'
#
# Run hazard curves homogenisation
../create_map_from_curves.py $I_JSON $PREFIX $O_NAME $O_PATH $IMTSTR $PEX
