#!/usr/bin/env bash
#
# This is the shell script used to run the code which produces an homogenised
# set of hazard curves with global coverage. Note that it requires
# the a-priori definition of two environment variables: REPOS and GEMDATA.
# The former contains the `mosaic` repository, the latter is the path to the
# repository containing the GEM dataset.
#
# Create the homogenised set of hazard curves
C_SHP='./../data/gis/contacts_between_models.shp'
#
# Output folder
O_PATH='/tmp/ghm'
#
# Path to the folder with the mosaic repository
# For the original hazard map we used this folder:
# /Users/mpagani/Documents/2018/diary/11/13_ucf/maps
D_PATH=$REPOS'/mosaic'
#
# Spatial index folder
SIDX=$GEMDATA'/global_grid/trigrd_split_9_spacing_13'
#
# Boundaries shapefile
B_SHP='./../data/gis/world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
#
# String with the intensity measure type
IMTSTR='PGA'
#
# Shapefile with inland territories
I_SHP='./../data/gis/inland.shp'
#
# List of models to be processed. This is an optional parameter. If not set,
# i.e. MDLS="", all the models specified in `openquake.ghm.mosaic.DATA`
# will be used.
#MDLS="-m als,can,usa,mex,ucf,cca,sam"
#MDLS="-m mex,usa"
#MDLS="-m cca,sam"
#
# Run hazard curves homogenisation
echo ../create_homogenised_curves.py $C_SHP $O_PATH $D_PATH $SIDX $B_SHP $IMTSTR $I_SHP $MDLS
../create_homogenised_curves.py $C_SHP $O_PATH $D_PATH $SIDX $B_SHP $IMTSTR $I_SHP $MDLS

