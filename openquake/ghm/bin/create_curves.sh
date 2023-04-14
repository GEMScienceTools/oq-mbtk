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
SIDX=$GEMDATA'/trigrd_split_9_spacing_13'
#
# Boundaries shapefilei
B_SHP='~/gem-hazard-data/gis/grid/gadm_410_level_0.gpkg'
#B_SHP='./../data/gis/world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
#
# String with the intensity measure type
IMTSTR='SA(1.0)'
#
# Shapefile with inland territories
I_SHP='./../data/gis/inland.shp'
#
BUF=50.0
#flag set to 1 for vs30 calc
VS30_FLAG=1
#
# List of models to be processed. This is an optional parameter. If not set,
# i.e. MDLS="", all the models specified in `openquake.ghm.mosaic.DATA`
# will be used.
#MDLS="-m als,arb,aus,cca,cea,chn,eur,gld,haw,idn,ind,jpn,kor,mex,mie,naf,nea,nwa,nzl,pac,phl,png,sam,sea,ssa,twn,waf,zaf"
#MDLS="-m usa,cnd"
#MDLS="-m waf,ssa"
#
# Run hazard curves homogenisation
../create_homogenised_curves.py $C_SHP $O_PATH $D_PATH $SIDX $B_SHP $IMTSTR $I_SHP $BUF $MDLS $VS30_FLAG
