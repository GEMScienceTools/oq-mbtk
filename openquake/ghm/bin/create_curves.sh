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

#
# Path to the folder with the mosaic repository
# For the original hazard map we used this folder:
# /Users/mpagani/Documents/2018/diary/11/13_ucf/maps
#D_PATH=$REPOS'/mosaic'
D_PATH=$GEM_MOSAIC
#
# Spatial index folder
SIDX='../GGrid/trigrd_split_9_spacing_13'
#
# Boundaries shapefilei
# $GEM_DATA is the path to gem-hazard-data
B_SHP=$GEM_DATA'gis/grid/gadm_410_level_0.gpkg'

# Former shapefile used before v2023
#B_SHP='./../data/gis/world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
#
#
# Shapefile with inland territories
I_SHP='./../data/gis/inland.shp'
#
BUF=50.0
#
# List of models to be processed. This is an optional parameter. If not set,
# i.e. MDLS="", all the models specified in `openquake.ghm.mosaic.DATA`
# will be used.
#MDLS="-m als,arb,aus,cca,cea,chn,eur,gld,haw,idn,ind,jpn,kor,mex,mie,naf,nea,nwa,nzl,pac,phl,png,sam,sea,ssa,twn,waf,zaf"
#MDLS="-m usa,cnd"
MDLS="-m waf,ssa"

# path to where the maps are stored
O_PATH_MAP='/tmp/ghm/global'
PREFIX='map'
# String with the intensity measure type
for IMTSTR in 'PGA' 'SA(0.2)' 'SA(0.3)' 'SA(0.6)' 'SA(1.0)' 'SA(2.0)'
do
	#flag set to 1 for vs30 calc or 0 for rock
	for VS30_FLAG in 0 1
	do
	
		# Output path for hazard curves
		if (($VS30_FLAG == 1 ))
		then
			SITECOND='vs30'
		else
			SITECOND='rock'
		fi

		O_PATH='/tmp/ghm/'$IMTSTR'-'$SITECOND

		# Run hazard curves homogenisation
		../create_homogenised_curves.py $C_SHP $O_PATH $D_PATH $SIDX $B_SHP $IMTSTR $I_SHP $BUF $MDLS $VS30_FLAG
		
		# loop through probabilities of exceedance
		for PEX in '0.002105' '0.000404'
		do
			# file name for final map
			O_NAME='v2023-1-'$IMTSTR'-'$SITECOND'-'$PEX'.csv'
			../create_map_from_curves.py $O_PATH $PREFIX $O_NAME $O_PATH_MAP $IMTSTR $PEX

		done
	done

done
