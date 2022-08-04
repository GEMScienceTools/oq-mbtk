### set up arrays for csv files to rasterize.  
# SRCDAT is the unique string inside the name of the csv file to be rasterized.
# LAYERS is the name of the data z-field (see header of relevant SRCDAT file)
# VERSION is the version of the global maps (i.e. v2022-1)
# VVER is required because a different string format is used to name the mosaic version 
# directory inside of global_map and the name of the csv file itself - could be changed
# after 2022?

SRCDAT=("pga_475" "pga_2475" "sa02_475" "sa02_2475" "sa10_475" "sa10_2475")
LAYERS=("PGA-0.002105" "PGA-0.000404" "SA(0.2)-0.002105" "SA(0.2)-0.000404" "SA(1.0)-0.002105" "SA(1.0)-0.000404")
VERSION="2022-1"
VVER="v2022_1"


### Create inland raster (only needs to be done once for all maps)
# using global_inland_areas.shp file  in gem-hazard-data/gis/inland_areas. 
# Must set $GEM_DATA environment variable 
# enable the ALL_TOUCHED (-at) option to include all pixels touched by lines or polygons
# using 0.049996666666667 to be consistent with the global tif map generated above

INSHP=$GEM_DATA"/gis/inland_areas/global_inland_areas.shp"
INTIF="inland.tif"
VRT="tmp.var"

gdal_rasterize -burn 1 -te -180 -60 180 89.99 -tr 0.05 0.049996666666667 -at -a_nodata nan -l global_inland_areas $INSHP $INTIF 



for i in ${!LAYERS[@]}; do

	
	# set up the .vrt file. variables that will replace those in template.var 
	# must be exported
	export LAY=${LAYERS[$i]} 
	SLROOT=$VVER"_"${SRCDAT[$i]}
	SDS=$GEM_MOSAIC"global_map/"$VERSION"/"$SLROOT".csv"
	export SL=$SLROOT"_rock"
	export CSV=$SL".csv"
	envsubst < template.var > $VRT 

	# copy the csv to a new temporary file to exclude the first line (comments)
	tail -n +2 $SDS > $CSV 

	# set the names of the output tifs: one full one, and one that clips to inland areas
	TIF=$SL"_full.tif"
	CUTTIF=$SL".tif"

	### Create hazard map raster via interpolation for each csv
	# Use 89.99 -60 instead of -60  89.99  to ensure upper-left corner 
	# coordinates are correct. Must do this for next step to work correctly
	# power=25 is the value needed for raster maps to match the csv values within ~0.05g

	echo "Making " $TIF "from" $LAY $VRT 
	gdal_grid -a invdistnn:power=25:max_points=3.0:nodata=-nan -txe -180 180 -tye 89.99 -60 -tr 0.05 0.05 -of GTiff -ot Float64 -l $LAY $VRT $TIF --config GDAL_NUM_THREADS ALL_CPUS

	###  Cut hazard map to inland areas
	echo "cutting "$TIF" with "$INTIF" to make "$CUTTIF
	gdal_calc.py -A $TIF -B $INTIF --outfile=$CUTTIF --calc='A*B' --overwrite

	# repeat for vs30
	SDS=$GEM_MOSAIC"global_map/"$VERSION"/vs30/"$SLROOT"_vs30.csv"
	export SL=$SLROOT"_vs30"
	export CSV=$SL".csv"
	envsubst < template.var > $VRT 
	tail -n +2 $SDS > $CSV 

	TIF=$SL"_full.tif"
	CUTTIF=$SL".tif"


	gdal_grid -a invdistnn:power=25:max_points=3.0:nodata=-nan -txe -180 180 -tye 89.99 -60 -tr 0.05 0.05 -of GTiff -ot Float64 -l $LAY $VRT $TIF --config GDAL_NUM_THREADS ALL_CPUS 

	gdal_calc.py -A $TIF -B $INTIF --outfile=$CUTTIF --calc='A*B' --overwrite

done

rm $VRT *csv *full.tif 

