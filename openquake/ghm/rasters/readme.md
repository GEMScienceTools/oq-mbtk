Use make_rasters.sh to build a raster for each specified IMT-POE combo.

First, set the variables SRCDAT, LAYERS, VERSION, and VVER at the top of the file
* SRCDAT: uses the unique part of the file name, for example v2022_1_sa02_475.csv -> "sa02_475"
* LAYERS: the name of the column in the csv file to be used as the raster cell value 
* VERSION: the version of the global map to be rasterized corresponding to the name of the 
	directory in mosaic/global_map that 
* VVER: the mosaic version as indicated in the csv names, i.e. v2022_1_sa02_475.csv -> "v2022_1" 
	(could be the same as VERSION)

You must also set two environmental variables (e.g. export GEM_DATA=<path_to_repo>) that point to required subrepos:

* GEM_DATA points to https://gitlab.openquake.org/hazard/gem-hazard-data/ 
* GEM_MOSAIC points to the location where https://gitlab.openquake.org/hazard/mosaic/global_map is cloned
