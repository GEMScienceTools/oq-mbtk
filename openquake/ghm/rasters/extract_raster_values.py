import processing

points_file_rock = '/Users/kjohnson/GEM/oq-mbtk/openquake/ghm/rasters/mosaic_pts.shp'
points_file_vs30 = '/Users/kjohnson/GEM/oq-mbtk/openquake/ghm/rasters/mosaic_pts_vs30.shp'
raster_path = '/Users/kjohnson/GEM/oq-mbtk/openquake/ghm/rasters/'
test_out = '/Users/kjohnson/GEM/oq-mbtk/openquake/ghm/rasters/test-rasters/'

rasters_rock = ['pga_475_rock', 'pga_2475_rock',
           'sa02_475_rock', 'sa02_2475_rock',
           'sa10_475_rock', 'sa10_2475_rock']

rasters_vs30 = ['pga_475_vs30','pga_2475_vs30', 
            'sa02_475_vs30','sa02_2475_vs30',
            'sa10_475_vs30', 'sa10_2475_vs30']

for raster in rasters_rock:
    print('processing {}'.format(raster))
    processing.run("native:rastersampling", \
    {'INPUT':points_file_rock,\
    'RASTERCOPY':raster_path+'v2022_1_{}.tif'.format(raster),\
    'COLUMN_PREFIX':'SAMPLE_',\
    'OUTPUT':test_out+'{}.csv'.format(raster)}
    )
    
for raster in rasters_vs30:
    print('processing {}'.format(raster))
    processing.run("native:rastersampling", \
    {'INPUT':points_file_vs30,\
    'RASTERCOPY':raster_path+'v2022_1_{}.tif'.format(raster),\
    'COLUMN_PREFIX':'SAMPLE_',\
    'OUTPUT':test_out+'{}.csv'.format(raster)}
    )
