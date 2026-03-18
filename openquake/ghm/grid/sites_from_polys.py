from shapely.geometry import shape
from shapely import wkt, Polygon, MultiPolygon, LineString, get_num_interior_rings
from shapely.affinity import translate
from shapely.ops import split
import geopandas as gpd
import h3
import pandas as pd
import numpy 
import os

def fill_poly(poly, h3_level):
    ''' 
    Fills a polygon with sites at specified h3 level
    Loops through multiple features if necessary
    Splits antimeridian polygons at -180, 180 

    :param poly:
        polygon to fill with sites
    :param h3_level:
        h3 level for filling 

    :returns:
        dataframe of lat, lon for sites in polygon

    '''
    new_sites = []

    gds = gpd.GeoSeries(poly)
    gdata = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gds))
        
    # Create geojson and find the indexes of the points inside
    eee = gds.explode(index_parts=True)
    feature_coll = eee.__geo_interface__
    print('found ', len(feature_coll['features']), 'polygons for source')
        
    for i in range(len(feature_coll['features'])):
        try: 
            tmp = feature_coll['features'][i]['geometry']
            tmp_poly = shape(feature_coll['features'][i])
            if (tmp_poly.bounds[2]-tmp_poly.bounds[0] > 180) or abs(tmp_poly.bounds[0]) > 180 or abs(tmp_poly.bounds[2]) > 180:
                
                print("found poly crossing antimeridian")
                modified_coords = []
                for coords in tmp_poly.exterior.coords:
                    long, lat = coords
                    
                    if tmp_poly.bounds[2] - long > 180:
                        long += 360
                    modified_coords.append((long, lat))
                
                modified_poly = Polygon(modified_coords)

                # Split the modified polygon at 180 or -180 degrees
                # Even if not split, this returns a multipolygon
                if modified_poly.bounds[2] < -180 or modified_poly.bounds[0] < -180:
                    split_line = LineString(((-180, -90), (-180, 90)))
                    split_polys = split(modified_poly, split_line)
                else:
                    split_line = LineString(((180, -90), (180, 90)))
                    split_polys = split(modified_poly, split_line)

                    
                # Move any parts east of split line back to western hemisphere
                polys = []
                for geom in split_polys.geoms:
                    if geom.bounds[0] >= 180:
                        geom = translate(geom, xoff=-360)
                    if geom.bounds[0] <= -180:
                        geom = translate(geom, xoff=360)
                    if geom.bounds[2] >= 180:
                        geom = translate(geom, xoff=-360)
                    if geom.bounds[2] <= -180:
                        geom = translate(geom, xoff=360)

                    tmp_feature = geom.__geo_interface__
                    tidx_a = h3.polygon_to_cells(h3.geo_to_h3shape(tmp_feature), h3_level)
                    print('adding ', len(tidx_a), ' sites')
                    new_sites.extend(tidx_a)

            else:
                tidx_a = h3.polygon_to_cells(h3.geo_to_h3shape(tmp), h3_level)
                print('adding ', len(tidx_a), ' sites')
                new_sites.extend(tidx_a)
        except:
            print("something went wrong :(")

    sites_indices = list(set(new_sites))
    sidxs = sorted(sites_indices)
    tmp = numpy.array([h3.cell_to_latlng(h) for h in sidxs])
    sites = numpy.fliplr(tmp)
    sites_test = numpy.flip(tmp, axis=1)
    sites_df = pd.DataFrame(tmp)
    
    # round to avoid some duplication issues
    sites_df[0] = numpy.round(sites_df[0], 5)
    sites_df[1] = numpy.round(sites_df[1], 5) 
    sites_df.columns=['lat', 'lon']

    return sites_df

def make_sites_for_polys_splitting(poly_df, out_folder, h3_fixed = False, h3_level = 6):
    '''
    Make sites from a polygon dataframe 
    Splits antimeridian polygons at -180, 180 

    :param poly_df:
        geopandas dataframe of polygons to fill with sites, 
        will loop through entries, should include a 'name' column.
    :param out_folder:
        location to save output
    :param h3_fixed:
        should a fixed h3 level be used? If not, there should be a column called 'resolution'
    :param h3_level:
        if h3 is fixed, specify the resolution for sites 
    
    '''
    
    for idx, row in poly_df.iterrows():
        
        if h3_fixed == True:
            h3_level = h3_level
        else:
            h3_level = row.resolution

        print('Using h3 level ', h3_level, ' for source ', row.name)

        sites_df = fill_poly(row.geometry, h3_level)
        
        # Should not be any duplicates! But it might be possible
        # especially with lower res, so check
        sites_df = sites_df.drop_duplicates()

        out_file = 'sites_mosaic_2026_{}.csv'.format(row.name)
        out = os.path.join(out_folder, out_file)
        print(out)
        sites_df.to_csv(out, index = False)