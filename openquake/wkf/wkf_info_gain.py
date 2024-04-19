#! /usr/bin/env python

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import h3

from shapely.geometry import Point
from openquake.baselib import sap
from openquake.hazardlib.geo.geodetic import geodetic_distance, distance, _prepare_coords 


def poiss_loglik(rate_lambda, obs, T):
    '''
    Calculates Poisson likelihood for given lambda_rate and observations (obs).
    
    :param rate_lambda:
        Rate of events (lambda)
    :param obs:
        Observed number of events
    :param T:
        Time over which model rate is constructed/tested. e.g. if the forecast rate is for 1 year and the observed data is for 10, T = 10
        
    '''
    l = -rate_lambda*T - np.log(sp.special.factorial(obs)) + np.log(rate_lambda*T)* obs
    return l

def information_gain(catalogue, h3_map, h3_level, smooth_out, T = 1, for_zone = False):
    '''
    Calculates the information gain from a smoothing model, comparing observed and smoothed rates per h3 cell.
    The information gain is calculated as exp(llhood - unif_llhood)/total_event_num, where poisson likelihoods are calculated using poiss_loglik function. 
    The uniform likelihood is determined by dividing the observed number of events uniformly across all h3 cells.
    Significantly slower than the version in adaptive_smoothing.py!
    
    :param catalogue:
        Location of the catalogue to be used when evaluating the model
    :param h3_map:
        Location of file including the h3 cells over which the model has been built
    :param h3_level:
        Resolution for h3 mapping (must match that used to create the h3_map)
    :param smooth_out:
        Output of smoothing file (from adaptive or fixed smoothing), containing columns named lon, lat, nocc. Lon/Lat locations should correspond to h3_cells
    :param T: 
        Time rescaling if neccessary
    
    :returns: 
        information gain for given model relative to uniform poisson model with correct number of events
    '''
    colnames = ["h3_cell", "zid"]
    h3_idx = pd.read_csv(h3_map, names=colnames, header = None)
    
    cat_df = pd.read_csv(catalogue)
    cat_df = gpd.GeoDataFrame(cat_df, crs='epsg:4326', geometry=[Point(xy) for xy
                       in zip(cat_df.longitude, cat_df.latitude)])
    #print(len(cat_df))                   
    smoothed = pd.read_csv(smooth_out)
    smoothed = gpd.GeoDataFrame(smoothed, crs='epsg:4326', geometry=[Point(xy) for xy
                       in zip(smoothed.lon, smoothed.lat)])
    
    
    # Find which cell each event in the catalogue belongs to    
    h3_cell_c = [0]*len(cat_df)
    for i in range(0,len(cat_df)):
        h3_cell_c[i] = h3.geo_to_h3(cat_df['latitude'][i], cat_df['longitude'][i], h3_level)
        
    cat_df['h3_cell'] = h3_cell_c
    
    ## Find which cell each smoothed value is in
    h3_cell_sm = [0]*len(smoothed)
    for i in range(0,len(smoothed)):
        h3_cell_sm[i] = h3.geo_to_h3(smoothed['lat'][i], smoothed['lon'][i], h3_level)
    
    smoothed['h3_cell'] = h3_cell_sm
    
    # Only keep cells where smoothed value is in h3_map
    to_use = smoothed[smoothed['h3_cell'].isin(list(h3_idx.h3_cell))]
    
    # count events in each h3 cell
    event_count = [0]*len(h3_idx)
    for i in range(0, len(h3_idx)):
        if cat_df['h3_cell'].str.contains(h3_idx.iloc[i, 0]).any():
            event_count[i] = cat_df['h3_cell'].value_counts()[h3_idx.iloc[i,0]]
    
    h3_idx['count'] = event_count
    
    # Combine cell information into one dataframe
    h3_idx =h3_idx.merge(to_use, how = 'left', on = "h3_cell")
    
    # For cells where there is no calculated smoothing, set to a very small value
    # (0s break the likelihood calculation, but 1E-15 is probably close enough)
    h3_idx['nocc'] = h3_idx['nocc'].replace(np.nan, 1E-15)
         
    # Uniform rate = total sum distributed over all hexagons, so uniform count is sum/num hexagons
    unif_cnt = sum(h3_idx['count'])/len(h3_idx)
    # Calculate poisson likelihood of uniform model
    unif_llhood = poiss_loglik(unif_cnt, h3_idx['count'], T)
    
    # Model likelihood
    mod_llhood = poiss_loglik(h3_idx['nocc'], h3_idx['count'], T)
    
    if any(mod_llhood < -500):
        print("-inf in likelihoods (probably cells with many events!)")
        mod_llhood[mod_llhood < -500] = -500
        unif_llhood[unif_llhood < -500] = -500
        
    # Information gain = exp(llhood - unif_llhood)/total_event_num
    IG = np.exp((sum(mod_llhood)-sum(unif_llhood))/sum(h3_idx['count']))
    return IG

