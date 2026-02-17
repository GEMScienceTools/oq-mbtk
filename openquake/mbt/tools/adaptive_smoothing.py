'''
Module :mod:`openquake.mbt.tools.adaptive_smoothing`

Implements the spatial adaptive smoothing method of Helmstetter et al (2007) 

'''
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
from shapely.geometry import Point
from openquake.baselib import sap
from openquake.hazardlib.geo.geodetic import geodetic_distance, distance, _prepare_coords 
import h3

class AdaptiveSmoothing(object):
    '''
    Applies Helmstetter (2007) adaptive smoothing kernel to the data
    
    
    '''
    
    def __init__(self, locations, grid=False, use_3d=False, bvalue=None, use_maxdist = False, weight = None):
        '''
        Instantiate class with locations at which to calculate intensity

        :param locations:
            Expects a list of longitude and latitude locations. Set grid = True to constrcut meshgrid from x, y locations.

        :param bool use_3d:
            Choose whether to use hypocentral distances for smoothing or only
            epicentral

        :param float bval:
            b-value for analysis

        :param bool use_maxdist:
            Use to impose maximum distance on events to consider as neighbour. If True, config should contain additional parameters:
            'maxdist' (km) and 'h3res' for required resolution.
        '''
        self.locations = locations
        self.catalogue = None
        self.grid = grid
        self.use_3d = use_3d
        self.bval = bvalue
        if self.bval:
            self.beta = self.bval * log(10.)
        else:
            self.beta = None
        self.data = None

        self.kernel = None
        self.use_maxdist = use_maxdist

    
    def run_adaptive_smooth(self, catalogue, config):
        '''
        Runs an analysis of adaptive smoothed seismicity of Helmstetter et al (2007).

        :param catalogue:
            Instance of the openquake.hmtk.seismicity.catalogue.Catalogue class
            catalogue.data dictionary containing the following -
            'year' - numpy.ndarray vector of years
            'longitude' - numpy.ndarray vector of longitudes
            'latitude' - numpy.ndarray vector of latitudes
            'depth' - numpy.ndarray vector of depths
            (can use CsvParser to adapt any csv file to Catalogue for this step)

        :param dict config:
            Configuration settings of the algorithm:
            * 'kernel' - Kernel choice for adaptive smoothing. Options are "Gaussian" 
               or "PowerLaw" (string)
            * 'n_v' - number of nearest neighbour to use for smoothing distance (int)
            * 'd_i_min' - minimum smoothing distance d_i, should be chosen based on 
              location uncertainty. Default of 0.5 in Helmstetter et al. (float)
            Optional (required only when use_maxdist = True)
            * 'maxdist' - in km, the maximum distance at which to consider other events
            * 'h3res' - the h3 resolution for the smoothing calculations
            
            
            
        :returns:
            Smoothed seismicity data as np.ndarray, of the form
            [Longitude, Latitude, Smoothed_value]
            if maxdist == True, the smoothed values will be normalised to the number of 
            events in the catalogue. This makes the output directly comparable with the 
            Gaussian smoothing options in the mbtk
        '''

        
        data = np.stack((catalogue.data['longitude'], 
        				 catalogue.data['latitude'],
        				 catalogue.data['depth'], 
        				 catalogue.data['magnitude']), axis = 1)
        self.data = data
        # To create a grid of data if required:
        if self.grid == True:
        	## Convert x, y locs to meshgrid
        	xx, yy = np.meshgrid(self.locations[0], self.locations[1])
        	## Flatten to get all x and y coords on grid
        	x = xx.flatten()
        	y = yy.flatten()
        else:
        	x = self.locations[0]
        	y = self.locations[1]
        	
	
        n_v = (config['n_v'] -1)
        kernel = config['kernel']

        # Use depths if 3D model required, otherwise all depths set to 0
        if self.use_3d:
            depth = data[:,2]
        else:
            depth = np.zeros(len(data))
        
        d_i = np.empty(len(data))
        
        # Get smoothing parameter d for each earthquake 

        if self.use_maxdist == True:
            maxdist = config['maxdist']
            h3res = config['h3res']

            def lat_lng_to_h3(row):
                return h3.latlng_to_cell(row.lon, row.lat, h3res)

            h3_df = pd.DataFrame(data)
            h3_df.columns = ['lon', 'lat', 'depth', 'mag']
            h3_df['h3'] = h3_df.apply(lat_lng_to_h3, axis=1)
            maxdistk = int(np.ceil(maxdist/h3.average_hexagon_edge_length(
                h3res, unit='km')))

            # Consider only neighbours within maxdistk
            for iloc in range(0, len(data)):
                base = h3_df['h3'][iloc]
                tmp_idxs = h3.grid_disk(base, maxdistk)
                ref_locs = h3_df.loc[h3_df['h3'].isin(tmp_idxs)]
                r = distance(ref_locs['lon'], ref_locs['lat'], ref_locs['depth'], data[iloc, 0], data[iloc, 1], depth[iloc])
                # because of filtering, we are now working with a series so treat accordingly!
                r.sort_values(inplace = True)
                
                if len(r) > (n_v + 1): 
                    # Have not removed distance to self here, so add 1 to n_v
                    d_i[iloc] = r.iloc[n_v + 1]
                    
                else:
                    # no n_vth neighour within maximum distance, set d_i to maxdist
                    d_i[iloc] = maxdist 
        else:
            # Get smoothing parameter d for each earthquake
            for iloc in range(0, len(data)):
                r = distance(data[:, 0], data[:, 1], depth, data[iloc, 0], data[iloc, 1], depth[iloc])
                r = np.delete(r, iloc)
                r.sort()
                d_i[iloc] = r[n_v]

        # Set minimum d_i
        d_i[d_i < config['d_i_min']] = config['d_i_min']
        mu_loc = np.empty(len(x))
        
	    # Calculate mu at each location 
        for iloc in range(0, len(x)):
 	    # Distance from each event to the location
            r_dists = distance(data[:, 0], data[:, 1], depth, x[iloc], y[iloc], 0)
            mu_loc[iloc] = self.mu_int(r_dists, d_i, kernel = kernel)
        
        if self.use_maxdist == True:
            # normalise mu_loc to number of observed events
            mu_norm = mu_loc/sum(mu_loc)
            nocc = mu_norm*len(catalogue.data['longitude']) 
        
            self.out = pd.DataFrame({'lon': x, 'lat' : y, 'nocc': nocc})

        else:
            self.out = pd.DataFrame({'lon': x, 'lat' : y, 'nocc': mu_loc})
        return self.out

    
    def Gaussian_K(self, r, d):
        '''
        Calculates value of Gaussian smoothing kernel K at a given point location
        
        :param d:
            Smoothing distance: the distance between event i and the nth neighbour, where n is an optimisable parameter   
        :param r:
            Distance between event and a point location 
            
        :returns:
            Kernel estimate of smoothed seisimicity at location r
            
        '''
        # Normalising factor C so integral of K = 1
        C = 1/(np.sqrt(2*np.pi)*d)
    	# Gaussian smoothing kernel with weight d
        K = C*np.exp(-(r**2/(2*d**2)))
        
        return(K)
    	
    def PL_K(self, r, d):
        '''
        Calculates value of power law smoothing kernel K at a given point location
        
        :param d:
            Smoothing distance: the distance between event i and the nth neighbour, where n is an optimisable parameter
        :param r:
            Distance between event and a point location 
            
        :returns:
            Kernel estimate of smoothed seisimicity at location r
            
        '''
        ## Normalising constant so that integral (-inf, inf) = 1
        C = (d**2)/2
        K = C/(r**2 + d**2)**1.5
        return(K)
    

    def mu_int(self, r, d, kernel = "Gaussian"):
        '''
        Calculates intensity at given locations as sum of all kernel contributions K
        
        :param d:
            Smoothing distance: the distance between event i and the nth neighbour, where n is an optimisable parameter
        :param r:
            Distance between event and a point location x
        :param kernel:
            Options for choice of smoothing kernel. Accepts "Gaussian" or "PowerLaw" and calls the appropriate kernel
            
        :returns:
            * mu intensity at a point x as a function of all surrounding events with smoothing distance d
            
        '''
        mu_i = np.empty(len(r))
    
        for i, distance in enumerate(r):
            
            if kernel == "Gaussian":
                mu_i[i] = self.Gaussian_K(distance, d[i])
            
            elif kernel == "PowerLaw":
                mu_i[i] = self.PL_K(distance, d[i])
    
        mu_sum = (np.sum(mu_i))
        return mu_sum

 
    
    def poiss_loglik(self, rate_lambda, obs, T=1):
        '''
        Calculate the poisson likelihood, given the observed number of events and the forecast rate. 
        
        :param rate_lambda:
            expected number of events 
        :param obs:
            observed number of events
        :param T:
            Scaling of forecast to observations. For example, a forecast built with 100 years of data tested over a ten year testing period should have T = 0.1, or a 1 year forecast tested over 5 years would have T = 5
            
        :returns:
            likelihood value 
        '''
        l = -rate_lambda*T - np.log(sp.special.factorial(obs)) + np.log(rate_lambda*T)* obs
        return l

    def plot_smoothing(self, log_density = False, plot_cat = False):
        '''
        Plot adaptive smoothing outputs given a csv file of smoothing values (lon, lat, nocc).
        Compatible with output of adaptive or fixed kernel smoothing approaches
        
        :param smooth_out:
            Location of output file containing the adaptive smoothing results
        
        '''
        out = self.out
        if (log_density == True):
            density = np.log10(out["nocc"])
            lab = "log10 event density"
        else:
            density = out["nocc"]
            lab = "event density"
        plt.scatter(out["lon"], out["lat"], c=density, cmap="viridis")
        plt.colorbar(label= lab)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        if (plot_cat == True):
            plt.scatter(self.data[:, 0], self.data[:,1], s = 1, c = "k")
        plt.show()
        

    def information_gain(self, counts, T = 1, read_counts_from_file = False, fname_counts = None):
        '''
        Calculates the information gain per event given a smoothing model and a list of model counts
        Counting results should match the cells of the smoothed rates. 
        
        :param counts:
            counts in grid cells matching the smoothed cells stored in self.out
        :param T:
            optional time scaling for testing rates over a fixed time period
        :param read_counts_from_file:
            if True, read count data from a csv instead of using directly supplied counts
        :param fname_counts:
            if read_counts_from_file is True, read counts from specified file.
             
        '''
        
        IG = _information_gain(out = self.out, counts = counts, T = 1, read_counts_from_file = False, fname_counts = None)
        
        return IG
        
def poiss_loglik(rate_lambda, obs, T=1):
        '''
        Calculate the poisson likelihood, given the observed number of events and the forecast rate. 
        
        :param rate_lambda:
            expected number of events 
        :param obs:
            observed number of events
        :param T:
            Scaling of forecast to observations. For example, a forecast built with 100 years of data tested over a ten year testing period should have T = 0.1, or a 1 year forecast tested over 5 years would have T = 5
            
        :returns:
            likelihood value 
        '''
        l = -rate_lambda*T - np.log(sp.special.factorial(obs)) + np.log(rate_lambda*T)* obs
        return l
        
def _information_gain(out, counts, T = 1, read_counts_from_file = False, fname_counts = None):
        '''
        Calculates the information gain per event given a smoothing model and a list of model counts
        Counting results should match the cells of the smoothed rates. 
        This version works on an output, so is applicable to a Gaussian smoothing model as well as 
        an adaptive-smoothing object.
        
        :param outs: 
            smoothing output, should contain columns 'lon', 'lat', 'nocc' 
            This is the standard output from both adaptive and Gaussian smoothing
        :param counts:
            counts in grid cells matching the smoothed cells stored in self.out
        :param T:
            optional time scaling for testing rates over a fixed time period
        :param read_counts_from_file:
            if True, read count data from a csv instead of using directly supplied counts
        :param fname_counts:
            if read_counts_from_file is True, read counts from specified file.
             
        '''
        
        # Set counts from a file if provided, or using count array directly
        if (read_counts_from_file == True):
            loc_df = pd.read_csv(fname_counts)
            loc_counts = gpd.GeoDataFrame(loc_df, crs='epsg:4326', geometry=[Point(xy) for xy
                       in zip(loc_df.lon, loc_df.lat)])
        else:
            loc_counts = pd.DataFrame({'lon': out['lon'], 'lat' : out['lat'], 'nocc' : out['nocc']})
    
        # Uniform rate = total sum distributed over all hexagons, so uniform count is sum/num cells
        unif_cnt = sum(loc_counts['nocc'])/len(loc_counts)
        # Calculate poisson likelihood of uniform model
        unif_llhood = poiss_loglik(unif_cnt, loc_counts['nocc'], T)
    
        #smoothed = out
        smoothed = gpd.GeoDataFrame(out, crs='epsg:4326', geometry=[Point(xy) for xy
              in zip(out.lon, out.lat)])
        
        # Model likelihood
        mod_llhood = poiss_loglik(smoothed['nocc'], loc_counts['nocc'], T)
        # replace nan values with 0 likelihood
        mod_llhood = mod_llhood.fillna(0)
        
        # Information gain = exp(llhood - unif_llhood)/total_event_num
        IG = np.exp((sum(mod_llhood)-sum(unif_llhood))/sum(loc_counts['nocc']))
        return IG
