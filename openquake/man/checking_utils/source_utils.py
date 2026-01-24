import os
import pathlib
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
from scipy.stats import poisson
from tabulate import tabulate

from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter


DEFAULT = SourceConverter(
	investigation_time=1.0,
	rupture_mesh_spacing=10.,
	area_source_discretization=10.)


def get_params(ssm, mmin = 0, total_for_zone = False):
	'''
	Retrieve source parameters from xml source file for multipoint source
	Specify total_for_zone to retain only total a value summed over all points

	:param ssm:
	    seismic source model retreived from xml file with to_python
	:param mmin:
	    minimum magnitude to consider
	:param total_for_zone:
	    if True, returns the total a-value for the zone. 
	    if False, returns a-values at point locations
	'''
	total_rate_above_zero = 0
	data = []
	for ps in ssm[0][0]:
		agr = ps.mfd.a_val
		bgr = ps.mfd.b_val
		mmax = ps.mfd.max_mag
		lo = ps.location.longitude
		la = ps.location.latitude
		log_rate_above = agr - bgr * mmin
		total_rate_above_zero += 10**(agr)
			
		data.append([lo, la, log_rate_above, bgr, mmax])

	if total_for_zone == True:  
		# total_rate_above += 10**(agr - bgr * m_min)
		return np.log10(total_rate_above_zero), bgr, mmax
	else: 
		return np.array(data)


def get_params_points(ssm, mmin = 0, total_for_zone = False):
	'''
	Retrieve source parameters from xml source file for point sources
	Specify total_for_zone to retain only total a value summed over all points

	:param ssm:
	    seismic source model retreived from xml file with to_python
	:param mmin:
	    minimum magnitude to consider
	:param total_for_zone:
	    if True, returns the total a-value for the zone. 
	    if False, returns a-values at point locations
	'''
	total_rate_above_zero = 0
	data = []

	for ps in ssm[0]:
		agr = ps.mfd.a_val
		bgr = ps.mfd.b_val
		mmax = ps.mfd.max_mag
		lo = ps.location.longitude
		la = ps.location.latitude
		log_rate_above = agr - bgr * mmin
		total_rate_above_zero += 10**(agr)
			
		data.append([lo, la, log_rate_above, bgr, mmax])

	if total_for_zone == True:  
		# total_rate_above += 10**(agr - bgr * m_min)
		return np.log10(total_rate_above_zero), bgr, mmax
	else: 
		return np.array(data)


def plot_sources(folder_name,
				 region,
				 mmin,
				 fig_folder,
				 rate_range=[-9, 3, 0.2],
				 mmax_range=[6.0, 9.0, 0.2], 
				 sconv=DEFAULT,
				 poly_name='',
				 plot_poly=False,
				 pointsource=False):
	'''
	Create plots of source rates and mmax for each source in folder_name

	:param folder_name:
		folder containing source xml files
	:param region:
	    region for pygmt plotting
	:param mmin:
		minimum magnitude to be used in model
	:param fig_folder:
		folder in which to store plots
	:param rate_range:
		range of rate to use in colour scale,
		specified as [min, max, interval]
	:param mmax_range:
	    range of mmax for colour scale, 
	    specified as [min, max, interval]
	:param poly_name:
		location of file containing the model source polygon (optional)
	:param plot_poly:
	    boolean specifying if polygon outline should be plotted
	:param pointsource:
		alternative where point sources are used instead of multipoint
	'''
	path = pathlib.Path(folder_name)

	# Make rate and mmax folders if they do not exist
	if os.path.exists(os.path.join(os.path.join(fig_folder, 'rate'))):
		print("found folders at ", os.path.join(os.path.join(fig_folder, 'rate')))
	else:
		os.mkdir(os.path.join(fig_folder, 'rate'))
		os.mkdir(os.path.join(fig_folder, 'mmax'))

	# Set up colour scales
	cpt_rate = os.path.join(fig_folder, 'rate.cpt')
	pygmt.makecpt(cmap="turbo", series=rate_range, output=cpt_rate)
	cpt_mmax = os.path.join(fig_folder, 'mmax.cpt')
	pygmt.makecpt(cmap="rainbow", series=mmax_range, output=cpt_mmax)
    
	if poly_name:
    	# Set plotting polygon
		poly = gpd.read_file(poly_name)
		poly["x"] = poly.representative_point().x
		poly["y"] = poly.representative_point().y

	for fname in sorted(path.glob('src_*.xml')):
		print(fname)
		ssm = to_python(fname, sconv)
		fig_a = pygmt.Figure()
		fig_a.basemap(region=region, projection="M15c", frame=True)
		fig_a.coast(land="grey", water="white")

		fig_b = pygmt.Figure()
		fig_b.basemap(region=region, projection="M15c", frame=True)
		fig_b.coast(land="grey", water="white")    
    
		vmin = +1e10
		vmax = -1e10
        
		if pointsource == False:
			data = get_params(ssm, mmin=mmin, total_for_zone = False)
			for grp in ssm:
				print(grp)
				for src in grp:
					name = src.name	
		else:
			data = get_params_points(ssm, mmin=mmin, total_for_zone = False)
			name = ssm.name

		vmin = np.min([vmin, min(data[:,2])])
		vmax = np.max([vmin, max(data[:,2])])

		fig_a.plot(x=data[:,0], 
			y=data[:,1], 
			style="h0.2", 
			fill=data[:, 2],
			cmap=cpt_rate)
 
		fig_b.plot(x=data[:,0], 
			y=data[:,1], 
			style="h0.2", 
			fill=data[:, 4],
			cmap=cpt_mmax)
        
		if plot_poly == True:
			fig_a.plot(data=poly, pen=".5p,black")
			fig_b.plot(data=poly, pen=".5p,black")
    
		fig_a.colorbar(frame=f'af+l"Log((N(m)>{mmin}))"', cmap=cpt_rate)    
		fig_b.colorbar(frame='af+l"Mmax"', cmap=cpt_mmax)
    
		out = os.path.join(fig_folder, 'rate', name+'rate.png')
		fig_a.savefig(out)
    
		out = os.path.join(fig_folder, 'mmax', name+'_mmax.png')
		fig_b.savefig(out)


def plot_all_sources(folder_name,
					 region,
					 mmin,
					 fig_folder,
				 	 rate_range=[-9, 3, 0.2],
					 mmax_range=[6.0, 9.0, 0.2], 
				 	 sconv=DEFAULT,
					 poly_name='',
					 plot_poly=False,
					 pointsource=False,
					 plot_fname=""):
	'''
	Plots a single plot of source rates and mmax using each source in folder_name

	:param folder_name:
		folder containing source xml files
	:param region:
	    region for pygmt plotting
	:param mmin:
		minimum magnitude to be used in model
	:param fig_folder:
		folder in which to store plots
	:param rate_range:
		range of rate to use in colour scale,
		specified as [min, max, interval]
	:param mmax_range:
	    range of mmax for colour scale, 
	    specified as [min, max, interval]
	:param poly_name:
		location of file containing the model source polygon (optional)
	:param plot_poly:
	    boolean specifying if polygon outline should be plotted
	:param pointsource:
		alternative where point sources are used instead of multipoint
	'''
	path = pathlib.Path(folder_name)

	# Make rate and mmax folders if they do not exist
	if os.path.exists(os.path.join(os.path.join(fig_folder, 'rate'))):
		print("found folders at ", os.path.join(os.path.join(fig_folder, 'rate')))
	else:
		os.mkdir(os.path.join(fig_folder, 'rate'))
		os.mkdir(os.path.join(fig_folder, 'mmax'))

	# Set up colour scales
	cpt_rate = os.path.join(fig_folder, 'rate.cpt')
	pygmt.makecpt(cmap="turbo", series=rate_range, output=cpt_rate)
	cpt_mmax = os.path.join(fig_folder, 'mmax.cpt')
	pygmt.makecpt(cmap="rainbow", series=mmax_range, output=cpt_mmax)
    
	if poly_name:
    	# Set plotting polygon
		poly = gpd.read_file(poly_name)
		poly["x"] = poly.representative_point().x
		poly["y"] = poly.representative_point().y

	dataset = np.empty([1,5])

	for fname in sorted(path.glob('src_*.xml')):
		ssm = to_python(fname, sconv)
		fig_a = pygmt.Figure()
		fig_a.basemap(region=region, projection="M15c", frame=True)
		fig_a.coast(land="grey", water="white")

		fig_b = pygmt.Figure()
		fig_b.basemap(region=region, projection="M15c", frame=True)
		fig_b.coast(land="grey", water="white")    
    
		vmin = +1e10
		vmax = -1e10
        
		if pointsource == False:
			data = get_params(ssm, mmin=mmin, total_for_zone = False)
			dataset = np.concatenate((dataset, data))

		else:
			data = get_params_points(ssm, mmin=mmin, total_for_zone = False)
			dataset = np.concatenate((dataset, data))
		
	vmin = np.min([vmin, min(dataset[:,2])])
	vmax = np.max([vmin, max(dataset[:,2])])

	fig_a.plot(x=dataset[:,0], 
		y=dataset[:,1], 
		style="h0.2", 
		fill=dataset[:, 2],
		cmap=cpt_rate)
 
	fig_b.plot(x=dataset[:,0], 
		y=dataset[:,1], 
		style="h0.2", 
		fill=dataset[:, 4],
		cmap=cpt_mmax)
        
	if plot_poly == True:
		fig_a.plot(data=poly, pen=".5p,black")
		fig_b.plot(data=poly, pen=".5p,black")
    
	fig_a.colorbar(frame=f'af+l"Log((N(m)>{mmin}))"', cmap=cpt_rate)    
	fig_b.colorbar(frame='af+l"Mmax"', cmap=cpt_mmax)

	out = os.path.join(fig_folder, 'rate', plot_fname+'_rate.png')
	fig_a.savefig(out)
    
	out = os.path.join(fig_folder, 'mmax', plot_fname+'_mmax.png')
	fig_b.savefig(out)


def simulate_occurrence(agr, bgr, rate, minmag, mmax, time_span, N=2000):
	'''
	Simulate number of occurrences from a Poisson distribution given the FMD parameters

	:param agr:
		a value for source
	:param bgr: 
	    b value for source
	:param minmag:
	    minimum magnitude to be considered 
	:param mmax: 
        maximum magnitude
    :param time_span:
        time span (in years) over which to simulate occurrences
    :param N:
        Number of simulations. Default to 2000
	'''
	num_occ = np.random.poisson(rate*time_span, N)
	return(num_occ)


def occurence_table(path_oq_input,
					path_to_subcatalogues,
					minmag,
					minyear,
					maxyear,
					N,
					src_ids=[],
					sconv=DEFAULT,
					pointsource=False):
	'''
	Check number of events expected from the source model against the number of observations.
	Uses N samples from a Poisson distribution with rate from source a and b value.
	Returns a table summarising the catalogue vs source model for zones in src_ids.

	:param path_oq_input: 
	    path to location of xml source models
	:param path_to_subcatalogues:
	    path to subcatalogues to compare source with
	:param minmag:
	    minimum magnitude to consider
	:param minyear: 
	    year to start analysis
	:param maxyear: 
	    end year for analysis
	:param N:
	    number of Poisson samples to use for comparison
	:param src_ids:
	    list of sources to use
	:param sconv:
	    source converter object specifying model setup
	:param pointsource:
		specify True if using set point source model rather than 
	'''
	table = []
	time_span = maxyear - minyear

	if len(src_ids) == 0:
    	# If source ids are not specified, take them from names of files in the specified folder
		dir_URL = pathlib.Path(path_oq_input)
		filename_list = [file.name for file in dir_URL.glob("*.xml")]
		for f in filename_list:
			m = re.search(r'(?<=_)\w+', f)
			src_ids.append(m.group(0))
            
	for src_id in sorted(src_ids):

		fname_src = os.path.join(path_oq_input, "src_{:s}.xml".format(src_id))
		ssm = to_python(fname_src, sconv)

		fname_cat = os.path.join(path_to_subcatalogues, "subcatalogue_zone_{:s}.csv".format(src_id))
		df = pd.read_csv(fname_cat)
		df = df.loc[df.magnitude > minmag]
		df = df.loc[(df.year >= minyear) & (df.year <= maxyear)]
		obs = len(df)

		if pointsource == False:
			agr, bgr, mmax = get_params(ssm, minmag, total_for_zone = True) 
		else:
			agr, bgr, mmax = get_params_points(ssm, minmag, total_for_zone = True) 

		rate = 10.0**(agr-minmag*bgr)-10.0**(agr-mmax*bgr)
		num_occ_per_time_span = simulate_occurrence(agr, bgr, rate, minmag, mmax, time_span, N)
    	
		mioc = min(num_occ_per_time_span)
		maoc = max(num_occ_per_time_span)
    
		perc_16 = np.percentile(num_occ_per_time_span, 16)
		perc_84 = np.percentile(num_occ_per_time_span, 84)
    
		perc_16 = poisson.ppf(0.16, rate*time_span)
		perc_84 = poisson.ppf(0.84, rate*time_span)
    
		agr_cat = np.nan
		if obs > 1e-10:
			agr_cat = np.log10(obs/time_span) + bgr * minmag
    
		pss = ""
		if obs >= perc_16 and obs <= perc_84:
			pss = "="
		elif obs < perc_16:
			pss = "<"
		else:
			pss = ">"

		table.append([src_id, agr_cat, agr, bgr, mioc, maoc, perc_16, perc_84, obs, pss])
	
	heads = ["Zone", "agr_cat", "agr", "bgr", "min", "max", "%16", "%84", "observed", "obs Vs. pred"]
	print(tabulate(table, headers=heads))


def source_info_table(folder_name, sconv = DEFAULT):
	'''
	Print a table describing the sources in this model, inclduing their ID, upper and lower depth limits,
	tectonic region, magnitude scaling relation, magnitude limits and depth distributions.

	:param folder_name:
	    folder to find soure xmls
	:param sconv:
	    source converter object
	'''
	columns = ['ID', 'Name', 'TR', 'USD', 'LSD', 'MSR', 'Mmin', 'Mmax', 'h_depths', 'b_vals']
	sdata = pd.DataFrame(columns=columns)
	path = pathlib.Path(folder_name)
	for fname in sorted(path.glob('src*.xml')):
		ssm = to_python(fname, sconv)
		for grp in ssm:
			for src in grp:
				mmin, mmax = src.get_min_max_mag()
				hdeps = [d[1] for d in src.hypocenter_distribution.data]
				row = [src.source_id,
					src.name,
					src.tectonic_region_type,
					src.upper_seismogenic_depth,
					src.lower_seismogenic_depth,
					src.magnitude_scaling_relationship,
					mmin,
					mmax,
					hdeps,
					src.mfd.kwargs['b_val']
					]
				sdata.loc[len(sdata.index)] = row
	
	print(tabulate(sdata, headers="keys", tablefmt="psql"))


def compare_sources_plot(src1_fname,
						 src2_fname,
						 region,
						 mmin,
						 fig_folder,
						 rate_range=[-9, 3, 0.2],
						 mmax_range=[6.0, 9.0, 0.2], 
						 sconv=DEFAULT,
						 poly_name='',
						 plot_poly=False,
						 pointsource=False,
						 plot_fname=""):
	'''
	Creates a plot showing the differences between two source models. 
	
	:param src1_fname:
		location of a file describing a point or multipoint source model
	:param src2_fname: 
		location of a file describing a point or multipoint source model
	:param region:
	    region for pygmt plotting
	:param mmin:
		minimum magnitude to be used in model
	:param fig_folder:
		folder in which to store plots
	:param rate_range:
		range of rate to use in colour scale,
		specified as [min, max, interval]
	:param mmax_range:
	    range of mmax for colour scale, 
	    specified as [min, max, interval]
	:param poly_name:
		location of file containing the model source polygon (optional)
	:param plot_poly:
	    boolean specifying if polygon outline should be plotted
	:param pointsource:
		alternative where point sources are used instead of multipoint
	'''
	# Set up colour scales
	cpt_rate = os.path.join(fig_folder, 'rate.cpt')
	pygmt.makecpt(cmap="turbo", series=rate_range, output=cpt_rate)
    
	if poly_name:
    	# Set plotting polygon
		poly = gpd.read_file(poly_name)
		poly["x"] = poly.representative_point().x
		poly["y"] = poly.representative_point().y

	ssm1 = to_python(src1_fname, sconv)
	ssm2 = to_python(src2_fname, sconv)

	if pointsource == False:
		data1 = get_params(ssm1, mmin=mmin, total_for_zone = False)
		data2 = get_params(ssm2, mmin=mmin, total_for_zone = False)

	else:
		data1 = get_params_points(ssm1, mmin=mmin, total_for_zone = False)
		data2 = get_params_points(ssm2, mmin=mmin, total_for_zone = False)

	rate_diff = data1[:,2] - data2[:,2]
	fig = pygmt.Figure()
	fig.basemap(region=region, projection="M15c", frame=True)
	fig.coast(land="grey", water="white")

	fig.plot(x=data1[:,0], 
		y=data1[:,1], 
		style="h0.75", 
		fill=rate_diff,
		cmap=cpt_rate)

	fig.colorbar(frame=f'af+l"Difference in log((N(m)>{mmin}))"', cmap=cpt_rate)    

	out = os.path.join(fig_folder, plot_fname+'_rate_diff.png')
	fig.savefig(out)
