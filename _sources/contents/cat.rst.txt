CAtalogue Toolkit (cat) module
##############################

The :index:`Catalogue Toolkit` module provides functionalities for the compilation of a homogenised catalogue starting from a collection of catalogues with different origins and magnitudes.

The formats of the original catalogues supported are:

- ISF (see http://www.isc.ac.uk/standards/isf/)
- GEM Hazard Modeller's Tookit .csv format
- GCMT .ndk formats (see https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained)

The module contains tools to transform between these different catalogue types, retaining the most neccessary information. The easiest way to build a homogenised catalogue within this framework is to run a bash script which includes the required inputs for each stage of the model and to specify the parameters with a toml file. We demonstrate below how to set this up, but individual steps can also be called directly in python if preffered. 

Setting up a bash script
========================

The bash script specifies all file locations and steps for generating a homogenised model. At each step, we provide a different .toml file specifying the necessary parameters. If you have all the neccessary files set out as below (and named run_all.sh) you should have no problems in running the script with ./run_all.sh

Further details on each step follow.

.. code-block:: ini

	#!/usr/bin/env bash

	CASE="homogenisedcat"

	# Merging catalogues
	ARG1=./settings/merge_$CASE.toml
	oqm cat merge $ARG1

	# Creating the homogenised catalogue 
	ARG1=./settings/homogenise_$CASE.toml
	ARG2=./h5/$CASE_otab.h5
	ARG3=./h5/$CASE_mtab.h5

	oqm cat homogenise $ARG1 $ARG2 $ARG3

	# Checking the homogenised catalogue 
	ARG1=./settings/check_$CASE.toml
	ARG2=./h5/$CASE_homogenised.h5

	oqm cat check_duplicates $ARG1 $ARG2

	# Create .csv
	ARG3=./csv/catalogue_$CASE.csv
	oqm cat create_csv $ARG2 $ARG3


Merging
=======

The first step in compiling a catalogue is merging information from different sources. This might include a global catalogue (e.g. ISC-GEM or GCMT), and various local catalogues that are more likely to have recorded smaller magnitude events, or contain more accurate locations. The merge tools are designed to allow multiple catalogues to be combined into one, regardless of original catalogue formats, and to retain only unique events across the catalogues. 

As we see in the bash script above, we run the merge with :code:`oqm cat merge merge.toml` where merge.toml contains all the necessary information for the merge. The :code:`merge` function takes the toml file as its single argument. An example of merge .toml file might look like this: 
 
.. code-block:: ini

	[general]
	## Set these or your output files will have bad names and be in very confusing places!
	output_path = "./../h5/"
	output_prefix = "homogenisedcat_"

	[[catalogues]]
	code = "ISCGEM"
	name = "ISC GEM Version 10.0"
	filename = "./iscgem10pt0.csv"
	type = "csv"

	[[catalogues]]
	code = "local"
	name = "local version 0.0"
	filename = "./local_00_cat.csv"
	type = "csv"
	delta_ll = 30
	delta_t =  10
	buff_ll = 0.0
	buff_t = 5.0
	use_kms = true
	#use_ids = true

This contains some general settings for the output, namely the path where the output should be saved and a prefix that will be used to name the file. If you are running the merge function as part of a homogenisation bash script, it is strongly recommended to make this consistent with the CASE argument (as in the example)! The toml file should also be named merge_$CASE. A minimumn magnitude can also be specified here, which will filter the catalogue to events above the specified minimum, and a polygon describing a geographic area of interest can also be added to filter the catalogue to that region.
The rest of the merge toml should contain the details of the catalogues to be merged. For each catalogue, it is necessary to specify a code, name, file location and catalogue type. The code and name are for the user to choose, but the code should be short as it will feature in the final catalogue to indicate which catalogue the event came from. The type argument will be used to process the catalogue, so should be one of "csv", "isf" or "gcmt".

To ensure events are not duplicated, the user can specify space-time windows over which events are considered to be the same. These are specified using :code:`delta_t` for time and :code:`delta_ll` for distance, where :code:`delta_ll` can be specified in degrees or kms by specifying :code:`use_km = True`. For both parameters, these can be specified as a single value, as a year-value pair to allow for changes in location/temporal accuracy in different time periods, or as a function of magnitude m, which is particularly useful when using the GCMT catalogue, which has some significant differences in location/time compared to other catalogues due to the moment tensor inversion considering these as model parameters. This can result in significant differences for large events, some of which may be so large that they are better removed manually (for example, the 3.5 minute time difference between ISC_GEM and GCMT for the 2004 Sumatra-Andaman earthquake). For the window parameters, we can also specify a buffer (:code:`buff_ll` or :code:`buff_t`) which highlights events which fall within some space/time of the window parameter and flags these as potential duplicates. The units for :code:`buff_ll` should be consistent with those used in :code:`delta_ll` and specified using the :code:`use_kms` argument (i.e. set use_kms = True to use km units or use_kms = False to use lat/lon). In the case where catalogues to be merged might come from the same source or otherwise have matching event ids, the :code:`use_ids` argument will remove duplicated event ids directly. 

The output of the :code:`merge` function will be two h5 files specifying information on the origin :code:`_otab.h5` and the magnitudes :code:`_mtab.h5`. The origin file will contain the event locations, depths, agency information and focal mechanism parameters where available, while the magnitudes file will include information on the event magnitude and uncertainties.

Homogenisation
==============

The next step in creating a catalogue is the homogenisation of magnitudes to moment magnitude M_w. The catalogue toolkit provides different tools to help with this. Homogenising magnitudes is normally done by using a regression to map from one magnitude to a desired magnitude. This requires that an event would need to be recorded in both magnitudes, and ideally a good number of matching events to ensure a significant result. In the toolkit, we use odr regression with scipy to find the best fit model, with options to fit a simple linear regression, an exponential regression, a polynomial regression, or a bilinear regression with a fixed point of change in slope. The function outputs parameters for the chosen fit, plus uncertainty that should be passed on to the next stage.

.. code-block:: ini

	from openquake.cat.catalogue_query_tools import CatalogueRegressor
	from openquake.cat.hmg.hmg import get_mag_selection_condition
	import pandas as pd
	import numpy as np
        
        def build_magnitude_query(mag_agencies, logic_connector):
    	"""
    	Creates a string for querying a DataFrame with magnitude data.
        
    	:param mag_agency:
        	A dictionary with magnitude type as key and a list of magnitude agencies as values
    	:param logic_connector"
        	A string.  Can be either "and"  or "or"
    	:return:
        	A string defining a query for an instance of :class:`pandas.DataFrame`
    	"""
    	    query = ""
    	    i = 0
    	    for mag_type in mag_agencies:
        	logic = "\" if logic_connector == 'or' else "&"
        	for agency in mag_agencies[mag_type]:
        	    cnd = get_mag_selection_condition(agency, mag_type, df_name="mdf")
        	    query += " {:s} ({:s})".format(logic, cnd) if i > 0 else "({:s})".format(cnd)
        	    i += 1
    	    return query


	def get_data(res):
    	"""
    	From a DataFrame obtained by merging two magnitude DataFrames it creates the input needed 
    	for performing orthogonal regression.
        
    	:param res:
        :class:`pandas.DataFrame`
    	"""
    	    data = np.zeros((len(res), 4))
    	    data[:, 0] = res["value_x"].values
            data[:, 1] = res["sigma_x"].values
    	    data[:, 2] = res["value_y"].values
    	    data[:, 3] = res["sigma_y"].values
    	    return data
        
	def getd(mdf, agenciesA, agenciesB):
        	queryA = build_magnitude_query(agenciesA, "or")
    		queryB = build_magnitude_query(agenciesB, "or")
        
    		selA = mdf.loc[eval(queryA), :]
    		selB = mdf.loc[eval(queryB), :]
        
    		res = selA.merge(selB, on=["eventID"], how="inner")
    		print("Number of values: {:d}".format(len(res)))
         
    		data = get_data(res)
    		return data
        
	def print_mbt_conversion(results, agency, magtype, **kwargs):
    		print("\n")
    		print("[magnitude.{:s}.{:s}]".format(agency, magtype))
    		print("# This is an ad-hoc conversion equation")
        
    		if "corner" in kwargs:
        		print("low_mags = [0.0, {:.1f}]".format(float(kwargs["corner"])))
        		fmt = "conv_eqs = [\"{:.4f} + {:.4f} * m\"]"
         		print(fmt.format(results.beta[0], results.beta[1]))
    		else:
        		print("low_mags = [0.0]")
        		fmt = "conv_eqs = [\"{:.4f} + {:.4f} * m\"]"
       			print(fmt.format(results.beta[0], results.beta[1]))
    	
    		fmt = "std_devs = [{:.4f}, {:.4f}]"
    		print(fmt.format(results.sd_beta[0], results.sd_beta[1]))
    		print("\n")

Using the above functions, we can query our catalogues to identify events that are present in both catalogues in both magnitude types. We can then use these to build a regression model and identify a relationship between different magnitude types. In the example below, we select mw magnitudes from our `local` catalogue and Mw magnitudes from `ISCGEM`. We specify a polynomial fit to the data, with starting parameter estimates for the regression of 1.2 and 0.7

.. code-block:: ini 

	agency = "local"
	magtype = "mw"
	amA = {magtype: [agency]}
	amB = {"Mw": ["ISCGEM"]}
	datambi = getd(gm, amA, amB)

	regress = CatalogueRegressor.from_array(datambi, keys="({:s}, {:s}) | (Mw)".format(agency, magtype))
	# Regression type to fit and starting parameters
	results = regress.run_regression("polynomial", [1.2, 0.7])
	# Results
        # Print resulting best fit
	print_mbt_conversion(results, agency, magtype)
	# plot the regression 
	regress.plot_model_density(overlay=False, sample=0)
	
Alternatively, if we wanted an example with a bilinear fit with a break in slope at M5.8, we could say

.. code-block:: ini

	results = regress.run_regression("2segmentM5.8", [0.3, 1.0, 4.5])

This would give us a different fit to our data and a different equation to supply to the homogenisation toml.

Where there are not enough events to allow for a direct regression or we are unhappy with the fit for our data, there are many conversions in the literature which may be useful. This process may take some revising and iterating - it is sometimes very difficult to identify a best fit, especially where we have few datapoints or highly uncertain data. Once we are happy with the fits to our data, we can add the regression equation to the homogenisation .toml file. This process should be repeated for every magnitude we wish to convert to Mw. 

The final homogenisation step itself is also controlled by a toml file, where each observed magnitude is specified individually and the regression coefficients and uncertainty are included. It is also necessary to specify a hierarchy of catalogues so that a preferred catalogue is used for the magnitude where the event has multiple entries. In the example below, we merge the ISCGEM and a local catalogue, preferring ISCGEM magnitudes where available as specified in the ranking. Because the ISCGEM already provides magnitudes in Mw, we simply retain all Mw magnitudes from ISCGEM. In this example, our local catalogue has two different magnitude types for which we have derived a regression. We specify how to convert to the standardised Mw from the local.mw and the standard deviations, which are outputs of the fitting we carried out above. 

.. code-block:: ini

	# This file contains a set of rules for the selection of origins and
	# the homogenisation of magnitudes. Used for the construction of the global catalogue
	# This version uses ad-hoc conversion parameters for ms and mb magnitudes, and that all Mw magnitudes are consistent
	#
	# Origin selection
	#

	[origin]
	# Specify preferred origin when multiple are available.
	ranking = ["ISCGEM",  "local"]

	#
	# Magnitude-conversion: Mw
	#
	# These are magnitudes we are happy with: don't convert
	# Homogenise all catalogues to iscgem Mw
	[magnitude.ISCGEM.Mw]
	low_mags = [0.0]
	conv_eqs = ["m"]

	[magnitude.local.mw]
	low_mags = [0.0]
	conv_eqs = ["0.1079 + 0.9806 * m"]
	std_devs = [0.0063, 0.0011]


	[magnitude.local.mww]
	low_mags = [0.0]
	conv_eqs = ["0.1928 + 0.9757 * m"]
	std_devs = [0.0091, 0.0016]

The actual homogenisation step is carried out by calling
:code:`oqm cat homogenise $ARG1 $ARG2 $ARG3`
as in the bash script example, where $ARG1 is the homogenisation toml file and and $ARG2 and $ARG3 are the hdf5 file outputs from the merge step, describing the origins and magnitude information for the merged catalogue respectively.

Checking for duplicate events
=============================

A common issue when merging catalogues is that there are differences in earthquake metadata in different catalogues. To avoid creating a catalogue with duplicate events, we specify the time and space criteria in the merge stage, so that events that are very close in time and space will not be added to the catalogue.  
We can check how well we have achieved this by looking at events that are retained in the final catalogue but fall within a certain time and space window. We can use the :code:`check_duplicates` function to do this, which takes in a check.toml file and the homogenised catalogue h5 file. A :code:`check.toml` file might look like this:

.. code-block:: ini

	[general]
	delta_ll = 0.3
	delta_t = 10.0
	output_path = "./tmp/"

where delta_ll and dela_t specify the time and space windows (in seconds and degrees respctively) to test for duplicate events. Again, we can specify different time limits and write the limits as functions of magnitudes i.e.:

.. code-block :: ini

	[general]
	delta_ll = [['1899', '100*m']]
	delta_t = [['1899', '30*m']]
	output_path = "./tmp/"

The check_duplicates output is a geojson file that draws lines between events that meet the criteria in the check.toml file. Each line segment contains the details of the two events, including their original magnitudes, the agencies that the events are taken from and the time and spatial distance between the two events, so that a user can check if they are happy for these events to be retained or would prefer to iterate on the parameters.

The process of building a reliable homogenised catalogue is iterative: at any step we may identify changes that should be made to merge criteria or regression parameters. It is also important to look at the resulting frequency-magnitude distribution to idenitfy any obvious changes in slope, which may indicate that our regressions are not performing as well as we would like. 


