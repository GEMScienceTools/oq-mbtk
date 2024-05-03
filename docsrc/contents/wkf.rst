SSC workflow (wkf) module
##########################

The :index:`workflow` utilises the tools in the model builder's toolkit to construct a seismic source model step-by-step. This allows us to create a source model in xml format from a seismic catalogue, a set of source polygons and a file specifying required model parameters. Using the workflow tools, we can easily prepare different versions of the source models, which makes sensitivity analysis easier and allows us to easily build logic tree branches. Here we show the steps required to build a distributed seismicity model with smoothed sources. In practice, the order of the steps is not strictly important so long as e.g. the completeness is performed before the frequency-magnitude distributions (FMDs) are calculated.

Some notes on setup
********************
Though most of the tools we use in model construction are in python, some steps are executed in `Julia <https://julialang.org/>`_ using the `PSHAModelBuilder <https://github.com/GEMScienceTools/PSHAModelBuilder>`_ tools. We use this for the boxcounting, Gaussian smoothing, and rate distribution steps, because these steps are particularly intensive and Julia makes the process more efficient. See `the PSHAModelBuilder Github <https://github.com/GEMScienceTools/PSHAModelBuilder>`_  for details on setup.

In general, the components of the workflow are designed so that they can be run directly in a terminal. In the examples below, we use a jupyter notebook and the python ``subprocess`` module to run the commands. To instead run from terminal, use the cmd output directly. You can see that most of these calls are to the wkf module specifically, but in many cases these functions are wrappers to other functions within the mbt or elsewhere in the mbtk. You can use ``oqm wkf -h`` to see the available functions within the wkf and ``oqm wkf <subcmd_name> --help`` to see the input parameters for each function.

If you are running in a jupyter notebook, we suggest setting up as below, using tools in the package ``os`` or ``pathlib`` to manage paths and specifying the locations of the wkf tools (and Julia when using Windows): 

.. code-block:: python

    import os
    import subprocess

    os.environ['USE_PYGEOS'] = '0'
    os.environ['NUMEXPR_MAX_THREADS'] = '8'

    # remember to change the path in these lines so they correspond to your computer!
    BIND = os.path.join('/Users', 'kjohnson', 'GEM','oq-mbtk', 'bin')
    BIND1 = os.path.join('/Users', 'kjohnson', 'GEM', 'oq-mbtk', 'openquake', 'bin')
    print(BIND)
    print(BIND1)

    # on windows, also add these lines
    #PATH = os.path.join('..', '..', 'AppData', 'Local', 'Programs', 'Julia-1.9.3', 'bin')
    #os.environ["PATH"] = os.environ["PATH"] + PATH

Workflow inputs
****************

The workflow starts from three inputs as outlined below:
	1. A homogenised earthquake catalogue in hmtk catalogue format. This can be a direct output of a catalogue prepared using the `catalogue toolkit <https://gemsciencetools.github.io/oq-mbtk/contents/cat.html>`_ or a catalogue (from a csv or ndk file) converted using the catalogue parsers in the hmtk. 
	2. Source polygons covering the area of interest. These should ideally be supplied as .geojson files. 
	3. A source parameter configuration file supplied as a .toml file. The toml file will set up paramaters for many steps of the workflow and be modified while running the code. The configuration toml is created by the modeller. Please note that for all the relative paths in the .toml file, the reference folder is the one where the .toml configuration file is located. An example is shown below. 

.. code-block:: ini   	

	name = "South America"
	
	mmin = 4.0
	bin_width = 0.1
	rupture_mesh_spacing = 2.5
	
	[smoothing]
	smoothing_method = "Gaussian"
	kernel_maximum_distance = 120.0
	kernel_smoothing = [ [ 0.8, 20.0,], [ 0.15, 40.0,], [ 0.05, 80.0,],]

        [completeness]
        num_steps = 0
	step = 8
	flexible = true
	years = [ 1920, 1940, 1960, 1970, 1990, 2000, 2010,]
	mags = [ 5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 8.0,]
	ref_mag = 4.5
	ref_upp_mag = 10.0
	bmin = 0.5
	bmax = 1.5
	optimization_criterion = "poisson"
        
	[default]
	name = "Default"
	tectonic_region_type = "Active Shallow Crust"
	completeness_table = [ [ 2000.0, 4.0,], [ 1980.0, 5.0,], [ 1900.0, 6.0,],]
	rupture_aspect_ratio = 1.0
	upper_seismogenic_depth = 0.0
	lower_seismogenic_depth = 35.0
	nodal_plane_distribution = [ [ 1.0, 180.0, 45.0, 90.0,],]
	hypocenter_distribution = [ [ 1.0, 15.0,],]
	agr_sig = 0.1
	bgr_sig = 0.5
	agr_sig_weichert = 0.1
	bgr_sig_weichert = 0.5
	mmax = 7.5

	[msr]
	"Active Shallow Crust" = "Leonard2014_Interplate"

	[sources.26]
	
	[sources.34]

	[sources.38]

The .toml file will be read by different functions at different stages of the workflow. In this example, a source model will consist of sources 26, 34 and 38 from the source polygons, and these are all active shallow crustal sources. If using the ``completeness_analysis`` function, sources will be added to the model after this step, but at least one named source will be required to start the analysis and if there are too few events in a source to establish magnitude of completeness (mc) and GR parameters these sources will be omitted, so best practice remains to specify the sources clearly in the toml. Source names or abbreviations can also be used here - it is not necessary to use only numeric source identifiers. Still, we recommend using a numbering scheme based on a standard format e.g. ASC001 (for source number 1 in active shallow crust), ASC002 and so on.

At various stages of the workflow, values will be added to the .toml file or modified as the model is constructed. 

To avoid losing track of the original model parameters, the 'check_toml' function will make a copy of the .toml file that is edited and used in the construction of the source zones, and retain the original input .toml file as provided. The ``check_toml`` file will also report if necessary inputs are missing, if parameters are included for different types of smoothing and the number of sources in the model.

.. code-block:: python  
  
    orig_config = "IND_full_config.toml"
    config = "IND_config_working_130224.toml"

    cmd = f"oqm wkf check_toml {orig_config} {config} \"{use}\""
    p = subprocess.run(cmd, shell=True)  # returns a CompletedProcess instance

	
Model set-up  
*************
To set-up the workflow, we start by specifying some necessary parameters we will need later. 

.. code-block:: python   

    # Set the resolution level for the h3 gridding
    h3_level = 5
    # Set max and min depths
    depth_max = 35
    depth_min = 0
    
    mmax_delta = 0.5
    generate_completeness_tables = True
   
    config = "config.toml"
           
For efficient handling of spatial datasets, we use the `h3 <https://h3geo.org/docs/>`_ package when smoothing the distributed seismicity and to create point sources. We set the resolution for these steps here for consistency. See `the h3 website <https://h3geo.org/docs/core-library/restable/>`_ for more details on h3 resolution.

We also set some depth limits for events to consider in the source model: in this case we are dealing with crustal earthquakes and so the limits for the depths of events are set to 0-35km. Note that some catalogues may contain negative depths if topography has been considered in the catalogue processing!

The parameter ``mmax_delta`` sets a fixed delta value to add to the observed largest event in the catalogue when considering suitable mmax per zone. If ``generate_completeness_tables`` is True, the code will process completeness for each zone. It is useful to be able to turn off this step where you are running the workflow multiple times as this step can be quite slow.

Finally we specify the location of the configuration toml file that contains further parameters for our models and will contain zone-based information to construct the source zones. 

Create sub-catalogues per zone
***********************************

In order to create models for individual zones, we need to partition the events in our catalogue over the source zones we wish to construct. To do this, we use the ``create_subcatalogues_per_zone`` function. This function takes the specified catalogue and the source polygons as input, and returns a new file for each zone containing events within the zone polygon. The input catalogue should be in the hmtk catalogue format and be suitably declustered. The outputs - individual catalogue csv files for each zone - are created in the specified folder. This function uses a simple point in polygon approach to allocate events to the relevant zone, with a modification for polygons that cross the international dateline.

.. code-block:: python  

    polygons = "./data/asrc/src22.geojson"
    subcatalogues_folder = "./model/asc/subcatalogues/"

    cmd = f"oqm wkf create_subcatalogues_per_zone {polygons} {cat} {subcatalogues_folder}"
    p = subprocess.run(cmd, shell=True)

Calculate and apply completeness 
*********************************
At this step, we wish to apply some completeness constraints. You may prefer to perform a completeness analysis separately, taking into account changes in expected completeness (for example, due to known changes in local recording stations or equipment). In this case, the identified completeness for each zone can be added to the .toml file before the other steps of the workflow are carried out. Alternatively, there are tools within the mbt for performing a completeness analysis.

The ``completeness_analysis`` tool takes in a set of possible years and magnitudes and tests all possible completeness windows from these sets for their respective fit to the best-fitting FMD given the specified windows. Different optimisation criteria are available for testing the goodness of fit of the different completeness windows, from a norm difference between observed rates and expected to a Poisson likelihood of observing events based on the window selection. As such there are two steps to the completeness analysis in the workflow: 
1. generating the initial completeness windows from the provided years and magnitudes in the config .toml [completeness] section using ``completeness_generate``; and
2. running the analysis for each subcatalogue with ``completeness_analysis``.

.. code-block:: python   
 
    completeness_param_folder = './completeness_windows/'
    cmd = f"oqm cat completeness_generate {config} {completeness_param_folder}"
    p = subprocess.run(cmd, shell=True)

    pattern = os.path.join(".", "model", "asc", "subcatalogues", "*.csv")
    folder_figs = "./zone_completeness_figs"
    folder_compl_results = "./zone_completeness"

    cmd = f"oqm cat completeness_analysis \"{pattern}\" {config} {folder_figs} {completeness_param_folder} {folder_compl_results}"
    p = subprocess.run(cmd, shell=True)
    
Running the above will generate the completeness windows to test from the years and magnitudes in the config and write them to files in the specified completeness_param_folder. Then, for each csv file in the subcatalogues folder, it will test the completeness windows for the catalogue, calculate the FMD parameters for the best fitting window and write these to the config along with the completeness windows, and plot the best-fitting model in a png stored in folder_figs. In some cases, the completeness_analysis may fail to return completeness windows for a zone. This may be because there are too few events in the catalogue once the completeness windows are applied or because the calculated b-value for all of the possible complete catalogues is outwith the range specified by bmin and bmax in the [completeness] section of the .toml file. In this case, completeness can be manually added to the source or, if nothing is specified for the source, the source will be assigned the [default] completeness_table in the config. 

Whether you have used the ``completeness_analysis`` or have manually specified completeness for each zone, you may wish to check plots of event-density in time with the chosen completeness. You can easily create plots of this for each zone using ``plot_completeness_data``:

.. code-block:: python  
  
    folder_figs = "./completeness_density"
    cmd = f"oqm wkf plot_completeness_data \"{pattern}\" {config} {folder_figs}"
    p = subprocess.run(cmd, shell = True)

Again this will create for each zone a plot of the event density in time based on the zone catalogue and the parameters in the toml file. For any zones without a specified completeness (i.e. where the completeness_analysis fails to return a result or where completeness has not been manually added), the default completeness specified in the [defaults] section of the .toml will be used. Note that the ``plot_completeness_data`` function will not modify the config.toml, unlike the ``completeness_analysis`` step.

Calculate  and set Gutenberg-Richter parameters
***************************************************
For each source polygon, we wish to calculate the Gutenberg-Richter a- and b-values that define the total rate expected in that source. 
The compute_gr_params function calculates these values. To easily do this for each source zone, we supply the 'pattern' of naming for the source zones (if we have not already done so) to the function ``compute_gr_params``, which calculates the Weichert a and b parameters using the supplied completeness in the config for each zone. 

.. code-block:: python  

    pattern = os.path.join(".", "model", "asc", "subcatalogues", "*.csv")
    cmd = f'oqm wkf compute_gr_params \"{pattern}\" {config} {folder_figs}'
    
This will write a- and b-values to the config for each zone, called agr_weichert and bgr_weichert respectively.
If using ``completeness_analysis``, we will have already returned the a- and b- values called agr_weichert and bgr_weichert so the ``compute_gr_parameters`` step is no longer neccessary. However in either case we wish to write the calculated values to the config as agr and bgr. First we must ensure that agr_sig and bgr_sig values are available, describing the uncertainty in a- and b-values. In this case we can set from the [defaults] section where we are missing these: 

.. code-block:: python   

    cmd = f'oqm wkf set_property_from_default {config} agr_sig_weichert'
    p = subprocess.run(cmd, shell=True)
    cmd = f'oqm wkf set_property_from_default {config} bgr_sig_weichert'
    p = subprocess.run(cmd, shell=True)

Which will update the config file to contain agr_sig_weichert and bgr_sig_weichert values. Then we can set the parameters with the ``set_gr_params`` function:

.. code-block:: python  
  
    cmd = f"oqm wkf set_gr_params {config} -u \"*\" -m \"weichert\""
    p = subprocess.run(cmd, shell=True)
    
This sets the GR parameters from the config. -u tells the function which zones to do this for, in this case we use * to specify we wish to do this for all zones. -m tells the function which bgr values to use - in this case weichert. 

In some cases, we may wish to change the b-value and find the appropriate a-value for the catalogue given this new b. To do this, we can use the compute_a_value function for a specific zone. In this example we set the b-value of zone 6 to 1.0:

.. code-block:: python  

    from openquake.wkf.compute_gr_params import compute_a_value

    compute_a_value("./subcatalogues/subcatalogue_zone_6.csv", bval = 1.0, fname_config= config,
                    folder_out = folder_out, folder_out_figs = folder_figs)
 
This will add the new b-value and the calculated a-value from the catalogue to the config as bgr_counting and agr_counting. Again, these can be set with ``set_gr_params``, which will update the bgr value for zone 6:

.. code-block:: python  

    cmd = f"oqm wkf set_gr_params {config} --use \"'6'\" -m \"counting\""
    p = subprocess.run(cmd, shell=True)


Estimate and set maximum magnitudes  
************************************

The simplest approach to defining a maximum magnitude is to find the largest recorded event in the catalogue for each zone. Again, we do this on a per-zone basis. The function compute_mmax_per_zone does this for us, taking in the zone polygons, the catalogue and the config file. When running this function, we attach the "obs" label to keep track of where this value is obtained from (i.e. from observed data).

.. code-block:: python  

    cmd = f"oqm wkf compute_mmax_per_zone {polygons} {cat} {config} \"obs\""
    p = subprocess.run(cmd, shell=True)

To allow for the (significant) possibility that the largest event is not recorded in the catalogue, we add a delta value (the 'mmax_delta' we specified earlier) to the maximum recorded magnitude. The next step writes the maximum values to our config file. We also set a minimum maximum magnitude (in this case 7.0) so that any zones with a maximum magnitude less than M7.0 are set to have a maximum magnitude of M7.0.

.. code-block:: python  

    cmd = f"oqm wkf set_mmax_plus_delta {config} {mmax_delta} 7.0"

Analyse and set hypocentral depth
*************************************
Hypocentral depths are also determined from our catalogue data. In this case, we specify depth bins for the events in the catalogue. The code below will create plots of the depth distribituion of events in each zone and save them to a specified output file. It will also write a depth distribution for the zone into our config file as the fraction of events in each bin, where a bin is described by its mean (so in the example below, bins are written into our config file as 5, 15, 27.5).
We have split the command into two lines for easier readability.

.. code-block:: python  

    depth_bins = "0.0,10.0,20.0,35.0"
    folder_figs = './model/figs/hypo_depth/'
    cmd = f"oqm wkf analysis_hypocentral_depth {subcatalogues_folder} --f {folder_figs}"
    cmd = f"{cmd} --depth-bins \"{depth_bins}\" -c {config}"

Model focal mechanism distribution
**************************************

Similarly our focal mechanism distribution is determined from the available catalogue. Here we can choose to either use the our existing catalogue or to use the gcmt catalogue, repeating the first few steps of breaking this into source zones. If we have focal mechanism data in our catalogue (i.e. strike, dip and rake values) then we can supply our existing catalogue here, though we should be careful to ensure that the column names are correct.

.. code-block:: python  

    pattern = os.path.join(gcmt_subcat_folder, "*.csv")
    folder_figs_gcmt = "./model/figs/focal_mech"
    cmd = f"oqm wkf analysis_nodal_plane \"{pattern}\" {folder_figs_gcmt}"

Running this code block will run the nodal plane analysis function for all files that match the specified pattern in the specified location and output figures of the nodal plane distribution to the folder_figs_gcmt folder. Rupture types are categorised according to the method of Kaverina et al. (1996).

In this case, we don't have a direct method to apply the focal mechanism distribution to our config file. This is because we often want to consider other local information when deciding on a focal mechanism distribution. Instead we review the plots from ``analysis_nodal_plane`` and add them to a different toml file we have named ``defaults``. For each source zone, we specify a nodal_plane distribution as a list of [weight, strike, dip, rake], for example:

.. code-block:: ini  

    [sources.26]
    nodal_plane_distribution = [[ 1.00, 180.0, 60.0, 90.0,]]


Running

.. code-block:: python  
    
    cmd = f"oqm wkf set_defaults {config} {defaults}"

will take the hypocentral distribution (and any other parameters from defaults) and apply it to our config file where information is missing.

Discretise model to h3 zones
******************************
Building a smoothed seismicity model can be particularly computationally intensive due to the spatial distribution we are trying to model. We use `h3 <https://h3geo.org/docs/>`_ to help with this, by covering our area of interest in hexagonal cells at a specified resolution (which we set earlier as h3_level). This step in the workflow generates the collection of h3 cells that covers our source polygons. The cell indices are written to the specified output repository, where they will be called in the next steps of the smoothing. 

.. code-block:: python  

    zones_h3_repr = './model/zones/h3/'
    cmd = f"oqm wkf set_h3_to_zones {h3_level} {polygons} {zones_h3_repr}"

If for some reason we don't want to generate h3 cells for all zones in a polygon set, we can specify the polygons we do want to use by supplying a list of polygon ids

Boxcounting (for smoothing)
******************************
For Gaussian smoothing approaches, and for calculating the information gain of a smoothing model, we need to know how many events occur in each spatial cell.
The ``wkf_boxcounting`` function requires the catalogue of earthquakes, the h3 mapping generated at the previous step and the config file. It will write the output - a dataframe containing locations of cells and the number of events in that cell - to the specified output folder. By default the function outputs a version with and without the h3 indices. 
Finally, we supply two extra paramters to the function directly. Firstly the end year is specified after the '-y' flag. Secondly, the weighting is provided using the -w flag. There are currently three options for this weighting:
* 'one' weights all earthquakes equally
* 'mfd' weights according to the rate of magnitudes based on the zonal MFD, so earthquakes occurring where the occurrence rates for the given magnitude are higher get weighted more.
* 'completeness' weights according to the inverse of the duration of completeness for that magnitude, so more weight is given to small earthquakes that weren't captured in the past.  


.. code-block:: python  
    
    fld_box_counting = os.path.join(".", "model", "boxcounting")
    tmp = os.path.join(BIND, "wkf_boxcounting_h3.jl")
    zones_h3_repr = os.path.join(zones_h3_repr, "mapping_h5.csv")
    cmd = f"julia {tmp} {cat} {zones_h3_repr} {config}"
    cmd = f"{cmd} {h3_level} {fld_box_counting} -y 2018 -w \"one\""
	

Apply smoothing
*****************
There are currently two options for smoothing included in the mbt. For either approach, the required parameters should be included in the toml file under the 'smoothing' section (see example above). In both cases, the output file is a smoothed rate in each h3 cell. Note that the rate returned by these functions comes from the events in the declustered catalogue. The next step will normalise these rates to be consistent with the rates from the FMD for each zone. 

Option 1: Gaussian smoothing kernels
=====================================

This approach applies Gaussian spatial kernels of fixed distance around each event in the catalogue. Multiple kernels and weightings can be specified. The ``kernel_smoothing`` in the config specifies the smoothing distances and their associated weights - in this case we apply three kernels with decreasing weight for increased smoothing distance. We also specify a ``kernel_maximum_distance`` as the upper limit on the Gaussian smoothing. The Gaussian smoothing approach takes the results of the boxcounting directly, so any specified weights in the previous step will be applied to the smoothing in this step. The boxcounting results file will be inside the boxcounting folder, and we set up a file to contain the smoothing results. 

.. code-block:: python  

    fname_bcounting = os.path.join(".", "model", "boxcounting", f"box_counting_h3_{cat}")
    fname_smoothing = os.path.join(".", "model", "smoothing", "smooth")
    tmp = os.path.join(BIND1, "wkf_smoothing.jl")
    cmd = f"julia {tmp} {fname_bcounting} {config} {fname_smoothing}"
    p = subprocess.run(cmd, shell=True)

Option 2: Helmstetter (2007) adaptive smoothing
================================================

This approach determines a smoothing distance for each event based on its proximity to other events. This means that the smoothing distance will be small in areas with many earthquakes and larger where there are fewer, further spaced events.
In this case, the parameters to be specified are a minimum smoothing distance (ideally close to the location uncertainty of a given catalogue), the nth neighbour to use for the smoothing distance (e.g. to use the distance to the 5th closest neighbour, we would specify n_v = 5) and the spatial kernel we want to use (either power-law or Gaussian), as well as a maximum smoothing distance (maxdist). Because the adaptive smoothing considers all events in the catalogue potential neighbours, including a ``maxdist`` is especially important for catalogues with sparse events covering large areas, but in practice we have found it does not impact the final smoothing results (either in terms of spatial pattern or information gain). These parameters should be specified in the [smoothing] part of the toml file. 

.. code-block:: python  

    h3_cells_loc = os.path.join(zones_h3_repr, "mapping_h5.csv")
    fname_smoothing = os.path.join(".", "model", "smoothing", "adapsmooth_nv5.csv")
    cmd = f"oqm wkf wkf_adaptive_smoothing {cat} {h3_cells_loc} {config} {fname_smoothing} "
    p = subprocess.run(cmd, shell=True)
    

In both cases, the output will be one large file containing the smoothing at all model locations. To split the smoothed results back into zones so that we can apply the correct rates, we use the following:

.. code-block:: python  

    fname_smoothing_source = './smoothing/adapn5_smooth'
    cmd = f"oqm wkf create_smoothing_per_zone {fname_smoothing} {polygons} {fname_smoothing_source} --use \"{use}\""
    p = subprocess.run(cmd, shell=True)

Specifying zone ids with ``use`` will return the smoothing only for the specified zones. The fname_smoothing_source input specifies the output folder in which to save the results. This will return for each source a csv of smoothed rates at the specified h3 locations.

Distribute rates in sources
*****************************
Now that we have determined a smoothing, we want to distribute the total earthquake rate for a source polygon in such a way that the rate is highest where the intensity of events is highest, that is we wish to distribute the total rate of events spatially. 

``eps_a`` and ``eps_b`` are epsilons to be applied to the sigma values from applying the weichert method. If set to zero, the ``agr`` and ``bgr`` are used, but if there is an epsilon and a reference magnitude (the a-value type sigma is for the rate above a reference magnitude), then the zonal mfd is adjusted accordingly before distributing the rates.

This will output point_src_input for each polygon.

.. code-block:: python  

    folder_point_srcs = os.path.join(".", "model", "point_src_input")
    tmp = os.path.join(BIND1, "wkf_rates_distribute.jl")
    cmd = f"julia {tmp} -r 0.0 -b 0.0 {fname_smoothing_source} {config} {folder_point_srcs}"


Write to xml
*************
Finally, we wish to write our crustal source models to .xml files that can be used in the OpenQuake engine. For this we use the ``create_nrml_sources`` function which takes the point sources we created for each zone in step 11 and other information from the config file to create source models in the specified folder. At this step, it is necessary to have specified several as-yet unused parameters in the config, such as the msr and the mmin, bin_width and rupture_mesh_spacing. 

.. code-block:: python  

    pattern = os.path.join(folder_point_srcs, "*.csv")
    folder_oq = os.path.join("./ssm")
    cmd = f"oqm wkf create_nrml_sources \"{pattern}\" {config} {folder_oq} -a"
    p = subprocess.run(cmd, shell=True)
