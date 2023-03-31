Strong-Motion Tools (SMT) module
################################

The :index:`Strong-Motion Tools (SMT)` module contains code for the selection of ground-motion prediction equations (GMPEs) and the subsequent development of a ground-motion characterisation (GMC). 

The main components of the Strong-Motion Tools (SMT) comprise of (1) parsing capabilities to generate metadata (2) capabilities for computation and plotting of ground-motion residual distributions (3) comparison of potentially viable GMPEs and (4) development of the GMC with the final selection(s) of GMPEs.

Here, we will demonstrate how each of these components can be implemented, in the context of aiming to develop a GMPE logic-tree approach GMC for Albania. 

Performing a Residual Analysis within the SMT
*********************************************
The SMT provides capabilities (parsers) for the parsing of an inputted dataset into metadata for the performing of a residual analysis, so as to evaluate GMPE performance against the inputted dataset.

The inputted dataset usually comprises of a ground-motion record flatfile. Many seismological institutions provide flatfiles of processed ground-motion records. These flatfiles often slightly differ in format, but generally follow a template of a .csv file in which each row represents a single ground-motion record, that is, a recording of the observed ground-motion at a single station. Each record contains information for (1) the associated earthquake (e.g. moment magnitude, hypocentral location, focal depth), (2) the associated site parameters (e.g. shear-wave velocity in the upper 30m of a site (Vs30)), (3) source-to-site distance metrics (e.g. epicentral distance, Joyner-Boore distance) and (4) ground-motion intensity values for various intensity measures (e.g. peak-ground acceleration (PGA), peak-ground velocity (PGV), spectral acceleration (SA) for various spectral ordinates).  

Within a residual analysis, the information provided in each ground-motion record is used to evaluate how closely a selection of GMPEs predict the expected (observed) ground-motion. The ground-motion records within a flatfile will usually comprise of earthquakes from the same region and of the same tectonic region type. This is because, if for example, we are trying to identify the best performing GMPEs for Albania, we will only want to examine how well the considered GMPEs predict the (observed) ground-motion for earthquakes originating from Albania and potentially the surrounding (tectonically similar) regions if we need supplementary ground-motion records to improve the dataset's coverage with respect to magnitude, distance etc.
Parsers are provided in the SMT for the most widely used flatfile formats (e.g. ESM, NGAWest2).

In this example, we will consider the ESM 2018 format parser for the parsing of a ESM 2018 flatfile comprising of earthquakes from Albania and the surrounding regions. We will then evaluate appropriate GMPEs using the parsed metadata in the explanations of the subsequent SMT components.
   
Parsing the metadata
====================

Herein we provide a brief description of the various steps for the parsing of an ESM 2018 flatfile. Note that we use the symbol ``>`` as the prompt in a terminal, hence every time you find some code starting with this symbol this indicate a command you must type in your terminal. 

Following the geographical filtering of the ESM 2018 flatfile for only earthquakes from Albania and the surrounding regions in this example, we can parse the flatfile using the ``ESM_flatfile_parser``.

1. First we must import the ``ESMFlatfileParser`` and the required python modules for managing the output directories:

    > # Import required python modules
    > import os
    > import shutil
    > from openquake.smt.parsers.esm_flatfile_parser import ESMFlatfileParser

2. Next we need to specify the base path, the flatfile location and the output location:

    > # Specify base path
    > DATA = os.path.abspath('')
    >
    > # Specify flatfile location
    > flatfile_directory = os.path.join(DATA,'ESM_flatfile_SA_geographically_filtered.csv')
    >
    > # Specify metadata output location
    > output_database = os.path.join(DATA,'metadata')
    >
    > # If the metadata already exists first remove
    > if os.path.exists(output_database):
    >     shutil.rmtree(output_database)

3. Now we can parse the metadata from the ESM 2018 flatfile using the ``ESMFlatfileParser`` with the autobuild class method:

    > # Specify metadata database ID and metadata database name:
    > DB_ID = '000'
    > DB_NAME = 'ESM18_Albania'
    >
    > # Parse flatfile
    > parser = ESMFlatfileParser.autobuild(DB_ID, DB_NAME, output_database, flatfile_directory)

4. The flatfile will now be parsed by the ``ESMFlatfileParser``, and a pickle (.pkl) file of the metadata will be outputted in the specified output location. We can now use this metadata to perform a GMPE residual analysis.

Specifying the inputs for the residual analysis
===============================================

Following the parsing of a flatfile into useable metadata, we can now specify the inputs for the performing of a residual analysis. Residual analysis compares the predicted and expected (i.e. observed) ground-motion for a combination of source, site and path parameters to evaluate the performance of GMPEs. Residuals are computed using the mixed effects methodology of Abrahamson and Youngs (1992), in which the total residual is split into an inter-event component and an intra-event component. Abrahamson and Youngs (1992) should be consulted for a detailed overview of ground-motion residuals, but a brief overview of the total residual, inter-event residual and intra-event residual terms is provided here. 

The total residual (one per ground-motion record) is computed as follows:

    total_residual = (log(observed_ground_motion) - log(predicted_ground_motion))/GMPE_sigma
    
The closer the computed residual is to zero the better the fit between the predicted ground-motion and the observed ground-motion. Given that the ground-motion predicted by a GMPE is assumed to be lognormally distributed with mean of mu and a standard deviation of sigma, a residual of 1.0 or -1.0 is representative of a mismatch of +1/-1 sigma respectively.

The inter-event residual is representative of how effectively a GMPE models the event-specific components of abn observed ground-motion (i.e. the source characteristics e.g. stress-drop, near-source velocity). The inter-event is computed from the mean of the total residuals for a single earthquake. Therefore, there is a single inter-event residual per an event. 

The intra-event residual is representative of how effectively a GMPE models record-specific components of an observed ground-motion (i.e. site-amplification, path effects, basin response). The intra-event residual for each record is computed by subtracting the inter-event for the associated earthquake (which generated the ground-shaking recorded in the record) from the corresponding total residual.

Now that we have an elementary overview of the residual components, we can specify the inputs to perform a residual analysis within the SMT are specified as follows:
    
1. Specify the base path, the path to the metadata we parsed in the previous stage and an output folder:

    > # Specify absolute path
    > DATA = os.path.abspath('')
    >
    > # Specify metadata directory
    > metadata_directory = os.path.join(DATA,'metadata')
    >
    > # Specify output folder
    > run_folder = os.path.join(DATA,results_preliminary)
    
3. Specify the GMPEs we want to evaluate, and the intensity measures we want to evaluate each GMPE for.

   The GMPEs and intensity measures to compute residuals for can be specified in two ways. The first is simply to specify a ``gmpe_list`` and an ``imt_list`` within the command line:

    > # Specify GMPEs and intensity measures within command line
    > gmpe_list = ['AbrahamsonEtAl2014','AkkarEtAlRjb2014','AmeriEtAl2017Rjb','BindiEtAl2014Rjb','BooreEtAl2014','BooreEtAl2020','CauzziEtAl2014','CampbellBozorgnia2014','ChiouYoungs2014','HassaniAtkinson2020Asc','KaleEtAl2015Turkey','KothaEtAl2020regional','LanzanoEtAl2019_RJB_OMO','LanzanoEtAl2020_ref']
    > imt_list = ['PGA','SA(0.1)','SA(0.2)','SA(0.5)','SA(1.0)']
    
   The second way is within a .toml file with the format specified below. The .toml file method is required for specifying the inputs of GMPEs with user-specifiable input parameters e.g. region or logic tree branch parameters. Note that here the GMPEs listed in the .toml file are not necessarily appropriate for Albania, but have been selected to demonstrate how GMPEs with additional inputs can be specified within a .toml file:

.. code-block:: ini

    [models]

    [models.AbrahamsonGulerce2020SInter]
    region = "GLO"
    
    [models.AbrahamsonGulerce2020SInter]
    region = "CAS"
    
    [models.AbrahamsonGulerce2020SInterCascadia]
    
    [models.NGAEastGMPE]
    gmpe_table = 'NGAEast_FRANKEL_J15.hdf5'
        
    [imts]
    imt_list = ['PGA','SA(0.2)','SA(0.5)','SA(1.0']
    
The additional input parameters which are specifiable for certain GMPEs are available within their corresponding GSIM files (found in oq-engine\openquake\hazardlib\gsim).
    
Computation of the residuals and basic residual plots
=====================================================

1. Following specification of the GMPEs and intensity measures, we can now compute the residuals using the Residuals module.

   We first need to get the metadata from the parsed pickle file (stored within the metadata folder):
   
   > # Import required python modules
   > import pickle
   > import openquake.smt.residuals.gmpe_residuals as res
   > import openquake.smt.residuals.residual_plotter as rspl
   >   
   > # Create path to metadata file
   > metadata = os.path.join(metadata_directory,'metadatafile.pkl')
   >
   > # Load metadata
   > sm_database = pickle.load(open(metadata,"rb"))
   >
   > # If the output folder already exists delete, then create output folder
   > if os.path.exists(run_folder):
   >    shutil.rmtree(run_folder)
   > os.mkdir(run_folder)

   Now we compute the residuals using the specified GMPEs and intensity measures for the metadata we have parsed from the flatfile:
   
   For computing the residuals from a list of GMPEs and intensity measures specified in the command line:
   
   > # Compute residuals using GMPEs and intensity measures specified in command line
   > resid1 = res.Residuals(gmpe_list,imt_list)
   > resid1.get_residuals(sm_database)
   
   OR for computing the residuals from a list of GMPEs and intensity measures specified in a .toml file:
   
   > # Compute residuals using GMPEs and intensity measures specified in .toml file
   > filename = os.path.join(DATA,'gmpes_and_imts_to_test.toml') # path to .toml file
   > resid1 = res.Residuals.from_toml(filename)
   > resid1.get_residuals(sm_database)
   
   The residuals (here specified as 'resid1') is an object which stores (1) the observed ground-motions and associated metadata from the parsed flatfile, (2) the corresponding predicted ground-motion per GMPE and (3) the computed residual components per GMPE per intensity measure. The residuals object also stores the gmpe_list (e.g. resid1.gmpe_list) and the imt_list (resid1.imts) if these inputs are specified within a .toml file. 
   
2. Now we have computed the residuals, we can generate various basic plots describing the residual distribution.

   We can first generate plots of the probability density function plots (for total, inter- and intra-event residuals), which compare the computed residual distribution to a standard normal distribution:
   
   > # Plot residual probability density function for a specified GMPE from gmpe_list and intensity measure from imt_list
   > rspl.ResidualPlot(resid1, gmpe_list[0], imt_list[2], filename, filetype='jpeg') # Plot for gmpe in position 0 in gmpe_list and intensity measure in position 2 in imt_list
   >
   > # OR from .toml file (GMPEs and intensity measures in this case are stored in the residuals object created during computation of the residuals)
   > rspl.ResidualPlot(resid1, resid1.gmpe_list[0], resid1.imts[2], filename, filetype='jpeg') # Plot for gmpe in position 0 in resid1.gmpe_list and intensity measure in position 2 in resid1.imts
    
   These plots can be used to evaluate how closely the residuals follow the expected trend of a standard normal distribution (which would be observed if the GMPE exactly predicts the expected ground-motion for the considered intensity measure for each record in the parsed metadata). Therefore, given that the residual distribution corresponding to perfect fit between a GMPE and the ground-motion records, a mean closer to zero is representative of a better fit than a mean further away from zero. Likewise, a standard deviation of 1 would be expected for a GMPE which fits exactly to the considered ground-motion records, and a standard deviation further away from 1 would be expected for a GMPE which fits less well to the considered ground-motion records.
      
   Note that the filename (position 3 argument in rspl.ResidualPlot) should specify the output directory and filename for the generated figure in each instance.
   
   We can also plot the probability density functions over all considered spectral periods at once, so as to better examine how the residual distributions vary per GMPE over each spectral period:
   > # Plot residual probability density functions over spectral periods:
   > rspl.PlotResidualPDFWithSpectralPeriod(resid1, filename)
   >
   > # Generate .csv of residual probability density function per imt per GMPE 
   > rspl.PDFTable(resid1, filename)

   Plots for residual trends (again for total, inter- and intra-event components) with respect to the most important GMPE inputs can also be generated in a similar manner. Here we will demonstrate for magnitude:
   
   > # Plot residuals w.r.t. magnitude from gmpe_list and imt_list
   > rspl.ResidualWithMagnitude(resid1, gmpe_list[0], imt_list[2], filename, filetype='jpeg'), filetype='jpg')
   >
   > # OR plot residuals w.r.t. magnitude from .toml file
   > rspl.ResidualWithMagnitude(resid1, resid1.gmpe_list[0], resid1.imts[2], filename, filetype='jpeg'), filetype='jpg')

   The functions for plotting of residuals w.r.t. distance, focal depth and Vs30 are called in a similar manner:
   
   > # From gmpe_list and imt_list:
   > rspl.ResidualWithDistance(resid1, gmpe_list[0], imt_list[2], filename, filetype='jpeg')
   > rspl.ResidualWithDepth(resid1, gmpe_list[0], imt_list[2],  filename, filetype='jpeg')
   > rspl.ResidualWithVs30(resid1, gmpe_list[0], imt_list[2],  filename, filetype='jpeg')
   >
   > # OR from .toml:
   > rspl.ResidualWithDistance(resid1, resid1.gmpe_list[0], resid1.imts[2], filename, filetype='jpeg')
   > rspl.ResidualWithDepth(resid1, resid1.gmpe_list[0], resid1.imts[2], filename, filetype='jpeg')
   > rspl.ResidualWithVs30(resid1, resid1.gmpe_list[0], resid1.imts[2], filename, filetype='jpeg')
                   
GMPE performance ranking methodologies
======================================

The SMT contains implementations of several published GMPE ranking methodologies, which allow additional inferences to be drawn from the computed residual distributions. Brief summaries of each ranking metric are provided here, but the corresponding publications should be consulted for more information.

1. Likelihood Plots (Scherbaum et al. 2004)

   The Likelihood method is used to assess the overall goodness of fit for a model (GMPE) to the dataset (observed) ground-motions. This method considers the probability that the absolute value of a random sample from a normalised residual distribution falls into the interval between the modulus of a particular observation and infinity. The likelihood value should equal 1 for an observation of 0 (i.e. the mean of the normalised residual distribution) and should approach zero for observations further away from the mean. Consequently, if the GMPE exactly matches the observed ground-motions, then the likelihood of a particular observation should be distributed evenly between 0 and 1, with a median value of 0.5
   
   Histograms of the likelihood values per GMPE per intensity measure can be plotted as follows:
   
   > # From gmpe_list and imt_list:
   > rspl.LikelihoodPlot(resid1, gmpe_list[0], imt_list[2], filename, filetype='jpeg')
   >
   > # OR from .toml:
   > rspl.LikelihoodPlot(resid1, resid1.gmpe_list[0], resid1.imts[2], filename, filetype='jpeg')

2. Loglikelihood Plots (Scherbaum et al. 2009)

   The loglikelihood method is used to assess information loss between GMPEs compared to the unknown "true" model. The comparison of information loss per GMPE compared to this true model is represented by the corresponding ground-motion residuals. A GMPE with a lower LLH value provides a better fit to the observed ground-motions (less information loss occurs when using the GMPE). It should be noted that LLH is a comparative measure (i.e. the LLH values have no physical meaning), and therefore LLH is only of use to evaluate two or more GMPEs.

   LLH values per GMPE aggregated over all considered intensity measures (i.e. those residuals are computed for as specified within either imt_list or the .toml file), LLH-based model weights and LLH per intensity measure can be computed as follows:

   > # From gmpe_list and imt_list
   > llh, model_weights, model_weights_with_imt = res.get_loglikelihood_values(resid1, imt_list)
   >
   > # OR from .toml:
   > llh, model_weights, model_weights_with_imt = res.get_loglikelihood_values(resid1, resid1.imts)
   >
   > # Generate a .csv table of LLH values
   > rspl.loglikelihood_table(resid1, filename)
   >
   > # Generate a .csv table of LLH-based model weights
   > rspl.llh_weights_table(resid1, filename)   
   
   Note that GMPE model weights should only be computed from a residual object created using a GMPE list (or .toml file) of only the candidate GMPEs for a GMPE logic tree (to ensure model weights are only distributed amongst the final selection of GMPEs).
   
   We can also plot LLH versus spectral period as follows:
   
   > # Plot LLH vs imt
   > rspl.plot_loglikelihood_with_spectral_period(resid1, filename)

3. Euclidean distance based ranking (Kale and Akkar, 2013)

   The Euclidean distance based ranking (EDR) method considers the probability that the absolute difference between an observed ground-motion and a predicted ground-motion is less than a specific estimate, and is repeated over a discrete set of such estimates (one set per observed ground-motion per GMPE per the specified intensity measure). The total occurrence probability for such a set is the modified Euclidean distance (MDE). The corresponding EDR value is computed by summing the MDE (one per observation), normalising by the number of observations and then introducing an additional parameter (Kappa) to penalise models displaying a larger predictive bias (here kappa is equal to the ratio of the Euclidean distance between obs. and pred. median ground-motion to the Euclidean distance between the obs. and pred. median ground-motion corrected by a predictive model derived from a linear regression of the observed data - the parameter kappa^0.5 therefore provides the performance of the median prediction per GMPE).

   EDR score, the normal distribution of modified Euclidean distance (MDE Norm) and k^0.5 (k is used henceforth to represent the median predicted ground-motion correction factor "Kappa" within the original methodology) per GMPE aggregated over all considered intensity measures can be computed as follows:
   
   > # Get EDR, MDE Norm and MDE per GMPE aggregated over all imts
   > res.get_edr_values(resid1)
   
   These same metrics can be computed per considered intensity measure also:
   
   > # Get EDR, MDE Norm and MDE for each considered imt
   > res.get_edr_values_wrt_spectral_period(resid1)
   
   EDR metrics per GMPE aggregated over all considered intensity measures, and per intensity measure, can be outputted together in a .csv as follows:
   
   > # Generate a .csv table of EDR values for each GMPE
   > rspl.edr_table(resid1,filename=EDR_table_output)
   
   As per LLH, model-weights can also be computed by normalising EDR. 
   
   > # Generate a .csv table of LLH-based model weights
   > rspl.edr_weights_table(resid1, filename)   

   And we can also plot EDR, MDE Norm and k^0.5 versus spectral period using:
   
   > # Plot EDR score vs imt
   > rspl.plot_plot_edr_metrics_with_spectral_period(resid1,filename)

Comparing GMPEs
===============

Alongside the SMT's capabilities for evaluating GMPEs in terms of residuals (within the residual module as demonstrated above), we can also evaluate GMPEs with respect to the predicted ground-motion for a given earthquake scenario. Such evaluations are useful in general, but especially so when the user has selected a shortlist of potentially viable GMPEs for a GMPE logic tree and wishes to further compare them, or wishes to examine how different scalings of a backbone GMPE affect the predicted ground-motion. The tools for comparing GMPEs are found within the Comparison module:  

   > # Import GMPE comparison tools
   > from openquake.smt.comparison import compare_gmpes as comp
   
The GMPE comparison tools include Sammon's maps, heirarchical clustering and matrix plots of Euclidean distance for both median and 84th percentile of predicted ground-motion per GMPE per intensity measure. Plotting capabilities for response spectra, GMPE sigma with respect to spectral period and trellis plots are also provided in this module. The inputs for these comparitive tools must be specified within a single .toml file with the following format:

.. code-block:: ini

    ### Input file for comparison of GMPEs using plotting functions in openquake.smt.comparison.compare_gmpes
    
    [general]
    imt_list = ['PGA','SA(0.1)','SA(0.5)','SA(1.0)','SA(2.0)']
    max_period = 2 # max period for response spectra
    maxR = 300 # max dist. used in trellis, Sammon's, clusters and matrix plots
    dist_list = [10, 100, 250] # distance intervals for use in spectra plots
    region = 0 # for NGAWest2 GMPE regionalisation
    eshm20_region = 4 # for KothaEtAl2020 ESHM20 GMPE regionalisation
    Nstd = 1 # num. of std. dev. to sample sigma for in median prediction (0, 1, 2 or 3)
    custom_colors_flag = 'False' #(set to "True" for custom colours in plots)
    custom_colors_list = ['lime','dodgerblue','gold','0.8']
    
    # Specify site properties
    [site_properties]
    vs30 = 800
    Z1 = -999
    Z25 = -999
    
    # Characterise earthquake for the region
    [source_properties]
    strike = -999
    dip =  60 # (Albania has predominantly reverse faulting)
    rake = 90 # (+ 90 for compression, -90 for extension)
    trellis_mag_list = [5,6,7] # mags used only for trellis
    trellis_depths = [20,20,20] # depth per magnitude
    
    # Specify magnitude array for Sammons, Euclidean dist and clustering
    [mag_values_non_trellis_functions]
    mmin = 5
    mmax = 7
    spacing = 0.1
    non_trellis_depths = [[5,20],[6,20],[7,20]] # [[mag,depth],[mag,depth],[mag,depth]] 
    
    # Specify label for gmpes
    [gmpe_labels]
    gmpes_label = ['B20','L19','BO14','BI14','C14','K20']
    
    # Specify gmpes
    [models] 
    [models.BooreEtAl2020]
    [models.LanzanoEtAl2019_RJB_OMO]
    [models.BooreEtAl2014]
    
    # Selected Kotha et al. (2020) GMPE logic tree branches
    [models.1-KothaEtAl2020ESHM20]
        sigma_mu_epsilon = 2.85697 
        c3_epsilon = 1.72    
    [models.2-KothaEtAl2020ESHM20]     
        sigma_mu_epsilon = 1.35563
        c3_epsilon = 0
    [models.3-KothaEtAl2020ESHM20]     
        sigma_mu_epsilon = 0
        c3_epsilon = 0        
    [models.4-KothaEtAl2020ESHM20]
        sigma_mu_epsilon = -1.35563
        c3_epsilon = 0 
    [models.5-KothaEtAl2020ESHM20]
        sigma_mu_epsilon = -2.85697 
        c3_epsilon = -1.72    
    
In the above .toml file we have specified the source parameters for earthquakes characteristic of Albania (compressional thrust faulting with magnitudes of interest in the range of Mw 5 to Mw 7), and we have specified a selection of GMPEs which may best capture the epistemic uncertainty associated with predicting the ground-shaking from earthquakes in/near Albania if implemented in a GMPE logic tree. Here, we are selecting 3 ergodic (fixed sigma per return period) GMPEs, and 5 scalings of the non-ergodic European Seismic Hazard Model 2020 (ESHM20) version Kotha et al. (2020) GMPE (see Weatherill et al. 2020 for more details on the ESHM20 version of Kotha et al. 2020). The ESHM20 version of Kotha et al. (2020) has been set to a regionalisation parameter of 2 in "general" params, which is representative of central region (regular) anelastic attenuation. 

Once we have defined our inputs for GMPE comparison, we can use each tool within the Comparison module to evaluate how similar the GMPEs predict ground-motion for a given ground-shaking scenario. We must first create the "Configuration" object which stores the information specified within the .toml file for use in the plotting functions:

    > # Generate config object (filename = path to input .toml file)
    > config = comp.Configurations(filename)

Once we have created the Configuration object we can use the plotting functions available within the Comparison module.

1. Trellis Plots 

   We can generate trellis plots (predicted ground-motion by each considered GMPE versus distance) for different magnitudes and intensity measures (specified in the .toml file) as follows: 
   
   > # Generate trellis plots
   > comp.plot_trellis(config, output_directory)
   
2. Spectra Plots

   We can plot response spectra and GMPE sigma spectra (sigma versues spectral period) as follows: 
   
   > # Generate spectra plots
   > comp.plot_spectra(config, output_directory)
   
3. Sammon's Maps

   We can plot Sammon's Maps to examine how similar the median (and 84th percentile) of predicted ground-motion is by each GMPE for the ground-shaking scenario specified within the .toml file (see Sammon, 1969 and Scherbaum et al. 2010 for more details on the Sammon's mapping procedure):
   
   > # Generate Sammon's Maps
   > comp.plot_sammons(config, output_directory)   
   
   A larger distance between two plotted GMPEs represents a greater difference in the predicted ground-motion. Therefore, if two or more GMPEs have a small distance between each other relative to the other GMPEs plotted, then only one of these adjacent GMPEs should be retained in the final GMPE logic tree (similarly predicting GMPEs minimises the epistemic uncertainty captured in the logic tree). It should be noted that: (1) more than one 2D configuration can exist for a given set of GMPEs and (2) that the absolute numbers on the axes do not have a physical meaning.
   
4. Heirarchical Clustering

   Dendrograms can be plotted as an alternative tool to evaluate how similarly the predicted ground-motion is by each GMPE:
   
   > # Generate dendrograms
   > comp.plot_cluster(config, output_directory)
   
   Within these plots the GMPEs are clustered hierarchically (i.e. the GMPEs which are clustered together at shorter Euclidean distances are more similar than those clustered together at larger Euclidean distances).
   
5. Matrix Plots of Euclidean Distance

   In addition to Sammon's Maps and heirarchical clustering, we can also plot the Euclidean distance between the predicted ground-motions by each GMPE in a matrix plot:
   
   > # Generate matrix plots of Euclidean distance
   > comp.plot_euclidean(config, output_directory)
   
   Within the matrix plots, the darker cells represent a smaller Euclidean distance (and therefore greater similarity) between each GMPE for the given intensity measure.   

References
==========

Abrahamson, N. A. and R. R. Youngs (1992). “A Stable Algorithm for Regression Analysis Using the Random Effects Model”. In: Bulletin of the Seismological Society of America 82(1), pages 505 – 510.

Kale, O and S. Akkar (2013). “A New Procedure for Selecting and Ranking Ground-Motion Prediction Equations (GMPES): The Euclidean Distance-Based Ranking (EDR) Method”. In: Bulletin of the Seismological Society of America 103(2A), pages 1069 – 1084.

Kotha, S. -R., G. Weatherill, and F. Cotton (2020). "A Regionally Adaptable Ground-Motion Model for Shallow Crustal Earthquakes in Europe." In: Bulletin  of Earthquake Engineering 18, pages 4091 – 4125.

Sammon, J. W. (1969). "A Nonlinear Mapping for Data Structure Analysis." In: IEEE Transactions on Computers C-18 (no. 5), pages 401 - 409.

Scherbaum, F., F. Cotton, and P. Smit (2004). “On the Use of Response Spectral-Reference Data for the Selection and Ranking of Ground Motion Models for Seismic Hazard Analysis in Regions of Moderate Seismicity: The Case of Rock Motion”. In: Bulletin of the Seismological Society of America 94(6), pages 2164 – 2184.

Scherbaum, F., E. Delavaud, and C. Riggelsen (2009). “Model Selection in Seismic Hazard Analysis: An Information-Theoretic Perspective”. In: Bulletin of the Seismological Society of America 99(6), pages 3234 – 3247.

Scherbaum, F., N. M., Kuehn, M. Ohrnberger and A. Koehler (2010). "Exploring the proximity of ground-motion models using high-dimensional visualization techniques." In: Earthquake Spectra 26(4), pages 1117 – 1138.

Weatherill G., S. -R. Kotha and F. Cotton. (2020). "A Regionally Adaptable  “Scaled Backbone” Ground Motion Logic Tree for Shallow Seismicity in  Europe: Application to the 2020 European Seismic Hazard Model." In: Bulletin of Earthquake Engineering 18, pages 5087 – 5117.