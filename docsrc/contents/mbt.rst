Model Building Toolkit (mbt) module
###################################

The :index:`Model Building Toolkit` module contains code for building a PSHA earthquake occurrence 
model. The main goals of this tools are to:

1. Streamline the process of building a PSHA earthquake occurrence model
2. Ensure that the process adopted to build the model is reproducible and 
   extendable.

Here you can find information on the available functions including their inputs and outputs.
The tools in the mbt cover many important aspects of model construction. Some other functions are instead
included in the wkf
For an example of how they might be used in a workflow, see the SSC workflow module.


Available mbt and wkf functions
********************************
The `mbt` functions exist as part of the original version of the mbtk, when it was smaller and relied solely on a sequence of jupyter notebooks. Many functions that were here have been improved and/or moved to other locations. The `wkf` functions add further tools for building hazard models in the SSC workflow. 


Catalogue tools
================

.. automodule:: openquake.wkf.catalogue  
   :members: extract, to_df, from_df, create_subcatalogues, get_dataframe, create_gcmt_files


Tectonic regionalisation tools  
==============================

These are used in conjunction with the `subduction tools <https://gemsciencetools.github.io/oq-mbtk/contents/sub.html>`_ for classifying events in subduction regions. 

.. autoclass:: openquake.mbt.tools.tr.set_subduction_earthquakes.SetSubductionEarthquakes
   :members: classify


MFD functions
=============

.. automodule:: openquake.wkf.compute_gr_params  
   :members: compute_a_value_from_density, get_exrs, get_agr, compute_a_value, get_weichert_confidence_intervals, subcatalogues_analysis, _weichert_plot, weichert_analysis


.. autofunction:: openquake.wkf.plot.completeness.completeness_plot


.. autofunction:: openquake.wkf.mfd.check_mfds


.. automodule:: openquake.wkf.plot_incremental_mfd  
   :members: plot_incremental_mfds, plot_GR_inc_fixedparams_completeness_imp


Declustering
============
Primarily, the declustering is handled in the oq-hmtk inside the OpenQuake engine, but this function in the mbtk
is most commonly used by the hazard team (for now).

.. autofunction:: openquake.mbt.tools.model_building.dclustering.decluster


Distributed seismicity tools
=============================

.. automodule:: openquake.wkf.distributed_seismicity
   :members: get_bounding_box, get_data, get_stacked_mfd, explode, remove_buffer_around_faults, from_list_ps_to_multipoint


.. autoclass:: openquake.mbt.tools.adaptive_smoothing.AdaptiveSmoothing   
   :members: run_adaptive_smooth, poiss_loglik, plot_smoothing, information_gain


Fault modeling tools
=======================

The fault modelling tools require a dictionary of inputs describing the fault. The functions use the following default if this is not provided: 

.. code-block:: python  

   defaults = {'name': 'unnamed',
               'b_value': 1.,
               'bin_width': 0.1,
               'm_min': 4.0,
               'm_max': None,
               'm_char': None,
               'm_cli': 6.0,
               'm_upper': 10.,
               'slip_class': 'mle',
               'aseismic_coefficient': 0.,
               'upper_seismogenic_depth': 0.,
               'lower_seismogenic_depth': 35.,
               'rupture_mesh_spacing': 2.,
               'rupture_aspect_ratio': 2.,
               'minimum_fault_length': 5.,
               'tectonic_region_type': 'Active Shallow Crust',
               'temporal_occurrence_model': hz.tom.PoissonTOM(1.0),
               'magnitude_scaling_relation': 'Leonard2014_Interplate',
               'width_scaling_relation': 'Leonard2014_Interplate',
               'subsurface_length': False,
               'rigidity': 32e9,
               'mfd_type': 'DoubleTruncatedGR'
            }



.. automodule:: openquake.mbt.tools.fault_modeler.fault_source_modeler  
   :members: read_config_file, build_fault_model

.. automodule:: openquake.mbt.tools.fault_modeler.fault_modeling_utils 
   :members: construct_sfs_dict, make_fault_source, get_scaling_rel, fetch_param_val, write_metadata, write_rupture_params, trace_from_coords, line_from_trace_coords, check_trace_coords_ordering, angle_difference, write_geom, get_rake, get_dip, fetch_slip_rates, get_net_slip_rate, net_slip_from_strike_slip_fault_geom, get_fault_length, get_fault_width, calc_fault_width_from_usd_lsd_dip, calc_fault_width_from_length, leonard_width_from_length, get_fault_area, get_m_max, calc_mfd_from_fault_params, calc_double_truncated_GR_mfd_from_fault_params, calc_youngs_coppersmith_mfd_from_fault_params



Other useful functions
======================
These are mostly used within the model-building workflow


.. autofunction:: openquake.wkf.compute_mmax_from_catalogues.compute_mmax


.. autofunction:: openquake.wkf.seismicity.hypocentral_depth.hypocentral_depth_analysis  


.. autofunction:: openquake.wkf.seismicity.nodal_plane.process_gcmt_datafames  
 

.. autofunction:: openquake.mbi.wkf.focal_mech_loc_plots.focal_mech_loc_plots  


.. autofunction:: openquake.wkf.seismicity.smoothing.create_smoothing_per_zone  


.. autofunction:: openquake.mbi.wkf.create_nrml_sources.write_as_multipoint_sources  










