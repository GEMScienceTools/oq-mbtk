SUBduction (sub) module
#######################

The :index:`Subduction` module contains software for the construction of subduction earthquake sources for the *oq-engine*. The components of this model can be used either independently or within a workflow similarly to what is described in this section.

Defining the geometry of the top of the slab
********************************************

The modeling of earthquake subduction sources starts with the definition of the geometry of the slab. The mbtk subduction module contains tools for the definition of the top of the slab. Two are the approaches available. The first one, the most comprehensive, requires a tedious process of digititazion of the profiles describing the position of the top of the slab versus depth along each cross section (see `Pagani et al. (2020) <https://sp.lyellcollection.org/content/early/2020/03/02/SP501-2019-120.abstract>`__ for a description of the methodology). The second one uses the geometries of the slab proposed by `Hayes et al. (2018) <https://science.sciencemag.org/content/362/6410/58>`__ (`dataset <https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467>`__).

The result of these two procedures is a folder containing a set of .csv files each one describing a profile. In this context a profile is a curve that lays on top of the slab and, generally, has a direction parallel to the dip.

.. _first approach:
   
First approach
==============

Herein we provide a brief description of the various steps. Note that we use the symbol ``>`` as the prompt in a terminal, hence every time you find some code starting with this symbol this indicate a command you must type in your terminal. 

1. The first step entails the definition of a configuration file. An example is provided herein

.. code-block:: ini

    [data]

    # Path to the text file with the coordinates of the trench axis
    trench_axis_filename = /Users/kjohnson/GEM/Regions/paisl18/data/subduction/trenches/kerton_trench.xy

    # Path to the pickled file (an instance of the hazard modeller's toolkit Catalogue) 
    catalogue_pickle_filename = /Users/kjohnson/GEM/Regions/paisl18/data/catalogues/locations/PI_cat_filt.p

    # Path to the Slab 1.0 text file with the coordinates of the top of the slab
    slab1pt0_filename = /Users/kjohnson/GEM/Regions/paisl18/data/subduction/slab1pt0/ker_slab1.0_clip.xyz

    # Path to the Crust 1.0 text file (see)
    crust1pt0_filename = /Users/kjohnson/GEM/Regions/paisl18/data/crustal_models/crust1pt0/crsthk.xyz

    # Path to the Litho 1.0 text file (see)
    litho_filename = /Users/kjohnson/GEM/Regions/paisl18/data/crustal_models/litho1pt0/litho_moho.xyz

    # Path to the file containing the focal mechanisms from the Global Centroid Moment Tensor project
    gcmt_filename = /Users/kjohnson/GEM/Regions/paisl18/data/catalogues/focal_mechanisms/GCMT_20151231.ndk

    # Path to the file with volcanoes
    volc_filename = /Users/kjohnson/GEM/Regions/paisl18/data/volcanoes/volcano_list.xy

    # Path to the text topography file 
    topo_filename = /Users/kjohnson/GEM/Regions/paisl18/data/topography/GEBCO_2014/pacisl_topobath_nf.xyz

    [section]

    # Length of each profile [km]
    lenght = 700

    # Spacing [km] between the profiles along the axis subduction trench 
    # specified in the ariable `trench_axis_filename`
    interdistance = 100 

    # Azimuth parameter. When equal to a real number in the range [0, 360] all 
    # the profiles will follow that direction. Ortherwise, if `None` the 
    # profiles will have a direction perpendicular to the trench axis
    azimuth = None

    # Maximum depth of each profile [km]
    dep_max = 700


2. Create a pickled version of your hmtk formatted catalog::

    > pickle_catalogue.py ./catalogues/cac.cat`

3. Create a set of cross-sections from the subduction trench axis::

    > create_multiple_cross_sections.py ./ini/central_america.ini

Check the traces of the cross-sections in the map created. It's possible to edit the traces or add new traces in the file ``cs_traces.cs``

4. Check the new set of traces in a map with the command::

    > plot_multiple_cross_sections_map.py ./ini/central_america.ini cs_traces.cs

5. Create one .pdf file for each cross-section with the available information: e.g., earthquake hypocentres, focal mechanism, slab 1.0 geometry, CRUST 1.0 Moho::

    > plot_multiple_cross_sections.py cs_traces.cs

This command will produce as many ``.pdf`` files as the number of cross-sections specified in the ``.cs`` file

6. Digitize the contact between the overriding plate and the subducted plate in each cross-section. The information in the command below corresponds to the longitude and the latitude of the origin of the cross-section, the length [km], the azimuth [decimal degrees], the cross-section ID and the name of the ``.ini`` file. For example::

    plot_cross_section.py -106.479700 21.250800 600.000000 89.098531 0 ./ini/central_america.ini

Once launched, by clicking on the image it is possible to digitize a sequence of points. Once completed the digitization, the points can be saved to a file whose name corresponds to ``cs_<section ID>.csv`` by pressing the ``f`` key on the keyboard. The points can be deleted with the key ``d``.

.. _second approach:

Second approach
===============

The second approach proposed is simpler than the first one. At the beginning, it requires to complete point 1 and point 3 described in the `first approach`_ section. Once we have a configuration file and a set of cross sections ready we can complete the construction of the set of profiles with the following command::

    > sub_create_sections_from_slab.py <slab_geometry.csv> <output_folder> <file_with_traces.cs>

Where:

- ``<slab_geometry.csv>`` is the name of the file
- ``<output_folder>`` is the name of the folder where to write the profiles
- ``<file_with_traces.cs>`` is the name of the file (produced by ``create_multiple_cross_sections.py``) with information aboout the traces of the cross-sections.

Building the top of the slab geometry
*************************************

Now that we have a set of profiles available, we will build the surface of subduction . The output of this procedure will be a new set of profiles and edges that can be used to define the surface of a complex fault modelling the subduction interface earthquakes and to create inslab sources.

This part of the procedure can be completed by running the 

1. Build the surface of the subduction interface using ``create_2pt5_model.py``. The input information in this case is:

    - The name of the folder ``<cs_folder>`` containing the ``cs_`` files created using either the procedure described in the `first approach`_ or `first approach`_ section;
    - The maximum sampling distance along a trace [km];
    - The output folder ``<output_folder>``;

Example::

    > create_2pt5_model.py <cs_folder> <sampl_distance> <output_folder>

The output is a set of interpolated profiles and edges that can be used to create a complex fault source for the OpenQuake engine.  The results of the code ``create_2pt5_model.py`` can be plotted using ``plot_2pt5_model.py``. Example::

    > plot_2pt5_model.py <output_folder> <configuration_file>

where ``<configuration_file>`` is the configuration file used to build the cross-sections.


Classifying an earthquake catalog using the top of the slab surface [incoplete]
*******************************************************************************

The ``create_2pt5_model.py`` code produces a set of profiles and edges describing the geometry of the top of the slab. With this information we can separate the seismicity in an earthquake catalog into a few subsets, each one representing a specific tectonic environment (e.g. `Abrahamson and Shedlock, 1997 <https://pubs.geoscienceworld.org/ssa/srl/article/68/1/9/142158/overview>`__ or `Chen et al., 2017 <https://academic.oup.com/gji/article/213/2/1263/4794950?login=true>`__ ). The procedure required to complete this task is the following:

Creating inslab sources for the OpenQuake Engine [incoplete]
************************************************************

The construction of subduction inslab sources involves the creation of `virtual faults` elongated along the stike of the slab surface and constrained within the slab volume.
    
1. Create a configuration file

.. code-block:: ini

    [main]

    reference_folder = /Users/kjohnson/GEM/Regions/paisl18u/

    profile_sd_topsl = 40.
    edge_sd_topsl = 40.

    sampling = 10.

    float_strike = -0.5
    float_dip = -1.0

    slab_thickness = 70.
    hspa = 20.
    vspa = 20.

    #profile_folder contains: resampled profiles and edges
    profile_folder = ./model/subduction/cs_profiles/kerton/edges_zone1_slab

    # the pickled catalogue has the hmtk format
    catalogue_pickle_fname = ./data/catalogues/locations/PI_cat.p

    # the file with labels identifying earthquakes belonging to a given class
    treg_fname = ./model/catalogue/PI_class_segments.hdf5
    label = slab_kerton1

    # output folder
    out_hdf5_fname = ./tmp/ruptures/ruptures_inslab_kerton_1.hdf5
    #out_hdf5_fname = ./model/subduction/intraslab/tmp/ruptures/ruptures_inslab_kerton_1.hdf5

    # output smoothing folder
    out_hdf5_smoothing_fname = ./tmp/smoothing/smoothing_kerton_1.hdf5
    #out_hdf5_smoothing_fname = ./model/subduction/intraslab/tmp/smoothing/smoothing_kerton_1.hdf5

    # this is a lists 
    dips = [45, 135]

    # this is a dictionary
    #aspect_ratios = {2.0: 0.5, 5.0: 0.3, 10.0: 0.2}
    aspect_ratios = {2.0: 0.4, 3.0: 0.3, 6.0: 0.2, 8.0: 0.1}
    #aspect_ratios = {1.0: 0.4, 3.0: 0.3, 6.0: 0.2, 8.0: 0.1}

    # this is a dictionary
    uniform_fraction = 1.0

    # magnitude scaling relationship 
    mag_scaling_relation = StrasserIntraslab

    # MFD
    agr = 5.945
    bgr = 1.057
    mmin = 6.5
    mmax = 7.80

